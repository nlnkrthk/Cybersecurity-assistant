"""
Supervisor graph of the incident-response assistant.

    START -> input_guard -> supervisor -> [agents in route order] -> final_answer -> output_guard -> [judge] -> END
                  |                               +-> ask_follow_up -> output_guard  (key metrics missing)
                  +-> END  (blocked: prompt injection or harmful request; fixed reply)

The judge node (LLM-as-Judge) runs only when the caller asks for it (judge=True or JUDGE_MODE=on)
and the turn produced a real answer (not blocked, out of scope or a follow-up question).

The supervisor picks a route for the latest user message; each route maps to an
ordered list of agents (agents.state.ROUTE_PLANS). Agents write findings into the
shared state; the final answer node turns all findings into one reply. Conversations
are kept per thread_id by the checkpointer, so a follow-up answer resumes the analysis.
"""
import json
import logging
import re
import uuid
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from agents.guardrails import BLOCKED_MESSAGE_PLACEHOLDER, check_input, check_output
from agents.judge import judge_answer, last_question, should_judge
from agents.prompts import FINAL_ANSWER_PROMPT, ROUTER_PROMPT
from agents.specialists import AGENT_TITLES, format_history, make_agent_node
from agents.trace import emit, now_ms
from agents.state import (
    CLASSIFICATION,
    INCIDENT_SEARCH,
    RESPONSE,
    RISK_PRIORITY,
    ROUTE_PLANS,
    AssistantState,
)

logger = logging.getLogger(__name__)

AGENTS = (CLASSIFICATION, RISK_PRIORITY, INCIDENT_SEARCH, RESPONSE)


class RouteDecision(BaseModel):
    """Route for the latest user message."""

    route: Literal["classify", "risk", "search", "recommend", "investigate", "general", "out_of_scope"] = Field(
        description="The route that best fits the latest user message."
    )
    reason: str = Field(description="One short sentence explaining the choice.")


# ============================================================================
# Nodes
# ============================================================================
def input_guard(state: AssistantState) -> Dict[str, Any]:
    """Block injections / harmful requests and redact secrets before any LLM sees the message."""
    started = now_ms()
    trace: List[Dict[str, Any]] = []  # first node of the turn: start a fresh trace
    message = state["messages"][-1]
    check = check_input(message.text)
    update: Dict[str, Any] = {
        "trace": trace, "guardrails": {"input": check.to_dict()}, "judgement": None, "turn_id": uuid.uuid4().hex,
    }

    if check.blocked:
        # Keep the blocked text out of the history that later prompts include.
        placeholder = BLOCKED_MESSAGE_PLACEHOLDER.format(category=check.category)
        update.update({
            "messages": [HumanMessage(placeholder, id=message.id), AIMessage(check.reply)],
            "route": "blocked",
            "route_reason": f"Input guardrail: {check.category} ({check.matched})",
            "plan": [],
            "completed": [],
            "findings": {},
            "evidence": [],
            "flow_args": None,
            "final_answer": check.reply,
        })
    elif check.text != message.text:
        update["messages"] = [HumanMessage(check.text, id=message.id)]  # same id: replaces the message

    emit(trace, {"type": "guard_input", **check.to_dict(), "elapsed_ms": round(now_ms() - started)})
    return update


def after_input_guard(state: AssistantState) -> str:
    return END if state.get("guardrails", {}).get("input", {}).get("action") == "blocked" else "supervisor"


def output_guard(state: AssistantState) -> Dict[str, Any]:
    """Verify the reply against the tool evidence, fix what can be fixed, then add it to the conversation."""
    started = now_ms()
    trace = list(state.get("trace", []))
    check = check_output(
        state.get("final_answer", ""),
        evidence=state.get("evidence") or [],
        context_texts=[m.text for m in state.get("messages", [])],
    )
    emit(trace, {"type": "guard_output", "issues": check.issues, "elapsed_ms": round(now_ms() - started)})
    return {
        "final_answer": check.text,
        "messages": [AIMessage(check.text)],
        "trace": trace,
        "guardrails": {**(state.get("guardrails") or {}), "output": check.to_dict()},
    }


def after_output_guard(state: AssistantState) -> str:
    wanted = state.get("judge_requested") and should_judge(state.get("route"), bool(state.get("follow_up")))
    return "judge" if wanted else END


def make_judge_node(get_judge_llm):
    def judge(state: AssistantState) -> Dict[str, Any]:
        started = now_ms()
        trace = list(state.get("trace", []))
        result = judge_answer(
            get_judge_llm(),
            question=last_question(state.get("messages", [])),
            answer=state.get("final_answer", ""),
            route=state.get("route"),
            findings=state.get("findings") or {},
            evidence=state.get("evidence") or [],
            guard_issues=(state.get("guardrails") or {}).get("output", {}).get("issues"),
        )
        emit(trace, {
            "type": "judge",
            "verdict": result["verdict"],
            "overall": result["overall"],
            "scores": {c: result["scores"][c] for c in result["applicable"]},
            "unsupported_claims": len(result["unsupported_claims"]),
            "error": result.get("error"),
            "elapsed_ms": round(now_ms() - started),
        })
        return {"judgement": result, "trace": trace}

    return judge


ROUTER_JSON_FORMAT = (
    '\n\nReply with ONLY a JSON object: {"route": "<classify|risk|search|recommend|investigate|general|'
    'out_of_scope>", "reason": "<one short sentence>"}'
)


def _is_refusal(message) -> bool:
    metadata = getattr(message, "response_metadata", None) or {}
    return metadata.get("stop_reason") == "refusal" or metadata.get("finish_reason") == "content_filter"


def _parse_route(message) -> Optional[RouteDecision]:
    match = re.search(r"\{.*\}", getattr(message, "text", "") or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        return RouteDecision.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError):
        return None


def route_message(llm, messages) -> RouteDecision:
    """Ask the model for a route; never raises.

    1) A plain request with JSON instructions, parsed here: works the same on every provider
       (guided json_schema decoding sometimes ran away to the token limit on gpt-oss).
    2) If that reply is not a valid decision, structured output (json_schema; forced tool
       calling is not supported for structured output on current Claude models).

    A model refusal (e.g. a request to write malware) becomes out_of_scope; any other failure
    falls back to general.
    """
    system, *rest = messages
    try:
        reply = llm.invoke([SystemMessage(system.content + ROUTER_JSON_FORMAT), *rest])
    except Exception as exc:
        logger.warning(f"Router call failed ({exc}); falling back to 'general'")
        return RouteDecision(route="general", reason="Router unavailable; answered as a general question.")
    if _is_refusal(reply):
        return RouteDecision(route="out_of_scope", reason="The model declined the request.")
    decision = _parse_route(reply)
    if decision is not None:
        return decision

    try:
        out = llm.with_structured_output(RouteDecision, method="json_schema", include_raw=True).invoke(messages)
    except Exception as exc:
        logger.warning(f"Router fallback failed ({exc}); falling back to 'general'")
        return RouteDecision(route="general", reason="Could not determine a route; answered as a general question.")
    if out.get("parsed") is not None:
        return out["parsed"]
    if out.get("raw") is not None and _is_refusal(out["raw"]):
        return RouteDecision(route="out_of_scope", reason="The model declined the request.")
    logger.warning(f"Router returned no valid decision ({out.get('parsing_error')}); falling back to 'general'")
    return RouteDecision(route="general", reason="Could not determine a route; answered as a general question.")


def make_supervisor(llm):
    router = llm

    def supervisor(state: AssistantState) -> Dict[str, Any]:
        messages = state["messages"]
        context = []
        if len(messages) > 1:
            context.append(f"Conversation so far:\n{format_history(messages[:-1])}")
        pending = None
        if state.get("follow_up"):
            pending = {"question": state["follow_up"], "route": state.get("route")}
            context.append(
                f"The previous assistant turn (route '{pending['route']}') asked for missing metrics: "
                f"{pending['question']!r}"
            )
        context.append(f"Latest user message:\n{messages[-1].text}")

        started = now_ms()
        decision = route_message(router, [SystemMessage(ROUTER_PROMPT), HumanMessage("\n\n".join(context))])
        logger.info(f"Route: {decision.route} ({decision.reason})")
        trace = list(state.get("trace", []))
        emit(trace, {
            "type": "route",
            "route": decision.route,
            "reason": decision.reason,
            "plan": ROUTE_PLANS[decision.route],
            "elapsed_ms": round(now_ms() - started),
        })
        return {
            "trace": trace,
            "route": decision.route,
            "route_reason": decision.reason,
            "plan": ROUTE_PLANS[decision.route],
            # Reset per-turn fields; keep the last analysed flow for references like "its risk".
            "completed": [],
            "findings": {},
            "evidence": [],
            "follow_up": None,
            "pending_follow_up": pending,
            "previous_flow_args": state.get("flow_args") or state.get("previous_flow_args"),
            "flow_args": None,
            "final_answer": "",
        }

    return supervisor


OUT_OF_SCOPE_REPLY = (
    "I can't help with that. I'm a network incident-response assistant: I can classify network flows, "
    "assess their intrusion risk and priority, find similar historical incidents, and recommend "
    "investigation and response steps."
)
EMPTY_REPLY_FALLBACK = (
    "I couldn't produce a complete answer this time. Here are the raw findings:\n\n{findings}"
)


def make_final_answer(llm):
    def final_answer(state: AssistantState) -> Dict[str, Any]:
        started = now_ms()
        trace = list(state.get("trace", []))
        findings = state.get("findings") or {}

        def done(text: str) -> Dict[str, Any]:
            emit(trace, {"type": "final", "findings": len(findings), "elapsed_ms": round(now_ms() - started)})
            return {"final_answer": text, "trace": trace}  # output_guard adds it to the conversation

        if state.get("route") == "out_of_scope":
            return done(OUT_OF_SCOPE_REPLY)

        messages = state["messages"]
        parts = []
        if len(messages) > 1:
            parts.append(f"Conversation so far:\n{format_history(messages[:-1])}")
        parts.append(f"Latest user message:\n{messages[-1].text}")
        parts.append(f"Route: {state.get('route')}")
        if findings:
            parts.append(
                "Findings:\n\n" + "\n\n".join(f"[{AGENT_TITLES[a]}]\n{text}" for a, text in findings.items())
            )
        else:
            parts.append("Findings: none (no specialist agent was needed).")
        parts.append("Write the reply to the user now.")

        reply = llm.invoke([SystemMessage(FINAL_ANSWER_PROMPT), HumanMessage("\n\n".join(parts))])
        text = reply.text.strip()
        if not text:
            text = EMPTY_REPLY_FALLBACK.format(
                findings="\n\n".join(findings.values()) or "none"
            ) if findings else OUT_OF_SCOPE_REPLY
        return done(text)

    return final_answer


def ask_follow_up(state: AssistantState) -> Dict[str, Any]:
    question = state["follow_up"]
    trace = list(state.get("trace", []))
    emit(trace, {"type": "follow_up", "question": question})
    return {"final_answer": question, "trace": trace}


def next_step(state: AssistantState) -> str:
    if state.get("follow_up"):
        return "ask_follow_up"
    done = set(state.get("completed", []))
    remaining = [agent for agent in state.get("plan", []) if agent not in done]
    return remaining[0] if remaining else "final_answer"


# ============================================================================
# Graph
# ============================================================================
def build_assistant(llm=None, checkpointer=None, judge_llm=None):
    """Compile the assistant graph.

    Args:
        llm: chat model for the supervisor, agents and final answer (default: from .env).
        checkpointer: conversation store (default: in memory).
        judge_llm: model for LLM-as-Judge. Default: llm when one is given, else the judge
            model from .env (JUDGE_PROVIDER / JUDGE_MODEL), created only when first needed.
    """
    if llm is None:
        from agents.llm import get_chat_model

        llm = get_chat_model()
        default_judge = None
    else:
        default_judge = llm

    def get_judge_llm():
        if judge_llm is not None:
            return judge_llm
        if default_judge is not None:
            return default_judge
        from agents.llm import get_judge_model

        return get_judge_model()

    graph = StateGraph(AssistantState)
    graph.add_node("input_guard", input_guard)
    graph.add_node("supervisor", make_supervisor(llm))
    for agent in AGENTS:
        graph.add_node(agent, make_agent_node(llm, agent))
    graph.add_node("final_answer", make_final_answer(llm))
    graph.add_node("ask_follow_up", ask_follow_up)
    graph.add_node("output_guard", output_guard)
    graph.add_node("judge", make_judge_node(get_judge_llm))

    targets = [*AGENTS, "final_answer", "ask_follow_up"]
    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", after_input_guard, ["supervisor", END])
    graph.add_conditional_edges("supervisor", next_step, targets)
    for agent in AGENTS:
        graph.add_conditional_edges(agent, next_step, targets)
    graph.add_edge("final_answer", "output_guard")
    graph.add_edge("ask_follow_up", "output_guard")
    graph.add_conditional_edges("output_guard", after_output_guard, ["judge", END])
    graph.add_edge("judge", END)

    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


class Assistant:
    """Convenience wrapper: ask questions in a conversation identified by thread_id."""

    def __init__(self, llm=None, checkpointer=None, judge_llm=None):
        self.graph = build_assistant(llm=llm, checkpointer=checkpointer, judge_llm=judge_llm)
        self._llm, self._judge_llm = llm, judge_llm

    @staticmethod
    def new_thread_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _prepare(
        message: str, thread_id: Optional[str], judge: Optional[bool] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        if not message or not message.strip():
            raise ValueError("Message must be a non-empty string")
        thread_id = thread_id or Assistant.new_thread_id()
        if judge is None:
            from llm.settings import judge_mode

            judge = judge_mode() == "on"
        inputs = {"messages": [HumanMessage(message.strip())], "judge_requested": bool(judge)}
        return inputs, {"configurable": {"thread_id": thread_id}}, thread_id

    def ask(self, message: str, thread_id: Optional[str] = None, judge: Optional[bool] = None) -> Dict[str, Any]:
        """Answer one message; the result includes the full agent trace.

        judge: grade the answer with LLM-as-Judge (None = JUDGE_MODE from .env).
        """
        inputs, config, thread_id = self._prepare(message, thread_id, judge)
        return self._result(self.graph.invoke(inputs, config=config), thread_id)

    def stream(
        self, message: str, thread_id: Optional[str] = None, judge: Optional[bool] = None
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Answer one message, yielding ("event", trace_event) live and finally ("result", result)."""
        inputs, config, thread_id = self._prepare(message, thread_id, judge)
        state: Dict[str, Any] = {}
        for mode, chunk in self.graph.stream(inputs, config=config, stream_mode=["custom", "values"]):
            if mode == "custom":
                yield "event", chunk
            else:
                state = chunk
        yield "result", self._result(state, thread_id)

    @staticmethod
    def _result(state: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
        return {
            "thread_id": thread_id,
            "answer": state.get("final_answer", ""),
            "route": state.get("route"),
            "route_reason": state.get("route_reason"),
            "agents_run": state.get("completed", []),
            "needs_follow_up": bool(state.get("follow_up")),
            "findings": state.get("findings", {}),
            "evidence": state.get("evidence", []),
            "flow_args": state.get("flow_args"),
            "trace": state.get("trace", []),
            "guardrails": state.get("guardrails", {}),
            "judgement": state.get("judgement"),
            "turn_id": state.get("turn_id"),
        }

    def judge_llm(self):
        """The model used for LLM-as-Judge (see build_assistant)."""
        if self._judge_llm is not None:
            return self._judge_llm
        if self._llm is not None:
            return self._llm
        from agents.llm import get_judge_model

        return get_judge_model()

    def turn_state(self, thread_id: str, turn_id: Optional[str] = None) -> Dict[str, Any]:
        """State at the end of one turn: the latest turn, or the turn with this turn_id.

        Raises KeyError if the conversation or the turn does not exist.
        """
        config = {"configurable": {"thread_id": thread_id}}
        latest = self.graph.get_state(config).values
        if not latest.get("messages"):
            raise KeyError(f"Conversation {thread_id} not found")
        if turn_id is None or latest.get("turn_id") == turn_id:
            return latest
        # History is newest first: the first snapshot with this turn_id is the end of that turn.
        for snapshot in self.graph.get_state_history(config):
            if snapshot.values.get("turn_id") == turn_id and not snapshot.next:
                return snapshot.values
        raise KeyError(f"Turn {turn_id} not found in conversation {thread_id}")

    def judge_last_turn(self, thread_id: str) -> Dict[str, Any]:
        """Grade the latest answer of a conversation (also when it was not judged at the time)."""
        state = self.graph.get_state({"configurable": {"thread_id": thread_id}}).values
        if not state.get("messages"):
            raise KeyError(f"Conversation {thread_id} not found")
        if not should_judge(state.get("route"), bool(state.get("follow_up"))):
            raise ValueError(f"The last turn (route {state.get('route')!r}) has no answer to judge.")
        return judge_answer(
            self.judge_llm(),
            question=last_question(state["messages"]),
            answer=state.get("final_answer", ""),
            route=state.get("route"),
            findings=state.get("findings") or {},
            evidence=state.get("evidence") or [],
            guard_issues=(state.get("guardrails") or {}).get("output", {}).get("issues"),
        )
