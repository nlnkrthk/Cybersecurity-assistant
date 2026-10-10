"""
Supervisor graph of the incident-response assistant.

    START -> input_guard -> supervisor -> [agents in route order] -> final_answer -> output_guard -> END
                  |                               +-> ask_follow_up -> output_guard  (key metrics missing)
                  +-> END  (blocked: prompt injection or harmful request; fixed reply)

The supervisor picks a route for the latest user message; each route maps to an
ordered list of agents (agents.state.ROUTE_PLANS). Agents write findings into the
shared state; the final answer node turns all findings into one reply. Conversations
are kept per thread_id by the checkpointer, so a follow-up answer resumes the analysis.
"""
import logging
import uuid
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agents.guardrails import BLOCKED_MESSAGE_PLACEHOLDER, check_input, check_output
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
    update: Dict[str, Any] = {"trace": trace, "guardrails": {"input": check.to_dict()}}

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


def route_message(router, messages) -> RouteDecision:
    """Ask the router model for a route; never raises.

    A model refusal (e.g. a request to write malware) becomes out_of_scope; any other
    failure to produce a valid decision falls back to general.
    """
    try:
        out = router.invoke(messages)
    except Exception as exc:
        logger.warning(f"Router call failed ({exc}); falling back to 'general'")
        return RouteDecision(route="general", reason="Router unavailable; answered as a general question.")
    if out.get("parsed") is not None:
        return out["parsed"]
    raw = out.get("raw")
    if raw is not None and raw.response_metadata.get("stop_reason") == "refusal":
        return RouteDecision(route="out_of_scope", reason="The model declined the request.")
    logger.warning(f"Router returned no valid decision ({out.get('parsing_error')}); falling back to 'general'")
    return RouteDecision(route="general", reason="Could not determine a route; answered as a general question.")


def make_supervisor(llm):
    # json_schema: forced tool calling is not supported for structured output on current Claude models.
    router = llm.with_structured_output(RouteDecision, method="json_schema", include_raw=True)

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
def build_assistant(llm=None, checkpointer=None):
    """Compile the assistant graph. Pass a chat model to override the default Claude model."""
    if llm is None:
        from agents.llm import get_chat_model

        llm = get_chat_model()

    graph = StateGraph(AssistantState)
    graph.add_node("input_guard", input_guard)
    graph.add_node("supervisor", make_supervisor(llm))
    for agent in AGENTS:
        graph.add_node(agent, make_agent_node(llm, agent))
    graph.add_node("final_answer", make_final_answer(llm))
    graph.add_node("ask_follow_up", ask_follow_up)
    graph.add_node("output_guard", output_guard)

    targets = [*AGENTS, "final_answer", "ask_follow_up"]
    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", after_input_guard, ["supervisor", END])
    graph.add_conditional_edges("supervisor", next_step, targets)
    for agent in AGENTS:
        graph.add_conditional_edges(agent, next_step, targets)
    graph.add_edge("final_answer", "output_guard")
    graph.add_edge("ask_follow_up", "output_guard")
    graph.add_edge("output_guard", END)

    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


class Assistant:
    """Convenience wrapper: ask questions in a conversation identified by thread_id."""

    def __init__(self, llm=None, checkpointer=None):
        self.graph = build_assistant(llm=llm, checkpointer=checkpointer)

    @staticmethod
    def new_thread_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _prepare(message: str, thread_id: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        if not message or not message.strip():
            raise ValueError("Message must be a non-empty string")
        thread_id = thread_id or Assistant.new_thread_id()
        return {"messages": [HumanMessage(message.strip())]}, {"configurable": {"thread_id": thread_id}}, thread_id

    def ask(self, message: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer one message; the result includes the full agent trace."""
        inputs, config, thread_id = self._prepare(message, thread_id)
        return self._result(self.graph.invoke(inputs, config=config), thread_id)

    def stream(self, message: str, thread_id: Optional[str] = None) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Answer one message, yielding ("event", trace_event) live and finally ("result", result)."""
        inputs, config, thread_id = self._prepare(message, thread_id)
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
        }
