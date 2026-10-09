"""
The four specialist agents: Classification, Risk & Priority, Incident Search, Response.

Each agent is an LLM with its own system prompt and tool set running a small
tool-calling loop. It reads the conversation and the findings of earlier agents
in the same turn, and returns its own findings, the tool evidence, the flow it
analysed, and a follow-up question when the user must supply missing metrics.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agents.prompts import AGENT_PROMPTS
from agents.state import AssistantState
from agents.tools import TOOLS_BY_AGENT
from agents.tools.common import error_result
from agents.trace import emit, now_ms, summarize_tool_result

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6
HISTORY_MESSAGES = 8
QUESTION_PREFIX = "QUESTION:"

# Tools whose arguments describe the flow being analysed (shared with later agents).
FLOW_TOOLS = {"classify_attack_type", "assess_intrusion_risk", "explain_flow", "prioritize_flow"}

AGENT_TITLES = {
    "classification": "Classification agent",
    "risk_priority": "Risk & Priority agent",
    "incident_search": "Incident Search agent",
    "response": "Response agent",
}


def format_history(messages: List[BaseMessage], limit: int = HISTORY_MESSAGES) -> str:
    lines = []
    for message in messages[-limit:]:
        role = "User" if message.type == "human" else "Assistant"
        lines.append(f"{role}: {message.text}")
    return "\n".join(lines)


def build_task(state: AssistantState) -> str:
    """The single user-turn message an agent receives: conversation, request and earlier findings."""
    messages = state.get("messages", [])
    history, latest = messages[:-1], messages[-1].text if messages else ""
    parts = []
    if history:
        parts.append(f"Conversation so far:\n{format_history(history)}")
    pending = state.get("pending_follow_up")
    if pending:
        parts.append(
            "The previous assistant turn asked the user for missing metrics: "
            f"{pending['question']!r}. The latest message is likely the answer; combine it with the "
            "metrics the user gave earlier in the conversation."
        )
    if state.get("flow_args"):
        parts.append(f"Flow already identified this turn (tool arguments): {json.dumps(state['flow_args'])}")
    elif state.get("previous_flow_args"):
        parts.append(
            "Latest flow analysed earlier in this conversation (tool arguments): "
            f"{json.dumps(state['previous_flow_args'])}"
        )
    parts.append(f"Latest user message:\n{latest}")
    findings = state.get("findings") or {}
    if findings:
        earlier = "\n\n".join(f"[{AGENT_TITLES[name]}]\n{text}" for name, text in findings.items())
        parts.append(f"Findings from other agents this turn:\n{earlier}")
    parts.append("Do your part of the analysis now.")
    return "\n\n".join(parts)


def _flow_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in args.items() if v is not None}


def run_specialist(llm, agent: str, state: AssistantState, max_rounds: int = MAX_TOOL_ROUNDS) -> Dict[str, Any]:
    """Run one agent's tool-calling loop and return its contribution to the state."""
    tools = TOOLS_BY_AGENT[agent]
    tools_by_name = {t.name: t for t in tools}
    model = llm.bind_tools(tools)

    conversation: List[BaseMessage] = [SystemMessage(AGENT_PROMPTS[agent]), HumanMessage(build_task(state))]
    evidence: List[Dict[str, Any]] = []
    flow_args: Optional[Dict[str, Any]] = None
    follow_up: Optional[str] = None
    reply: Optional[AIMessage] = None
    trace: List[Dict[str, Any]] = list(state.get("trace", []))
    started, llm_calls = now_ms(), 0
    emit(trace, {"type": "agent_start", "agent": agent})

    for _ in range(max_rounds):
        reply = model.invoke(conversation)
        llm_calls += 1
        conversation.append(reply)
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            tool = tools_by_name.get(call["name"])
            emit(trace, {"type": "tool_call", "agent": agent, "tool": call["name"], "args": call["args"]})
            tool_started = now_ms()
            if tool is None:
                result = error_result(f"Unknown tool {call['name']!r}; available: {sorted(tools_by_name)}")
            else:
                result = tool.invoke(call["args"])
            emit(trace, {
                "type": "tool_result",
                "agent": agent,
                "tool": call["name"],
                "args": call["args"],
                "status": result.get("status"),
                "summary": summarize_tool_result(call["name"], result),
                "elapsed_ms": round(now_ms() - tool_started),
            })
            evidence.append({"agent": agent, "tool": call["name"], "args": call["args"], "result": result})
            if call["name"] in FLOW_TOOLS and result.get("status") in {"ok", "needs_follow_up"}:
                flow_args = _flow_args(call["args"])
            if result.get("status") == "needs_follow_up" and not follow_up:
                follow_up = result.get("follow_up_question")
            conversation.append(
                ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"], name=call["name"])
            )
        if follow_up:
            break
    else:
        # Tool budget used up: ask for the findings without further tool use.
        conversation.append(HumanMessage("Stop calling tools and give your findings now."))
        reply = model.invoke(conversation)
        llm_calls += 1

    text = (reply.text if reply is not None else "").strip()
    if not follow_up and text.startswith(QUESTION_PREFIX):
        follow_up = text[len(QUESTION_PREFIX):].strip()

    emit(trace, {
        "type": "agent_end",
        "agent": agent,
        "outcome": "follow_up" if follow_up else "findings",
        "llm_calls": llm_calls,
        "tool_calls": len(evidence),
        "elapsed_ms": round(now_ms() - started),
    })
    update: Dict[str, Any] = {
        "completed": [*state.get("completed", []), agent],
        "evidence": [*state.get("evidence", []), *evidence],
        "trace": trace,
    }
    if follow_up:
        update["follow_up"] = follow_up
    else:
        update["findings"] = {**(state.get("findings") or {}), agent: text}
    if flow_args:
        update["flow_args"] = flow_args
    return update


def make_agent_node(llm, agent: str):
    def node(state: AssistantState) -> Dict[str, Any]:
        logger.info(f"Running {agent} agent")
        return run_specialist(llm, agent, state)

    node.__name__ = f"{agent}_node"
    return node
