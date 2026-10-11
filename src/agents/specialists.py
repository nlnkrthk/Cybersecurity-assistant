"""
The four specialist agents: Classification, Risk & Priority, Incident Search, Response.

Each agent is an LLM with its own system prompt and tool set running a small
tool-calling loop. It reads the conversation and the findings of earlier agents
in the same turn, and returns its own findings, the tool evidence, the flow it
analysed, and a follow-up question when the user must supply missing metrics.

Agents of the same stage (agents.state.ROUTE_STAGES) run at the same time in
threads; their results are merged in plan order. When one agent answers a
question alone, it writes the reply itself (direct_reply) and the final answer
node passes it through instead of rewriting it.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables.config import ContextThreadPoolExecutor

from agents.prompts import agent_system_prompt
from agents.state import AssistantState, next_stage
from agents.tools import TOOLS_BY_AGENT
from agents.tools.common import error_result
from agents.trace import emit, now_ms, summarize_tool_result

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6
HISTORY_MESSAGES = 8
QUESTION_PREFIX = "QUESTION:"

# Tools whose arguments describe the flow being analysed (shared with later agents).
FLOW_TOOLS = {"classify_attack_type", "assess_intrusion_risk", "explain_flow", "prioritize_flow"}
# Tools whose results include a P1-P4 priority.
PRIORITY_TOOLS = {"explain_flow", "prioritize_flow", "prioritize_flows"}

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


def _has_priority(evidence: List[Dict[str, Any]]) -> bool:
    return any(c.get("tool") in PRIORITY_TOOLS and (c.get("result") or {}).get("status") == "ok" for c in evidence)


def _run_tool(tool, args: Dict[str, Any], agent: str, trace: List[Dict[str, Any]], auto: bool = False) -> Dict[str, Any]:
    marker = {"auto": True} if auto else {}
    emit(trace, {"type": "tool_call", "agent": agent, "tool": tool.name, "args": args, **marker})
    started = now_ms()
    result = tool.invoke(dict(args))
    emit(trace, {
        "type": "tool_result",
        "agent": agent,
        "tool": tool.name,
        "args": args,
        "status": result.get("status"),
        "summary": summarize_tool_result(tool.name, result),
        "elapsed_ms": round(now_ms() - started),
        **marker,
    })
    return result


def _priority_line(result: Dict[str, Any]) -> str:
    priority = result.get("priority") or {}
    line = f"Priority (prioritize_flow): {priority.get('priority')}"
    if priority.get("label"):
        line += f" {priority['label']}"
    if priority.get("reason"):
        line += f" - {priority['reason']}"
    for override in priority.get("overrides_applied") or []:
        line += f"; override: {override}"
    return line


def ensure_priority(model, agent: str, state: AssistantState, conversation: List[BaseMessage],
                    evidence: List[Dict[str, Any]], flow_args: Optional[Dict[str, Any]],
                    trace: List[Dict[str, Any]], tools_by_name: Dict[str, Any], text: str) -> Tuple[str, int]:
    """The Risk & Priority agent must always report a priority for a single flow.

    If the LLM assessed the risk but skipped the prioritiser, run it here with the same flow and
    let the agent rewrite its findings with the result. Returns the findings and the extra LLM calls.
    """
    args = flow_args or state.get("flow_args")
    tool = tools_by_name.get("prioritize_flow")
    if (agent != "risk_priority" or not args or tool is None
            or _has_priority([*state.get("evidence", []), *evidence])):
        return text, 0
    result = _run_tool(tool, args, agent, trace, auto=True)
    evidence.append({"agent": agent, "tool": tool.name, "args": args, "result": result, "auto": True})
    if result.get("status") != "ok":
        return text, 0
    appended = f"{text}\n\n{_priority_line(result)}".strip()
    conversation.append(HumanMessage(
        "The prioritisation tool was run for this flow:\n"
        f"{json.dumps(result, default=str)}\n"
        "Rewrite your final message in the same format so it also states this priority, its reason "
        "and any overrides. "
        "Do not call tools."
    ))
    try:
        reply = model.invoke(conversation)
    except Exception:  # keep the findings and append the priority deterministically
        logger.exception("Rewriting findings with the priority failed")
        return appended, 1
    rewritten = (reply.text or "").strip()
    if reply.tool_calls or not rewritten:
        return appended, 1
    return rewritten, 1


def run_specialist(llm, agent: str, state: AssistantState, max_rounds: int = MAX_TOOL_ROUNDS,
                   direct_reply: bool = False) -> Dict[str, Any]:
    """Run one agent's tool-calling loop and return its contribution to the state.

    direct_reply: the agent works alone and writes the reply to the user instead of findings.
    """
    tools = TOOLS_BY_AGENT[agent]
    tools_by_name = {t.name: t for t in tools}
    model = llm.bind_tools(tools)

    conversation: List[BaseMessage] = [
        SystemMessage(agent_system_prompt(agent, direct_reply)), HumanMessage(build_task(state)),
    ]
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
    if not follow_up:
        text, extra_calls = ensure_priority(model, agent, state, conversation, evidence, flow_args, trace,
                                            tools_by_name, text)
        llm_calls += extra_calls

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


def merge_updates(state: AssistantState, agents: List[str], updates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine the updates of agents that ran in parallel from the same state, in plan order."""
    base_evidence, base_trace = len(state.get("evidence", [])), len(state.get("trace", []))
    merged: Dict[str, Any] = {
        "completed": [*state.get("completed", []), *agents],
        "evidence": list(state.get("evidence", [])),
        "trace": list(state.get("trace", [])),
        "findings": dict(state.get("findings") or {}),
    }
    for agent, update in zip(agents, updates):
        merged["evidence"] += update["evidence"][base_evidence:]
        merged["trace"] += update["trace"][base_trace:]
        if agent in update.get("findings", {}):
            merged["findings"][agent] = update["findings"][agent]
        for key in ("follow_up", "flow_args"):  # the first agent in plan order wins
            if update.get(key) and key not in merged:
                merged[key] = update[key]
    return merged


def run_stage(llm, agents: List[str], state: AssistantState, direct_reply: bool = False) -> Dict[str, Any]:
    """Run one stage: a single agent directly, several agents in parallel threads."""
    if len(agents) == 1:
        return run_specialist(llm, agents[0], state, direct_reply=direct_reply)
    # ContextThreadPoolExecutor copies the context, so live trace events still reach the stream.
    with ContextThreadPoolExecutor(max_workers=len(agents)) as pool:
        updates = list(pool.map(lambda agent: run_specialist(llm, agent, state), agents))
    return merge_updates(state, agents, updates)


def make_agents_node(llm, parallel: bool = True, direct_reply: bool = True):
    """Graph node that runs the next stage of the plan."""
    def agents(state: AssistantState) -> Dict[str, Any]:
        stage = next_stage(state, parallel)
        logger.info(f"Running agents: {', '.join(stage)}")
        alone = direct_reply and len(state.get("plan", [])) == 1
        return run_stage(llm, stage, state, direct_reply=alone)

    return agents
