"""
Agent trace: which route was chosen, which agents ran, which tools they called and
what each tool returned (one line), with timings.

Events are appended to the turn's trace (returned with every answer) and also sent
live through LangGraph's custom stream, so callers can print them as they happen.

Event types:
    guard_input   input guardrail: action (allowed | redacted | blocked), category, matched, redactions
    route         supervisor decision: route, reason, plan
    agent_start   an agent begins
    tool_call     an agent calls a tool: tool, args
    tool_result   the tool returned: status, summary, elapsed_ms
    agent_end     an agent finished: outcome (findings | follow_up), llm_calls, tool_calls, elapsed_ms
    follow_up     the turn ends with a question to the user
    final         the final answer node finished: elapsed_ms
    guard_output  output guardrail: issues found and fixed (empty list = passed)
    judge         LLM-as-Judge: verdict, overall score, per-criterion scores, unsupported claims
"""
import json
import time
from typing import Any, Callable, Dict, List, Optional

from langgraph.config import get_stream_writer

AGENT_NAMES = {
    "classification": "Classification agent",
    "risk_priority": "Risk & Priority agent",
    "incident_search": "Incident Search agent",
    "response": "Response agent",
}


def now_ms() -> float:
    return time.perf_counter() * 1000


def emit(trace: List[Dict[str, Any]], event: Dict[str, Any]) -> None:
    """Record an event and stream it live when running inside the graph."""
    trace.append(event)
    try:
        get_stream_writer()(event)
    except RuntimeError:
        pass  # called outside a graph run (e.g. a unit test of one node)


# ============================================================================
# One-line tool result summaries
# ============================================================================
def _pct(value: Optional[float]) -> str:
    return "?" if value is None else f"{value:.0%}"


def summarize_tool_result(tool: str, result: Dict[str, Any]) -> str:
    status = result.get("status")
    if status == "error":
        return f"error: {result.get('error', '')[:150]}"
    if status == "needs_follow_up":
        missing = ", ".join(result.get("missing_key_features", [])) or "key metrics"
        return f"needs follow-up: missing {missing}"

    incident = result.get("incident")
    prefix = f"{incident['incident_id']}: " if incident else ""
    quality = result.get("input", {}).get("input_quality")
    note = f" [input {quality}]" if quality and quality != "full" else ""

    if tool == "classify_attack_type":
        attack = result["attack_type"]
        return f"{prefix}{attack['prediction']} ({_pct(attack['confidence'])}){note}"
    if tool == "assess_intrusion_risk":
        risk, anomaly = result["risk"], result["anomaly"]
        return f"{prefix}risk {_pct(risk['probability'])} {risk['level']}, anomaly {anomaly['level']}{note}"
    if tool == "explain_flow":
        attack, risk, priority = result["attack_type"], result["risk"], result.get("priority", {})
        return (
            f"{prefix}{attack['prediction']} ({_pct(attack['confidence'])}), risk {risk['level']}, "
            f"anomaly {result['anomaly']['level']}, {priority.get('priority')}{note}"
        )
    if tool == "prioritize_flow":
        priority = result["priority"]
        return f"{prefix}{priority['priority']} {priority['label']} (score {priority['score']}) - {result['attack_type']['prediction']}{note}"
    if tool == "prioritize_flows":
        counts = ", ".join(f"{k} x{v}" for k, v in result.get("priority_counts", {}).items())
        top = result.get("top") or [{}]
        first = top[0].get("incident_id") or f"flow {top[0].get('flow_index')}"
        return f"{result.get('total_flows')} flows ranked ({counts}); most urgent: {first}"
    if tool == "search_similar_incidents":
        incidents = result.get("incidents", [])
        types: Dict[str, int] = {}
        for item in incidents:
            types[item["attack_type"]] = types.get(item["attack_type"], 0) + 1
        mix = ", ".join(f"{k} x{v}" for k, v in types.items()) or "none"
        filters = result.get("filters_applied") or {}
        filter_txt = f" | filters {json.dumps(filters)}" if filters else ""
        return f"{len(incidents)} incidents ({mix}) from {result.get('candidates_after_filters')} candidates{filter_txt}"
    if tool == "get_attack_profile":
        return f"{result['attack_type']} profile (severity {result['severity']})"
    if tool == "list_attack_types":
        return f"{len(result.get('attack_types', []))} attack types"
    if tool == "lookup_incident":
        return f"{result['incident_id']}: historical label {result['historical_label']}, port {result['dst_port']}"
    return str(status)


# ============================================================================
# Console formatting
# ============================================================================
def _format_args(args: Dict[str, Any], limit: int = 110) -> str:
    text = ", ".join(f"{k}={v}" for k, v in args.items() if v not in (None, [], {}))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _secs(event: Dict[str, Any]) -> str:
    ms = event.get("elapsed_ms")
    return "" if ms is None else f"  ({ms / 1000:.1f}s)"


def _describe_issue(issue: Dict[str, Any]) -> str:
    check = issue["check"]
    if check == "unverified_incident_ids":
        return f"unverified incident ID(s) marked: {', '.join(issue['ids'])}"
    if check == "unmatched_percentages":
        return "figure(s) not in tool results flagged: " + ", ".join(f"{v:g}%" for v in issue["values"])
    if check == "priority_mismatch":
        return f"priority {', '.join(issue['stated'])} differs from tools ({', '.join(issue['tools'])}); flagged"
    if check == "dangerous_commands":
        return f"{len(issue['reasons'])} dangerous command(s) removed"
    if check == "confidence_disclaimer":
        return f"disclaimer added ({issue['level']} confidence)"
    if check == "historical_label_mismatch":
        return f"note added: {issue['incident_id']} label {issue['historical_label']} vs prediction {issue['prediction']}"
    if check == "sensitive_data":
        return f"redacted: {', '.join(issue['kinds'])}"
    if check == "system_prompt_leak":
        return "system prompt leak blocked; reply replaced"
    return check


def format_event(event: Dict[str, Any]) -> Optional[str]:
    """One console line per event (tool_call is folded into its tool_result line)."""
    kind = event["type"]
    if kind == "guard_input":
        if event["action"] == "blocked":
            return f"[GUARD-IN]  BLOCKED: {event['category']} (rule: {event['matched']})"
        notes = []
        if event.get("redactions"):
            notes.append(f"redacted: {', '.join(event['redactions'])}")
        if event.get("truncated"):
            notes.append("message trimmed")
        return "[GUARD-IN]  allowed" + (" | " + " | ".join(notes) if notes else "")
    if kind == "judge":
        if event["verdict"] == "error":
            return f"[JUDGE] not scored: {event.get('error')}{_secs(event)}"
        scores = ", ".join(f"{name} {value}" for name, value in event["scores"].items())
        claims = event.get("unsupported_claims") or 0
        note = f" | {claims} unsupported claim(s)" if claims else ""
        return f"[JUDGE] {event['overall']:.1f}/5 {event['verdict']} | {scores}{note}{_secs(event)}"
    if kind == "guard_output":
        issues = event.get("issues") or []
        if not issues:
            return "[GUARD-OUT] passed, no changes"
        return "[GUARD-OUT] " + "; ".join(_describe_issue(i) for i in issues)
    if kind == "route":
        plan = " -> ".join(AGENT_NAMES[a] for a in event["plan"]) or "no agents"
        return (
            f"[SUPERVISOR] route = {event['route']}{_secs(event)}\n"
            f"             reason: {event['reason']}\n"
            f"             plan:   {plan}"
        )
    if kind == "agent_start":
        return f"[AGENT] {AGENT_NAMES[event['agent']]}"
    if kind == "tool_call":
        return None
    if kind == "tool_result":
        return f"    [TOOL] {event['tool']}({_format_args(event['args'])})\n           -> {event['summary']}{_secs(event)}"
    if kind == "agent_end":
        outcome = "asked a follow-up question" if event["outcome"] == "follow_up" else "findings ready"
        return (
            f"    done: {outcome}; {event['llm_calls']} LLM call(s), "
            f"{event['tool_calls']} tool call(s){_secs(event)}"
        )
    if kind == "follow_up":
        return "[FOLLOW-UP] asking the user for missing or corrected information"
    if kind == "final":
        return f"[FINAL ANSWER] written from {event['findings']} agent finding(s){_secs(event)}"
    return None


def print_trace(trace: List[Dict[str, Any]], printer: Callable[[str], None] = print) -> None:
    for event in trace:
        line = format_event(event)
        if line:
            printer(line)
