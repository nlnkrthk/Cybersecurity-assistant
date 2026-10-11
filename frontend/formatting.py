"""
Pure helpers for the front end (no Streamlit calls, so they are unit-tested):
flow fields, trace-event lines, badges and result summaries.
"""
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# (key, label as returned by GET /incidents/{id}, unit hint) in model feature order.
FLOW_FIELDS: List[Tuple[str, str, str]] = [
    ("flow_duration", "Flow duration", "µs, or e.g. '2 s'"),
    ("total_fwd_packets", "Forward packets", "packets"),
    ("total_bwd_packets", "Backward packets", "packets"),
    ("total_len_fwd", "Forward bytes", "bytes, or e.g. '1.2 KB'"),
    ("total_len_bwd", "Backward bytes", "bytes, or e.g. '4 KB'"),
    ("flow_packets_s", "Flow packet rate", "packets/s"),
    ("fwd_packets_s", "Forward packet rate", "packets/s"),
    ("bwd_packets_s", "Backward packet rate", "packets/s"),
    ("fwd_packet_length_mean", "Mean forward packet size", "bytes"),
    ("bwd_packet_length_mean", "Mean backward packet size", "bytes"),
    ("min_packet_length", "Minimum packet size", "bytes"),
    ("max_packet_length", "Maximum packet size", "bytes"),
    ("packet_length_mean", "Mean packet size", "bytes"),
    ("packet_length_std", "Packet size std. dev.", "bytes"),
    ("syn_flag_count", "SYN flag", "count"),
    ("ack_flag_count", "ACK flag", "count"),
    ("rst_flag_count", "RST flag", "count"),
    ("psh_flag_count", "PSH flag", "count"),
    ("fin_flag_count", "FIN flag", "count"),
    ("down_up_ratio", "Download/upload ratio", "ratio"),
    ("init_win_bytes_forward", "Initial TCP window (forward)", "bytes"),
    ("init_win_bytes_backward", "Initial TCP window (backward)", "bytes"),
    ("active_mean", "Mean active time", "µs"),
    ("idle_mean", "Mean idle time", "µs"),
]
FIELD_GROUPS = {
    "Volume and timing": FLOW_FIELDS[0:8],
    "Packet sizes": FLOW_FIELDS[8:14],
    "Flags, TCP window and activity": FLOW_FIELDS[14:24],
}
LABEL_TO_KEY = {label: key for key, label, _ in FLOW_FIELDS}

# One historical incident per attack type (from data/processed/incidents_100k.parquet).
EXAMPLE_INCIDENTS = {
    "BENIGN": "INC-000002",
    "Bot": "INC-000110",
    "DDoS": "INC-000006",
    "DoS GoldenEye": "INC-000012",
    "DoS Hulk": "INC-000001",
    "DoS Slowhttptest": "INC-000129",
    "DoS slowloris": "INC-000015",
    "FTP-Patator": "INC-000031",
    "Heartbleed": "INC-002545",
    "Infiltration": "INC-004393",
    "PortScan": "INC-000005",
    "SSH-Patator": "INC-000035",
    "Web Attack - Brute Force": "INC-000056",
    "Web Attack - Sql Injection": "INC-011119",
    "Web Attack - XSS": "INC-000018",
}

PRIORITY_COLORS = {"P1": "#DC2626", "P2": "#EA580C", "P3": "#CA8A04", "P4": "#16A34A"}
PRIORITY_INFO = {  # label, action
    "P1": ("Critical", "Investigate now"),
    "P2": ("High", "Investigate today"),
    "P3": ("Medium", "Review when possible"),
    "P4": ("Low", "Monitor"),
}
ROUTE_LABELS = {
    "classify": "Attack classification", "risk": "Risk & priority", "search": "Incident search",
    "recommend": "Response advice", "investigate": "Full investigation", "general": "General question",
    "out_of_scope": "Out of scope", "blocked": "Blocked",
}
AGENT_NAMES = {
    "classification": "Classification agent",
    "risk_priority": "Risk & Priority agent",
    "incident_search": "Incident Search agent",
    "response": "Response agent",
}
FLOW_TOOLS = {"explain_flow", "classify_attack_type", "assess_intrusion_risk", "prioritize_flow", "lookup_incident"}
INCIDENT_ID = re.compile(r"^INC-\d{6}$", re.IGNORECASE)






def route_badge(route: Optional[str]) -> str:
    color = "red" if route == "blocked" else "blue"
    return f":{color}-badge[{ROUTE_LABELS.get(route or '', route or 'unknown')}]"


def pct(value: Optional[float]) -> str:
    return "–" if value is None else f"{value:.0%}"






def describe_issue(issue: Dict[str, Any]) -> str:
    """One readable line for an output-guardrail issue."""
    check = issue.get("check")
    if check == "unverified_incident_ids":
        return f"Incident ID(s) not found in the tool results were marked unverified: {', '.join(issue.get('ids', []))}"
    if check == "unmatched_percentages":
        return "Figure(s) not found in the tool results were flagged: " + ", ".join(
            f"{v:g}%" for v in issue.get("values", []))
    if check == "priority_mismatch":
        return (f"Stated priority {', '.join(issue.get('stated', []))} differs from the computed "
                f"{', '.join(issue.get('tools', []))}; flagged")
    if check == "dangerous_commands":
        return f"{len(issue.get('reasons', []))} dangerous command(s) removed"
    if check == "confidence_disclaimer":
        return f"Disclaimer added ({issue.get('level')} confidence)"
    if check == "historical_label_mismatch":
        return (f"Note added: {issue.get('incident_id')} was historically {issue.get('historical_label')}, "
                f"the model predicts {issue.get('prediction')}")
    if check == "sensitive_data":
        return f"Sensitive data redacted: {', '.join(issue.get('kinds', []))}"
    if check == "system_prompt_leak":
        return "Possible system-prompt leak blocked; reply replaced"
    return str(check)


AGENT_ROLES = {
    "classification": "Identifies the attack type",
    "risk_priority": "Scores intrusion risk, anomaly and priority",
    "incident_search": "Finds similar past incidents",
    "response": "Recommends investigation and response steps",
}
TOOL_LABELS = {
    "lookup_incident": "Looked up the incident",
    "classify_attack_type": "Classified the attack type",
    "explain_flow": "Analysed the flow",
    "assess_intrusion_risk": "Assessed intrusion risk",
    "prioritize_flow": "Set the priority",
    "prioritize_flows": "Ranked the flows",
    "search_similar_incidents": "Searched past incidents",
    "get_attack_profile": "Loaded the attack-type profile",
    "list_attack_types": "Listed the attack types",
}
GUARD_CATEGORIES = {
    "prompt_injection": "attempt to change the assistant's instructions",
    "harmful_request": "request for harmful content",
    "too_long": "message too long",
}


def clean_summary(text: str) -> str:
    """Tool summary for people: '| filters {"label": "SSH-Patator"}' -> '· filtered by label SSH-Patator'."""
    def filters(match: "re.Match") -> str:
        try:
            parts = json.loads(match.group(1))
        except ValueError:
            return match.group(0)
        names = {"label": "attack type", "dst_port": "port"}
        return "· filtered by " + ", ".join(f"{names.get(k, k)} {v}" for k, v in parts.items())

    text = re.sub(r"\|\s*filters\s*(\{.*\})", filters, str(text or ""))
    return text.replace("[input reduced]", "(partial input)").replace("[input low]", "(very partial input)")


def _step(key: str, title: str, role: str = "") -> Dict[str, Any]:
    return {"key": key, "title": title, "role": role, "status": "pending", "detail": "", "elapsed_ms": None,
            "tools": []}


def pipeline_steps(events: List[Dict[str, Any]], judge: bool = False, running: bool = False) -> List[Dict[str, Any]]:
    """Turn agent-trace events into the steps shown in the progress tracker.

    Step status: pending | running | done | warning | blocked | skipped. While `running`, the first
    step not yet finished is shown as running; when the turn is over, unfinished steps are skipped.
    """
    steps = [_step("guard_in", "Safety check", "Screens the question for unsafe requests and secrets"),
             _step("route", "Planning", "Decides which specialist agents are needed"),
             _step("final", "Writing the answer", "Combines the agents' findings into one reply"),
             _step("guard_out", "Answer check", "Verifies incident IDs, figures, priorities and commands")]
    if judge:
        steps.append(_step("judge", "Quality grading", "LLM-as-Judge scores the answer 1–5"))
    by_key = {s["key"]: s for s in steps}
    stopped = False

    for event in events:
        kind = event.get("type")
        if kind == "guard_input":
            step = by_key["guard_in"]
            if event.get("action") == "blocked":
                category = event.get("category") or ""
                step.update(status="blocked", detail=f"Blocked: {GUARD_CATEGORIES.get(category, category.replace('_', ' '))}")
                stopped = True
            elif event.get("redactions"):
                step.update(status="done", detail=f"Passed. Removed before processing: {', '.join(event['redactions'])}")
            else:
                step.update(status="done", detail="Passed")
        elif kind == "route":
            step = by_key["route"]
            plan = event.get("plan") or []
            agents = ", ".join(AGENT_NAMES.get(a, a) for a in plan) or "no specialist agents needed"
            step.update(status="done", elapsed_ms=event.get("elapsed_ms"),
                        detail=f"{ROUTE_LABELS.get(event.get('route'), event.get('route'))}: {agents}")
            step["note"] = event.get("reason", "")
            index = steps.index(by_key["final"])
            for offset, agent in enumerate(plan):
                agent_step = _step(f"agent:{agent}", AGENT_NAMES.get(agent, agent), AGENT_ROLES.get(agent, ""))
                steps.insert(index + offset, agent_step)
                by_key[agent_step["key"]] = agent_step
            if event.get("route") == "out_of_scope":
                stopped = True
        elif kind == "agent_start":
            step = by_key.get(f"agent:{event.get('agent')}")
            if step:
                step["status"] = "running"
        elif kind in ("tool_call", "tool_result"):
            step = by_key.get(f"agent:{event.get('agent')}") or next(
                (s for s in steps if s["key"].startswith("agent:") and s["status"] == "running"), None)
            if step is None:
                continue
            label = TOOL_LABELS.get(event.get("tool"), event.get("tool"))
            if kind == "tool_call":
                step["tools"].append({"tool": event.get("tool"), "label": label, "status": "running", "detail": ""})
            else:
                tool = next((t for t in reversed(step["tools"])
                             if t["tool"] == event.get("tool") and t["status"] == "running"), None)
                if tool is None:
                    tool = {"tool": event.get("tool"), "label": label}
                    step["tools"].append(tool)
                failed = str(event.get("status")) == "error" or str(event.get("summary", "")).startswith("error")
                tool.update(status="warning" if failed else "done", detail=clean_summary(event.get("summary", "")),
                            elapsed_ms=event.get("elapsed_ms"))
        elif kind == "agent_end":
            step = by_key.get(f"agent:{event.get('agent')}")
            if step:
                follow_up = event.get("outcome") == "follow_up"
                calls = event.get("tool_calls", 0)
                step.update(status="warning" if follow_up else "done", elapsed_ms=event.get("elapsed_ms"),
                            detail="Needs more information from you" if follow_up
                            else f"Done · {calls} tool call{'s' if calls != 1 else ''}")
        elif kind == "follow_up":
            by_key["final"].update(status="done", title="Asking you a question",
                                   detail="Some details are missing, so the assistant asks for them")
        elif kind == "final":
            by_key["final"].update(status="done", elapsed_ms=event.get("elapsed_ms"),
                                   detail=f"Written from {event.get('findings', 0)} agent finding(s)")
        elif kind == "guard_output":
            issues = event.get("issues") or []
            by_key["guard_out"].update(
                status="warning" if issues else "done",
                detail="Passed, no changes" if not issues else "Adjusted: " + "; ".join(describe_issue(i) for i in issues))
        elif kind == "judge" and "judge" in by_key:
            if event.get("verdict") == "error":
                by_key["judge"].update(status="warning", detail="Could not be scored")
            else:
                verdict = str(event.get("verdict", "")).replace("_", " ")
                by_key["judge"].update(status="done" if event.get("verdict") == "pass" else "warning",
                                       elapsed_ms=event.get("elapsed_ms"),
                                       detail=f"{event.get('overall', 0):.1f} / 5 · {verdict}")

    if stopped:
        for step in steps:
            if step["status"] == "pending":
                step["status"] = "skipped"
    elif running:
        if not any(s["status"] == "running" for s in steps):
            next_step = next((s for s in steps if s["status"] == "pending"), None)
            if next_step:
                next_step["status"] = "running"
    else:
        for step in steps:
            if step["status"] in ("pending", "running"):
                step["status"] = "skipped"
    return steps


def pipeline_summary(steps: List[Dict[str, Any]]) -> str:
    """One line for the collapsed tracker, e.g. 'Safety check → Planning → 4 agents → Answer check'."""
    agents = sum(s["key"].startswith("agent:") and s["status"] != "skipped" for s in steps)
    tools = sum(len(s["tools"]) for s in steps)
    parts = [f"{agents} agent{'s' if agents != 1 else ''}"] if agents else []
    if tools:
        parts.append(f"{tools} tool call{'s' if tools != 1 else ''}")
    if any(s["status"] == "blocked" for s in steps):
        return "Stopped by the safety check"
    return " · ".join(parts) or "Answered directly"


def has_flow_analysis(result: Dict[str, Any]) -> bool:
    """True when the turn analysed a flow or incident, so an attack-type correction makes sense."""
    return any(item.get("tool") in FLOW_TOOLS and (item.get("result") or {}).get("status") == "ok"
               for item in result.get("evidence") or [])


def predicted_label(result: Dict[str, Any]) -> Optional[str]:
    """The model's attack-type prediction for the turn, if one tool produced it."""
    for item in result.get("evidence") or []:
        attack = (item.get("result") or {}).get("attack_type")
        if isinstance(attack, dict) and attack.get("prediction"):
            return attack["prediction"]
    return None


def guardrail_notices(result: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(input notices, output notices) to show under an answer."""
    guard = result.get("guardrails") or {}
    inputs, outputs = [], []
    check = guard.get("input") or {}
    if check.get("action") == "blocked":
        inputs.append(f"The question was blocked by the input guardrail ({check.get('category')}).")
    elif check.get("redactions"):
        inputs.append(f"Sensitive data was removed from your question before processing: "
                      f"{', '.join(check['redactions'])}.")
    if check.get("truncated"):
        inputs.append("The question was too long and was trimmed.")
    for issue in (guard.get("output") or {}).get("issues") or []:
        if issue.get("check") != "confidence_disclaimer":
            outputs.append(describe_issue(issue))
    return inputs, outputs


def parse_flow_inputs(values: Dict[str, str]) -> Dict[str, Any]:
    """Form text → API flow fields: numbers become floats, unit strings ('2 s', '43k') are passed through."""
    flow: Dict[str, Any] = {}
    for key, raw in values.items():
        text = (raw or "").strip().replace(",", "")
        if not text:
            continue
        try:
            flow[key] = float(text)
        except ValueError:
            flow[key] = (raw or "").strip()
    return flow


def metrics_to_form(metrics: Dict[str, Any]) -> Dict[str, str]:
    """GET /incidents/{id} metrics (display names) → form values keyed by field."""
    form = {}
    for label, value in metrics.items():
        key = LABEL_TO_KEY.get(label)
        if key is None or value is None:
            continue
        form[key] = f"{value:g}" if isinstance(value, (int, float)) else str(value)
    return form


def parse_incident_ids(text: str) -> Tuple[List[str], List[str]]:
    """Split pasted text into (valid IDs, invalid tokens); IDs are upper-cased and de-duplicated."""
    valid, invalid, seen = [], [], set()
    for token in re.split(r"[\s,;]+", text or ""):
        if not token:
            continue
        if INCIDENT_ID.match(token):
            token = token.upper()
            if token not in seen:
                seen.add(token)
                valid.append(token)
        else:
            invalid.append(token)
    return valid, invalid


def shap_rows(reasons: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """SHAP reasons → rows for a bar chart, strongest at the top."""
    rows = [
        {"feature": (r.get("text") or r.get("feature") or "").rsplit(" (", 1)[0],  # "Backward bytes = 7,879,536 bytes"
         "text": r.get("text") or r.get("feature"), "shap": float(r.get("shap") or 0),
         "effect": r.get("effect", ""), "value": r.get("value")}
        for r in reasons or []
    ]
    return sorted(rows, key=lambda r: abs(r["shap"]))


def ranking_rows(top: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch prioritisation results → table rows."""
    rows = []
    for rank, item in enumerate(top or [], start=1):
        index = item.get("flow_index")
        rows.append({
            "rank": rank,
            "flow": item.get("incident_id") or (f"row {index + 1}" if index is not None else f"#{rank}"),
            "priority": item.get("priority") or "needs data",
            "score": item.get("score"),
            "attack_type": item.get("attack_type"),
            "confidence": item.get("confidence"),
            "risk": item.get("risk_probability"),
            "anomaly": item.get("anomaly_level"),
            "historical_label": item.get("historical_label"),
            "input_quality": item.get("input_quality"),
            "reason": item.get("reason") or item.get("error") or item.get("follow_up_question"),
        })
    return rows
