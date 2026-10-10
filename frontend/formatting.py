"""
Pure helpers for the front end (no Streamlit calls, so they are unit-tested):
flow fields, trace-event lines, badges and result summaries.
"""
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

PRIORITY_COLORS = {"P1": "#d62728", "P2": "#ff7f0e", "P3": "#e6b800", "P4": "#2ca02c"}
PRIORITY_BADGES = {"P1": ":red-badge[P1 Critical]", "P2": ":orange-badge[P2 High]",
                   "P3": ":yellow-badge[P3 Medium]", "P4": ":green-badge[P4 Low]"}
VERDICT_BADGES = {"pass": ":green-badge[✓ pass]", "needs_review": ":orange-badge[needs review]",
                  "fail": ":red-badge[✗ fail]", "error": ":gray-badge[not scored]"}
ROUTE_LABELS = {
    "classify": "Classification", "risk": "Risk & priority", "search": "Incident search",
    "recommend": "Response", "investigate": "Full investigation", "general": "General",
    "blocked": "Blocked",
}
AGENT_NAMES = {
    "classification": "Classification agent",
    "risk_priority": "Risk & Priority agent",
    "incident_search": "Incident Search agent",
    "response": "Response agent",
}
FLOW_TOOLS = {"explain_flow", "classify_attack_type", "assess_intrusion_risk", "prioritize_flow", "lookup_incident"}
INCIDENT_ID = re.compile(r"^INC-\d{6}$", re.IGNORECASE)


def priority_badge(priority: Optional[str]) -> str:
    return PRIORITY_BADGES.get(priority or "", f":gray-badge[{priority or '?'}]")


def verdict_badge(verdict: Optional[str]) -> str:
    return VERDICT_BADGES.get(verdict or "", f":gray-badge[{verdict or '?'}]")


def route_badge(route: Optional[str]) -> str:
    color = "red" if route == "blocked" else "blue"
    return f":{color}-badge[{ROUTE_LABELS.get(route or '', route or 'unknown')}]"


def pct(value: Optional[float]) -> str:
    return "–" if value is None else f"{value:.0%}"


def secs(ms: Optional[float]) -> str:
    return "" if ms is None else f" ({ms / 1000:.1f}s)"


def _args_text(args: Dict[str, Any], limit: int = 90) -> str:
    text = ", ".join(f"{k}={v}" for k, v in (args or {}).items() if v not in (None, [], {}, ""))
    return text if len(text) <= limit else text[: limit - 1] + "…"


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


def trace_line(event: Dict[str, Any]) -> Optional[str]:
    """Markdown line for a live agent-trace event; None for events folded into others."""
    kind = event.get("type")
    if kind == "guard_input":
        if event.get("action") == "blocked":
            return f"🛡️ **Input guardrail:** blocked ({event.get('category')})"
        notes = []
        if event.get("redactions"):
            notes.append(f"redacted {', '.join(event['redactions'])}")
        if event.get("truncated"):
            notes.append("message trimmed")
        return "🛡️ **Input guardrail:** allowed" + (f" — {'; '.join(notes)}" if notes else "")
    if kind == "route":
        plan = " → ".join(AGENT_NAMES.get(a, a) for a in event.get("plan", [])) or "no agents"
        return (f"🧭 **Supervisor:** route `{event.get('route')}`{secs(event.get('elapsed_ms'))} — "
                f"{event.get('reason', '')}  \nPlan: {plan}")
    if kind == "agent_start":
        return f"🤖 **{AGENT_NAMES.get(event.get('agent'), event.get('agent'))}** started"
    if kind == "tool_result":
        return (f"&nbsp;&nbsp;&nbsp;🔧 `{event.get('tool')}({_args_text(event.get('args', {}))})` → "
                f"{event.get('summary', '')}{secs(event.get('elapsed_ms'))}")
    if kind == "agent_end":
        outcome = "asked a follow-up question" if event.get("outcome") == "follow_up" else "findings ready"
        return (f"&nbsp;&nbsp;&nbsp;✅ {outcome}: {event.get('llm_calls', 0)} LLM call(s), "
                f"{event.get('tool_calls', 0)} tool call(s){secs(event.get('elapsed_ms'))}")
    if kind == "follow_up":
        return "❓ Asking you for missing or corrected information"
    if kind == "final":
        return f"📝 **Final answer** written from {event.get('findings', 0)} agent finding(s){secs(event.get('elapsed_ms'))}"
    if kind == "guard_output":
        issues = event.get("issues") or []
        if not issues:
            return "🛡️ **Output guardrail:** passed"
        return "🛡️ **Output guardrail:** " + "; ".join(describe_issue(i) for i in issues)
    if kind == "judge":
        if event.get("verdict") == "error":
            return f"⚖️ **Judge:** not scored ({event.get('error')})"
        return (f"⚖️ **Judge:** {event.get('overall', 0):.1f}/5 {event.get('verdict')}"
                f"{secs(event.get('elapsed_ms'))}")
    return None


def status_label(event: Dict[str, Any]) -> Optional[str]:
    """Short label for the running status box."""
    kind = event.get("type")
    if kind == "route":
        return f"Routing to {ROUTE_LABELS.get(event.get('route'), event.get('route'))}…"
    if kind == "agent_start":
        return f"{AGENT_NAMES.get(event.get('agent'), 'Agent')} working…"
    if kind == "final":
        return "Checking the answer…"
    if kind == "guard_output":
        return "Finishing…"
    return None


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
