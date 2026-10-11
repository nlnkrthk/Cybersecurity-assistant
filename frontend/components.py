"""
Shared Streamlit pieces: API client, styles, page header, status sidebar, the agent progress
tracker, result cards and charts.
"""
import html
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import ApiClient, ApiError, base_url
import formatting as fmt

STYLES = """
<style>
/* ---- page header ---- */
.page-subtitle { color:#475569; font-size:1.02rem; margin:-0.6rem 0 0.4rem 0; }

/* ---- sidebar status card ---- */
.status-list { list-style:none; padding:0; margin:0.2rem 0 0.6rem 0; }
.status-list li { display:flex; justify-content:space-between; gap:0.5rem; padding:0.28rem 0;
                  border-bottom:1px solid rgba(255,255,255,0.08); font-size:0.88rem; }
.status-list li:last-child { border-bottom:none; }
.status-list .name { opacity:0.8; }
.status-list .value { font-weight:600; text-align:right; word-break:break-all; }
.dot { display:inline-block; width:0.55rem; height:0.55rem; border-radius:50%; margin-right:0.4rem; }
.dot.ok { background:#22C55E; } .dot.warn { background:#F59E0B; } .dot.bad { background:#EF4444; }
.side-title { font-size:0.75rem; letter-spacing:0.08em; text-transform:uppercase; opacity:0.7;
              margin:0.2rem 0 0.3rem 0; font-weight:700; }

/* ---- agent progress tracker ---- */
.pipe { border:1px solid #E2E8F0; border-radius:0.75rem; padding:0.6rem 0.9rem; background:#F8FAFC; }
.pipe-head { display:flex; justify-content:space-between; font-weight:650; color:#0F172A;
             padding-bottom:0.35rem; margin-bottom:0.35rem; border-bottom:1px solid #E2E8F0; }
.pipe-head .muted { font-weight:500; color:#64748B; }
.step { display:flex; gap:0.7rem; padding:0.35rem 0; position:relative; }
.step:not(:last-child)::after { content:""; position:absolute; left:0.72rem; top:1.9rem; bottom:-0.35rem;
                                width:2px; background:#E2E8F0; }
.icon { flex:0 0 1.5rem; height:1.5rem; border-radius:50%; display:flex; align-items:center;
        justify-content:center; font-size:0.8rem; font-weight:700; z-index:1; }
.step.done .icon { background:#DCFCE7; color:#15803D; }
.step.running .icon { background:#DBEAFE; color:#1D4ED8; }
.step.warning .icon { background:#FEF3C7; color:#B45309; }
.step.blocked .icon { background:#FEE2E2; color:#B91C1C; }
.step.pending .icon, .step.skipped .icon { background:#F1F5F9; color:#94A3B8; }
.step.pending .title, .step.skipped .title { color:#94A3B8; }
.spinner { width:0.8rem; height:0.8rem; border:2px solid #93C5FD; border-top-color:#1D4ED8;
           border-radius:50%; animation:spin 0.8s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
.body { flex:1; min-width:0; }
.title { font-weight:600; color:#0F172A; }
.title .time { float:right; font-weight:500; color:#64748B; font-size:0.82rem; }
.role { color:#64748B; font-size:0.82rem; }
.detail { color:#334155; font-size:0.88rem; margin-top:0.1rem; }
.step.blocked .detail { color:#B91C1C; font-weight:600; }
.step.warning .detail { color:#92400E; }
.note { color:#64748B; font-size:0.82rem; font-style:italic; }
.tools { list-style:none; padding:0; margin:0.25rem 0 0 0; }
.tools li { font-size:0.84rem; color:#334155; padding:0.12rem 0 0.12rem 0.2rem; }
.tools li .tool-name { font-weight:600; }
.tools li .muted { color:#64748B; }

/* ---- verdict banner (flow analysis) ---- */
.verdict { border-radius:0.75rem; padding:0.9rem 1.1rem; margin:0.4rem 0 0.8rem 0; color:#0F172A;
           border-left:6px solid var(--c); background:var(--bg); }
.verdict .big { font-size:1.25rem; font-weight:700; }
.verdict .small { color:#334155; margin-top:0.15rem; }

/* ---- legend chips ---- */
.chips { display:flex; flex-wrap:wrap; gap:0.4rem; margin:0.2rem 0 0.6rem 0; }
.chip { border-radius:999px; padding:0.15rem 0.65rem; font-size:0.82rem; font-weight:600; color:white; }
</style>
"""

STATUS_ICONS = {"done": "✓", "warning": "!", "blocked": "✕", "skipped": "–", "pending": "○"}
VERDICT_BG = {"P1": "#FEF2F2", "P2": "#FFF7ED", "P3": "#FEFCE8", "P4": "#F0FDF4"}


def inject_styles() -> None:
    st.html(STYLES)


# ============================================================================
# API access
# ============================================================================
@st.cache_resource
def _cached_client(url: str, client_class_id: int) -> ApiClient:
    return ApiClient(url)


def get_client() -> ApiClient:
    """The HTTP client with this browser session's login token. The connection pool is shared
    (one per API URL); the token never is. Keyed on the class too, so a hot reload of api_client
    (code change while running) never leaves an old client raising the old ApiError class."""
    return _cached_client(base_url(), id(ApiClient)).with_token(st.session_state.get("auth_token"))


get_client.clear = _cached_client.clear


# ============================================================================
# Signed-in user (kept in the Streamlit session: closing the tab or a hard refresh signs out)
# ============================================================================
def current_user() -> Optional[Dict[str, Any]]:
    return st.session_state.get("auth_user")


def start_session(session: Dict[str, Any]) -> None:
    """Sign in with an /auth/login or /auth/signup response, starting from a clean session."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.auth_token = session["token"]
    st.session_state.auth_user = session["user"]


def end_session(revoke: bool = True, notice: Optional[str] = None) -> None:
    """Sign out: revoke the token on the server and forget everything of this user in the browser."""
    if revoke and st.session_state.get("auth_token"):
        try:
            get_client().logout()
        except ApiError:
            pass  # the token expires on its own
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    if notice:
        st.session_state.auth_notice = notice


def session_expired(exc: ApiError) -> bool:
    """True (and signed out) when the API rejected the token; the next rerun shows the sign-in page."""
    if exc.status == 401 and st.session_state.get("auth_token"):
        end_session(revoke=False, notice="Your session has expired. Please sign in again.")
        return True
    return False


def call(fn: Callable[[], Any], what: str = "The request") -> Optional[Any]:
    """Run an API call; show a friendly error and return None on failure."""
    try:
        return fn()
    except ApiError as exc:
        if session_expired(exc):
            st.rerun()
        st.error(f"{what} failed. {exc.message}", icon=":material/error:")
        return None


@st.cache_data(ttl=15, show_spinner=False)
def _health() -> Dict[str, Any]:
    return get_client().health()


def health() -> Dict[str, Any]:
    """API status (cached 15 s); an unreachable API is not cached, so it is retried on every rerun."""
    try:
        return _health()
    except ApiError as exc:
        return {"status": "down", "error": exc.message}


health.clear = _health.clear


@st.cache_data(ttl=600, show_spinner=False)
def _attack_type_names() -> List[str]:
    return [t["attack_type"] for t in get_client().attack_types()]


def attack_type_names() -> List[str]:
    """The 15 attack types from the API (cached); the built-in list while the API is unreachable."""
    try:
        return _attack_type_names()
    except ApiError:
        return list(fmt.EXAMPLE_INCIDENTS)


attack_type_names.clear = _attack_type_names.clear


# ============================================================================
# Layout pieces
# ============================================================================
def page_header(title: str, subtitle: str, how_to: Sequence[str] = (), icon: str = "") -> None:
    """Title, one-line explanation and an optional 'How to use this page' list."""
    st.title(f"{icon} {title}".strip())
    st.markdown(f'<div class="page-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)
    if how_to:
        with st.expander("How to use this page", icon=":material/help:"):
            st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(how_to, start=1)))


def _status_row(name: str, value: str, state: str) -> str:
    return (f'<li><span class="name"><span class="dot {state}"></span>{html.escape(name)}</span>'
            f'<span class="value">{html.escape(value)}</span></li>')


def account_sidebar() -> None:
    """Signed-in user and the Log out button."""
    user = current_user()
    if not user:
        return
    with st.sidebar:
        st.markdown('<div class="side-title">Account</div>', unsafe_allow_html=True)
        name = user["username"].replace("_", r"\_")  # usernames are letters, digits, '.', '_', '-'
        st.markdown(f":material/account_circle: Signed in as **{name}**")
        st.button("Log out", icon=":material/logout:", on_click=end_session, width="stretch", key="logout")
        st.divider()


def status_sidebar() -> Dict[str, Any]:
    """System status card shown on every page."""
    info = health()
    with st.sidebar:
        st.markdown('<div class="side-title">System status</div>', unsafe_allow_html=True)
        if info.get("status") == "down":
            st.markdown('<ul class="status-list">' + _status_row("Backend API", "Offline", "bad") + "</ul>",
                        unsafe_allow_html=True)
            st.caption(f"Cannot reach `{get_client().url}`. Start it with `scripts/api/run_api.py`, then retry.")
            if st.button("Retry connection", icon=":material/refresh:", width="stretch", key="retry_api"):
                health.clear()
                attack_type_names.clear()
                st.rerun()
            return info
        index = info.get("search_index")
        rows = [
            _status_row("Backend API", "Online" if info.get("status") == "ok" else "Degraded",
                        "ok" if info.get("status") == "ok" else "warn"),
            _status_row("AI assistant", (info.get("llm_model") or "").split(":")[-1].split("/")[-1] or "ready"
                        if info.get("assistant_ready") else "Unavailable",
                        "ok" if info.get("assistant_ready") else "bad"),
            _status_row("Incident search", {"ready": "Ready", "loading": "Loading…", "not_loaded": "Not loaded",
                                            "error": "Error"}.get(index, str(index)),
                        {"ready": "ok", "loading": "warn"}.get(index, "bad")),
            _status_row("Attack classifier", "Original" if (info.get("classifier_version") or "original") == "original"
                        else f"Retrained ({info['classifier_version']})", "ok"),
        ]
        st.markdown('<ul class="status-list">' + "".join(rows) + "</ul>", unsafe_allow_html=True)
        if not info.get("assistant_ready") and info.get("assistant_error"):
            st.caption(info["assistant_error"])
        if st.button("Refresh status", icon=":material/refresh:", width="stretch", key="refresh_status"):
            health.clear()
            st.rerun()
    return info


def legend_chips(items: Sequence[tuple]) -> None:
    """Coloured chips, e.g. the priority legend: [(text, colour), ...]."""
    chips = "".join(f'<span class="chip" style="background:{c}">{html.escape(t)}</span>' for t, c in items)
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)


def priority_legend() -> None:
    legend_chips([(f"{p} {label} · {action}", fmt.PRIORITY_COLORS[p])
                  for p, (label, action) in fmt.PRIORITY_INFO.items()])


# ============================================================================
# Agent progress tracker
# ============================================================================
def _secs(ms: Optional[float]) -> str:
    return "" if ms is None else f"{ms / 1000:.1f}s"


def pipeline_html(steps: List[Dict[str, Any]], heading: str = "", right: str = "") -> str:
    """The progress tracker as HTML (escaped)."""
    rows = []
    for step in steps:
        status = step["status"]
        icon = '<div class="spinner"></div>' if status == "running" else STATUS_ICONS.get(status, "○")
        time = f'<span class="time">{_secs(step.get("elapsed_ms"))}</span>' if step.get("elapsed_ms") is not None else ""
        lines = [f'<div class="title">{html.escape(step["title"])}{time}</div>']
        if step.get("role") and status in ("pending", "running"):
            lines.append(f'<div class="role">{html.escape(step["role"])}</div>')
        if step.get("detail"):
            lines.append(f'<div class="detail">{html.escape(step["detail"])}</div>')
        if step.get("note"):
            lines.append(f'<div class="note">{html.escape(step["note"])}</div>')
        if step.get("tools"):
            items = []
            for tool in step["tools"]:
                mark = {"running": "…", "done": "✓", "warning": "!"}.get(tool.get("status"), "•")
                detail = f' <span class="muted">→ {html.escape(str(tool.get("detail")))}</span>' if tool.get("detail") else ""
                items.append(f'<li>{mark} <span class="tool-name">{html.escape(tool["label"])}</span>{detail}</li>')
            lines.append(f'<ul class="tools">{"".join(items)}</ul>')
        rows.append(f'<div class="step {status}"><div class="icon">{icon}</div><div class="body">{"".join(lines)}</div></div>')
    head = (f'<div class="pipe-head"><span>{html.escape(heading)}</span><span class="muted">{html.escape(right)}</span></div>'
            if heading else "")
    return f'<div class="pipe">{head}{"".join(rows)}</div>'


def show_pipeline(target, steps: List[Dict[str, Any]], heading: str = "", right: str = "") -> None:
    target.markdown(pipeline_html(steps, heading, right), unsafe_allow_html=True)


# ============================================================================
# Assistant results
# ============================================================================
def judgement_card(judgement: Dict[str, Any]) -> None:
    if not judgement:
        return
    verdict = judgement.get("verdict")
    if verdict == "error":
        st.caption(f"Quality grading could not score this answer: {judgement.get('error')}")
        return
    label = {"pass": "Passed", "needs_review": "Needs review", "fail": "Failed"}.get(verdict, verdict)
    with st.expander(f"Quality check: {judgement.get('overall', 0):.1f} / 5 · {label}",
                     icon=":material/verified:" if verdict == "pass" else ":material/report:",
                     expanded=verdict != "pass"):
        st.caption("An independent LLM grader checks the answer against the data the agents used.")
        scores = judgement.get("scores") or {}
        applicable = judgement.get("applicable") or list(scores)
        cols = st.columns(max(len(scores), 1))
        explain = {"faithfulness": "Only states facts found in the data",
                   "correctness": "Predictions and priorities reported correctly",
                   "relevance": "Answers the question asked",
                   "actionability": "Gives concrete next steps",
                   "safety": "No harmful advice; uncertainty stated"}
        for col, (name, value) in zip(cols, scores.items()):
            col.metric(name.capitalize(), f"{value}/5" if name in applicable else "n/a", help=explain.get(name))
        for name, reason in (judgement.get("reasons") or {}).items():
            st.markdown(f"**{name.capitalize()}:** {reason}")
        claims = judgement.get("unsupported_claims") or []
        if claims:
            st.markdown("**Statements not supported by the data:**")
            for claim in claims:
                st.markdown(f"- {claim}")


def evidence_expander(result: Dict[str, Any]) -> None:
    evidence = result.get("evidence") or []
    if not evidence:
        return
    with st.expander(f"Data used ({len(evidence)} tool result{'s' if len(evidence) != 1 else ''})",
                     icon=":material/database:"):
        st.caption("The raw results the agents worked from. The answer may only use facts from here.")
        for item in evidence:
            st.markdown(f"**{fmt.TOOL_LABELS.get(item.get('tool'), item.get('tool'))}** "
                        f"· {fmt.AGENT_NAMES.get(item.get('agent'), item.get('agent'))}")
            st.json(item.get("result"), expanded=False)


def trace_expander(result: Dict[str, Any]) -> None:
    trace = result.get("trace") or []
    if not trace:
        return
    judged = any(e.get("type") == "judge" for e in trace)
    steps = fmt.pipeline_steps(trace, judge=judged)
    with st.expander(f"How this answer was produced · {fmt.pipeline_summary(steps)}", icon=":material/account_tree:"):
        st.markdown(pipeline_html(steps), unsafe_allow_html=True)


def answer_extras(result: Dict[str, Any]) -> None:
    """Summary line, guardrail notices, quality check, process and data under an answer."""
    parts = [fmt.route_badge(result.get("route"))]
    agents = [fmt.AGENT_NAMES.get(a, a).replace(" agent", "") for a in result.get("agents_run") or []]
    if agents:
        parts.append(":gray-badge[Agents: " + ", ".join(agents) + "]")
    if result.get("needs_follow_up"):
        parts.append(":violet-badge[Waiting for your reply]")
    if result.get("elapsed_ms") is not None:
        parts.append(f":gray-badge[:material/timer: {result['elapsed_ms'] / 1000:.0f}s]")
    st.markdown(" ".join(parts))
    inputs, outputs = fmt.guardrail_notices(result)
    for note in inputs:
        st.info(note, icon=":material/shield:")
    if outputs:
        st.warning("**The answer check adjusted this reply:**  \n" + "  \n".join(f"• {n}" for n in outputs),
                   icon=":material/shield:")
    judgement_card(result.get("judgement"))
    trace_expander(result)
    evidence_expander(result)


# ============================================================================
# Flow analysis
# ============================================================================
def shap_chart(reasons: List[Dict[str, Any]], title: str) -> Optional[go.Figure]:
    rows = fmt.shap_rows(reasons)
    if not rows:
        return None
    colors = ["#DC2626" if r["shap"] > 0 else "#2563EB" for r in rows]
    fig = go.Figure(go.Bar(
        x=[r["shap"] for r in rows], y=[r["feature"] for r in rows], orientation="h",
        marker_color=colors, customdata=[r["text"] for r in rows],
        hovertemplate="%{customdata}<br>contribution %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=110 + 42 * len(rows),
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Contribution to the result",
                      yaxis=dict(automargin=True), plot_bgcolor="white")
    return fig


def verdict_banner(result: Dict[str, Any]) -> None:
    attack, risk = result["attack_type"], result["risk"]
    priority = (result.get("priority") or {}).get("priority")
    label, action = fmt.PRIORITY_INFO.get(priority, ("", ""))
    color = fmt.PRIORITY_COLORS.get(priority, "#64748B")
    benign = attack["prediction"] == "BENIGN"
    headline = ("Looks like normal traffic" if benign
                else f"{html.escape(attack['prediction'])} detected")
    st.markdown(
        f'<div class="verdict" style="--c:{color};--bg:{VERDICT_BG.get(priority, "#F8FAFC")}">'
        f'<div class="big">{headline} · {priority or "?"} {label}</div>'
        f'<div class="small">Recommended action: <b>{html.escape(action)}</b>. '
        f'Model confidence {fmt.pct(attack["confidence"])}, intrusion risk {fmt.pct(risk["probability"])} '
        f'({html.escape(risk["level"])}).</div></div>',
        unsafe_allow_html=True)


def flow_result(result: Dict[str, Any]) -> None:
    """Result of POST /ml/explain."""
    if result.get("status") == "needs_follow_up":
        st.warning(f"**More information needed.** {result.get('follow_up_question')}", icon=":material/help:")
        missing = ", ".join(result.get("missing_key_features") or [])
        if missing:
            st.caption(f"Missing: {missing} · metrics provided: {result.get('input', {}).get('known_metrics')}")
        return

    attack, risk, anomaly = result["attack_type"], result["risk"], result["anomaly"]
    priority = result.get("priority") or {}
    incident = result.get("incident")
    if incident:
        st.markdown(f"**{incident['incident_id']}** · recorded in the dataset as `{incident['historical_label']}` "
                    f"· destination port {incident.get('dst_port')}")
    verdict_banner(result)

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])  # attack names can be long ("Web Attack - Sql Injection")
    c1.metric("Attack type", attack["prediction"], f"{fmt.pct(attack['confidence'])} confidence",
              delta_color="off", delta_arrow="off", border=True,
              help="Which of 15 traffic types (14 attacks or normal) the classifier thinks this flow is.")
    c2.metric("Intrusion risk", fmt.pct(risk["probability"]), risk["level"], delta_color="off", delta_arrow="off",
              border=True, help="Probability that the flow is malicious at all, whatever the attack type.")
    c3.metric("Unusualness", anomaly["level"], f"above {anomaly.get('benign_percentile', 0):.0f}% of normal",
              delta_color="off", delta_arrow="off", border=True,
              help="How different the flow is from normal traffic (anomaly detector). Can flag new attack types.")
    c4.metric("Priority", priority.get("priority", "–"), priority.get("label"), delta_color="off", delta_arrow="off",
              border=True, help="P1 Critical (investigate now) to P4 Low (monitor), from type, risk and unusualness.")

    if priority.get("reason"):
        st.caption(f"Priority reasoning: {priority['reason']}")
    for override in priority.get("overrides_applied") or []:
        st.caption(f"Rule applied: {override}")
    if incident and incident["historical_label"] != attack["prediction"]:
        st.info(f"The model's prediction ({attack['prediction']}) differs from the label recorded in the dataset "
                f"({incident['historical_label']}).", icon=":material/info:")
    quality = result.get("input", {})
    if quality.get("input_quality") not in (None, "full"):
        st.warning(f"Only {quality.get('known_metrics')} metrics were provided. The missing ones were filled with "
                   "typical normal-traffic values, so treat this result with care (it tends to look more normal).",
                   icon=":material/warning:")
    if attack.get("alternatives"):
        st.caption("Other possibilities: " + ", ".join(f"{a.get('attack_type')} ({fmt.pct(a.get('probability'))})"
                                                       for a in attack["alternatives"]))

    explanation = result.get("explanation") or {}
    st.subheader("Why the models decided this")
    st.caption("Each bar is a traffic metric. Red bars pushed the result toward the prediction or higher risk; "
               "blue bars pushed against it. Longer bars mattered more (SHAP values).")
    left, right = st.columns(2)
    for column, key, reasons, title in ((left, "shap_attack", explanation.get("attack_type_reasons"),
                                         f"Toward “{attack['prediction']}”"),
                                        (right, "shap_risk", explanation.get("risk_reasons"), "Toward higher risk")):
        fig = shap_chart(reasons, title)
        if fig:
            column.plotly_chart(fig, key=key)
    unusual = anomaly.get("most_unusual_features") or []
    if unusual:
        st.markdown("**Most unusual metrics compared with normal traffic**")
        st.dataframe(pd.DataFrame([{"Metric": u["text"], "Higher than % of normal traffic": u.get("benign_percentile")}
                                   for u in unusual]), hide_index=True)
    with st.expander("Technical details (raw API response)", icon=":material/code:"):
        st.json(result, expanded=False)


def priority_counts_chart(counts: Dict[str, int]) -> go.Figure:
    order = [k for k in ("P1", "P2", "P3", "P4", "needs_follow_up") if k in counts]
    order += [k for k in counts if k not in order]
    labels = [k if k != "needs_follow_up" else "Needs data" for k in order]
    fig = go.Figure(go.Bar(
        x=labels, y=[counts[k] for k in order], text=[counts[k] for k in order], textposition="auto",
        marker_color=[fmt.PRIORITY_COLORS.get(k, "#94A3B8") for k in order],
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Flows", plot_bgcolor="white")
    return fig


def ranking_table(top: List[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(fmt.ranking_rows(top))
    if frame.empty:
        st.info("No flows to show.")
        return frame
    frame = frame.dropna(axis=1, how="all")

    def color(value):
        c = fmt.PRIORITY_COLORS.get(value)
        return f"background-color: {c}; color: white; font-weight: 600" if c else ""

    styled = frame.style.map(color, subset=["priority"])
    st.dataframe(styled, hide_index=True, column_config={
        "rank": st.column_config.NumberColumn("#", width="small"),
        "flow": "Flow / incident",
        "priority": "Priority",
        "score": st.column_config.ProgressColumn("Urgency score", min_value=0, max_value=100, format="%.0f"),
        "attack_type": "Attack type",
        "confidence": st.column_config.NumberColumn("Confidence", format="percent"),
        "risk": st.column_config.NumberColumn("Intrusion risk", format="percent"),
        "anomaly": "Unusualness",
        "historical_label": "Recorded label",
        "input_quality": "Input quality",
        "reason": st.column_config.TextColumn("Why", width="large"),
    })
    return frame
