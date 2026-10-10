"""
Shared Streamlit pieces: API client, status sidebar, result cards and charts.
"""
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import ApiClient, ApiError, base_url
import formatting as fmt


@st.cache_resource
def _cached_client(url: str, client_class_id: int) -> ApiClient:
    return ApiClient(url)


def get_client() -> ApiClient:
    """One shared HTTP client per API URL. Keyed on the class too, so a hot reload of api_client
    (code change while running) never leaves an old client raising the old ApiError class."""
    return _cached_client(base_url(), id(ApiClient))


get_client.clear = _cached_client.clear


def call(fn: Callable[[], Any], what: str = "The request") -> Optional[Any]:
    """Run an API call; show a friendly error and return None on failure."""
    try:
        return fn()
    except ApiError as exc:
        st.error(f"{what} failed. {exc.message}", icon="⚠️")
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


def status_sidebar() -> Dict[str, Any]:
    """Backend status block shown on every page."""
    info = health()
    with st.sidebar:
        st.caption("BACKEND")
        if info.get("status") == "down":
            st.error(info["error"], icon="🔌")
            st.caption(f"API: `{get_client().url}` — start it with `scripts/api/run_api.py`.")
            if st.button("Retry", icon="🔄", type="tertiary"):
                health.clear()
                attack_type_names.clear()
                st.rerun()
            return info
        ok = "🟢" if info.get("status") == "ok" else "🟠"
        st.markdown(f"{ok} API {info.get('status')} · `{get_client().url}`")
        if info.get("assistant_ready"):
            st.markdown(f"🤖 LLM `{info.get('llm_model')}`")
        else:
            st.markdown(f"🤖 Assistant unavailable: {info.get('assistant_error') or 'not configured'}")
        index = {"ready": "ready", "loading": "loading…", "not_loaded": "not loaded",
                 "error": "error"}.get(info.get("search_index"), info.get("search_index"))
        st.markdown(f"🔎 Search index {index}")
        st.markdown(f"🌲 Classifier `{info.get('classifier_version') or 'original'}`")
        if st.button("Refresh status", icon="🔄", type="tertiary"):
            health.clear()
            st.rerun()
    return info


# ============================================================================
# Assistant results
# ============================================================================
def judgement_card(judgement: Dict[str, Any]) -> None:
    if not judgement:
        return
    verdict = judgement.get("verdict")
    if verdict == "error":
        st.caption(f"⚖️ Judge: not scored — {judgement.get('error')}")
        return
    with st.expander(f"⚖️ Judge: {judgement.get('overall', 0):.1f}/5 · {verdict.replace('_', ' ')}",
                     expanded=verdict != "pass"):
        st.markdown(fmt.verdict_badge(verdict))
        scores = judgement.get("scores") or {}
        applicable = judgement.get("applicable") or list(scores)
        cols = st.columns(max(len(scores), 1))
        for col, (name, value) in zip(cols, scores.items()):
            col.metric(name.capitalize(), f"{value}/5" if name in applicable else "n/a")
        for name, reason in (judgement.get("reasons") or {}).items():
            st.markdown(f"**{name.capitalize()}:** {reason}")
        claims = judgement.get("unsupported_claims") or []
        if claims:
            st.markdown("**Unsupported claims:**")
            for claim in claims:
                st.markdown(f"- {claim}")


def evidence_expander(result: Dict[str, Any]) -> None:
    evidence = result.get("evidence") or []
    if not evidence:
        return
    with st.expander(f"🔧 Evidence: {len(evidence)} tool call(s)"):
        for item in evidence:
            st.markdown(f"**{fmt.AGENT_NAMES.get(item.get('agent'), item.get('agent'))}** · "
                        f"`{item.get('tool')}`")
            st.json(item.get("result"), expanded=False)


def trace_expander(result: Dict[str, Any]) -> None:
    lines = [line for line in (fmt.trace_line(e) for e in result.get("trace") or []) if line]
    if lines:
        with st.expander("🧭 Agent trace"):
            st.markdown("  \n".join(lines))


def answer_extras(result: Dict[str, Any], show_trace: bool = True) -> None:
    """Badges, guardrail notices, judge and evidence under an assistant answer."""
    parts = [fmt.route_badge(result.get("route"))]
    if result.get("needs_follow_up"):
        parts.append(":violet-badge[needs more data]")
    if result.get("elapsed_ms") is not None:
        parts.append(f":gray-badge[{result['elapsed_ms'] / 1000:.1f}s]")
    st.markdown(" ".join(parts))
    inputs, outputs = fmt.guardrail_notices(result)
    for note in inputs:
        st.info(note, icon="🛡️")
    if outputs:
        st.warning("**Output guardrail adjusted this answer:**  \n" + "  \n".join(f"• {n}" for n in outputs),
                   icon="🛡️")
    judgement_card(result.get("judgement"))
    evidence_expander(result)
    if show_trace:
        trace_expander(result)


# ============================================================================
# Flow analysis
# ============================================================================
def shap_chart(reasons: List[Dict[str, Any]], title: str) -> Optional[go.Figure]:
    rows = fmt.shap_rows(reasons)
    if not rows:
        return None
    colors = ["#d62728" if r["shap"] > 0 else "#1f77b4" for r in rows]
    fig = go.Figure(go.Bar(
        x=[r["shap"] for r in rows], y=[r["feature"] for r in rows], orientation="h",
        marker_color=colors, text=[r["effect"] for r in rows], textposition="auto",
        customdata=[r["text"] for r in rows], hovertemplate="%{customdata}<br>SHAP %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(title=title, height=120 + 45 * len(rows), margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="SHAP contribution (positive = pushes toward the prediction)",
                      yaxis=dict(automargin=True))
    return fig


def flow_result(result: Dict[str, Any]) -> None:
    """Card for a POST /ml/explain result."""
    if result.get("status") == "needs_follow_up":
        st.warning(f"**More data needed.** {result.get('follow_up_question')}", icon="❓")
        missing = ", ".join(result.get("missing_key_features") or [])
        if missing:
            st.caption(f"Missing: {missing} · known metrics {result.get('input', {}).get('known_metrics')}")
        return

    attack, risk, anomaly = result["attack_type"], result["risk"], result["anomaly"]
    priority = result.get("priority") or {}
    incident = result.get("incident")
    if incident:
        st.markdown(f"**{incident['incident_id']}** · historical label `{incident['historical_label']}` · "
                    f"port {incident.get('dst_port')}")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])  # attack names can be long ("Web Attack - Sql Injection")
    c1.metric("Attack type", attack["prediction"], f"{fmt.pct(attack['confidence'])} confidence",
              delta_color="off", delta_arrow="off", border=True)
    c2.metric("Intrusion risk", fmt.pct(risk["probability"]), risk["level"], delta_color="off", delta_arrow="off", border=True)
    c3.metric("Anomaly", anomaly["level"], f"score {anomaly.get('score')}", delta_color="off", delta_arrow="off", border=True,
              help="How unusual the flow is compared with normal traffic (isolation forest).")
    c4.metric("Priority", priority.get("priority", "–"), priority.get("label"), delta_color="off", delta_arrow="off", border=True)

    if priority:
        st.markdown(f"{fmt.priority_badge(priority.get('priority'))} **{priority.get('action')}** — "
                    f"{priority.get('reason')}")
        for override in priority.get("overrides_applied") or []:
            st.caption(f"Rule override: {override}")
        if priority.get("confidence_note"):
            st.caption(priority["confidence_note"])

    if incident and incident["historical_label"] != attack["prediction"]:
        st.info(f"The model's prediction ({attack['prediction']}) differs from the historical label "
                f"({incident['historical_label']}).", icon="ℹ️")
    quality = result.get("input", {})
    if quality.get("input_quality") not in (None, "full"):
        st.warning(f"Input quality **{quality['input_quality']}** ({quality.get('known_metrics')} metrics known). "
                   "Missing metrics were filled with typical normal values, so treat the result with care.",
                   icon="⚠️")
    for warning in quality.get("warnings") or []:
        st.caption(f"⚠️ {warning}")
    if attack.get("alternatives"):
        st.caption("Alternatives: " + ", ".join(f"{a.get('attack_type')} ({fmt.pct(a.get('probability'))})"
                                                 for a in attack["alternatives"]))

    explanation = result.get("explanation") or {}
    st.subheader("Why? (SHAP explanation)")
    st.caption(explanation.get("summary", ""))
    for key, reasons, title in (("shap_attack", explanation.get("attack_type_reasons"),
                                 f"Why the classifier says {attack['prediction']}"),
                                ("shap_risk", explanation.get("risk_reasons"), "What drives the intrusion risk")):
        fig = shap_chart(reasons, title)
        if fig:
            st.plotly_chart(fig, key=key)
    unusual = anomaly.get("most_unusual_features") or []
    if unusual:
        st.markdown("**Most unusual metrics vs normal traffic**")
        st.dataframe(pd.DataFrame([{"metric": u["text"], "normal percentile": u.get("benign_percentile")}
                                   for u in unusual]), hide_index=True)
    with st.expander("Raw API response"):
        st.json(result, expanded=False)


def priority_counts_chart(counts: Dict[str, int]) -> go.Figure:
    order = [k for k in ("P1", "P2", "P3", "P4", "needs_follow_up") if k in counts]
    order += [k for k in counts if k not in order]
    fig = go.Figure(go.Bar(
        x=order, y=[counts[k] for k in order], text=[counts[k] for k in order], textposition="auto",
        marker_color=[fmt.PRIORITY_COLORS.get(k, "#7f7f7f") for k in order],
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="flows")
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
        "score": st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%.1f"),
        "confidence": st.column_config.NumberColumn(format="percent"),
        "risk": st.column_config.NumberColumn(format="percent"),
        "reason": st.column_config.TextColumn(width="large"),
    })
    return frame
