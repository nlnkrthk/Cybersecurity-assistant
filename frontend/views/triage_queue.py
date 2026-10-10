"""
Triage queue: rank many flows or incidents by priority (P1 Critical .. P4 Low).
"""
import streamlit as st

import formatting as fmt
from components import call, get_client, priority_counts_chart, ranking_table

ss = st.session_state
ss.setdefault("triage_result", None)

SAMPLE_CSV = "incident_id\n" + "\n".join(
    fmt.EXAMPLE_INCIDENTS[t] for t in ("DoS Hulk", "PortScan", "Heartbleed", "SSH-Patator", "BENIGN",
                                       "Web Attack - XSS", "Bot", "DDoS")
) + "\n"

st.title("🚨 Triage queue")
st.caption("Ranks flows by urgency from the predicted attack type and its severity, intrusion risk and "
           "anomaly, with rule overrides (e.g. reconnaissance capped at P3). No LLM involved.")

csv_tab, ids_tab = st.tabs(["Upload CSV", "Incident IDs"])
with csv_tab:
    st.markdown("One flow per row: an `incident_id` column and/or flow metric columns "
                "(`flow_duration`, `total_fwd_packets`, … or the CIC-IDS2017 names). Max 1000 rows, 5 MB.")
    upload = st.file_uploader("Flows CSV", type=["csv"])
    c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
    top_n = c1.number_input("Show top", 1, 1000, 50)
    c2.download_button("Download a sample CSV", SAMPLE_CSV, "sample_flows.csv", "text/csv", icon="📄")
    if st.button("Rank flows", type="primary", disabled=upload is None):
        with st.spinner("Scoring flows…"):
            ss.triage_result = call(lambda: get_client().prioritize_csv(upload.getvalue(), upload.name, int(top_n)),
                                    "Ranking")

with ids_tab:
    text = st.text_area("Incident IDs (comma, space or newline separated)",
                        value=", ".join(SAMPLE_CSV.split()[1:]), height=100)
    top_ids = st.number_input("Show top", 1, 1000, 10, key="top_ids")
    if st.button("Rank incidents", type="primary"):
        ids, invalid = fmt.parse_incident_ids(text)
        if invalid:
            st.warning(f"Ignored (not incident IDs): {', '.join(invalid[:10])}")
        if not ids:
            st.error("Enter at least one incident ID such as INC-000035.")
        else:
            with st.spinner("Scoring incidents…"):
                ss.triage_result = call(lambda: get_client().prioritize_batch(incident_ids=ids, top_n=int(top_ids)),
                                        "Ranking")

result = ss.triage_result
if result:
    st.divider()
    counts = result.get("priority_counts") or {}
    cols = st.columns(5)
    cols[0].metric("Flows", result.get("total_flows"), border=True)
    for col, level in zip(cols[1:], ("P1", "P2", "P3", "P4")):
        col.metric(level, counts.get(level, 0), border=True)
    if counts.get("needs_follow_up"):
        st.warning(f"{counts['needs_follow_up']} flow(s) lack key metrics and could not be ranked.")
    left, right = st.columns([1, 2])
    with left:
        st.plotly_chart(priority_counts_chart(counts), key="triage_counts")
    with right:
        frame = ranking_table(result.get("top") or [])
    if not frame.empty:
        st.download_button("Download ranking (CSV)", frame.to_csv(index=False), "triage_ranking.csv",
                           "text/csv", icon="⬇️")
