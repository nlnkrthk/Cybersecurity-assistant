"""
Triage Queue: rank many flows or incidents by priority (P1 Critical .. P4 Low).
"""
import streamlit as st

import formatting as fmt
from components import call, get_client, page_header, priority_counts_chart, priority_legend, ranking_table

ss = st.session_state
ss.setdefault("triage_result", None)

SAMPLE_CSV = "incident_id\n" + "\n".join(
    fmt.EXAMPLE_INCIDENTS[t] for t in ("DoS Hulk", "PortScan", "Heartbleed", "SSH-Patator", "BENIGN",
                                       "Web Attack - XSS", "Bot", "DDoS")
) + "\n"

page_header("Triage Queue",
            "Rank a batch of flows or incidents by urgency, so you know what to investigate first.",
            how_to=["Upload a CSV file with one flow per row (columns: `incident_id` and/or flow metrics such as "
                    "`flow_duration`, `total_fwd_packets`), or paste a list of incident IDs.",
                    "Click **Rank**. Each flow gets a priority from P1 (critical) to P4 (low), based on the "
                    "predicted attack type and its severity, the intrusion risk and how unusual the traffic is.",
                    "Flows with too few metrics are listed as *Needs data*.",
                    "Download the ranked list as CSV for your ticketing system."],
            icon=":material/format_list_numbered:")
priority_legend()

csv_tab, ids_tab = st.tabs(["Upload a CSV file", "Paste incident IDs"])
with csv_tab:
    upload = st.file_uploader("Flows CSV (max 1000 rows, 5 MB)", type=["csv"])
    c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="bottom")
    top_n = c1.number_input("Show the top", 1, 1000, 50)
    rank_csv = c2.button("Rank flows", type="primary", disabled=upload is None, icon=":material/sort:", width="stretch")
    c3.download_button("Download a sample CSV", SAMPLE_CSV, "sample_flows.csv", "text/csv", icon=":material/download:")
    if rank_csv:
        with st.spinner("Scoring flows…"):
            ss.triage_result = call(lambda: get_client().prioritize_csv(upload.getvalue(), upload.name, int(top_n)),
                                    "Ranking")

with ids_tab:
    text = st.text_area("Incident IDs (separated by commas, spaces or new lines)",
                        value=", ".join(SAMPLE_CSV.split()[1:]), height=100)
    c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
    top_ids = c1.number_input("Show the top", 1, 1000, 10, key="top_ids")
    if c2.button("Rank incidents", type="primary", icon=":material/sort:"):
        ids, invalid = fmt.parse_incident_ids(text)
        if invalid:
            st.warning(f"Ignored because they are not incident IDs: {', '.join(invalid[:10])}")
        if not ids:
            st.error("Enter at least one incident ID such as INC-000035.")
        else:
            with st.spinner("Scoring incidents…"):
                ss.triage_result = call(lambda: get_client().prioritize_batch(incident_ids=ids, top_n=int(top_ids)),
                                        "Ranking")

result = ss.triage_result
if result:
    st.divider()
    st.subheader("Results")
    counts = result.get("priority_counts") or {}
    cols = st.columns(5)
    cols[0].metric("Flows ranked", result.get("total_flows"), border=True)
    for col, level in zip(cols[1:], ("P1", "P2", "P3", "P4")):
        label, action = fmt.PRIORITY_INFO[level]
        col.metric(f"{level} · {label}", counts.get(level, 0), border=True, help=action)
    if counts.get("needs_follow_up"):
        st.warning(f"{counts['needs_follow_up']} flow(s) have too few metrics to be ranked (shown as Needs data).",
                   icon=":material/warning:")
    left, right = st.columns([1, 2.4])
    with left:
        st.plotly_chart(priority_counts_chart(counts), key="triage_counts")
    with right:
        frame = ranking_table(result.get("top") or [])
    if not frame.empty:
        st.download_button("Download the ranking (CSV)", frame.to_csv(index=False), "triage_ranking.csv",
                           "text/csv", icon=":material/download:")
