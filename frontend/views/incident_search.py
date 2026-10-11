"""
Incident Search: hybrid keyword + semantic retrieval over historical incidents with filters,
incident details, and attack-type traffic profiles.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import formatting as fmt
from components import attack_type_names, call, get_client, page_header

ss = st.session_state
ss.setdefault("search_result", None)

FEATURE_NAMES = {key: label for key, label, _ in fmt.FLOW_FIELDS}
FEATURE_NAMES.update({"flow_packets_per_sec": "Packet rate", "total_packets": "Total packets"})
OPERATORS = {">": "more than", ">=": "at least", "<": "less than", "<=": "at most", "=": "equal to", "==": "equal to"}

page_header("Incident Search",
            "Find similar incidents among about 97,000 historical cases, and see what each attack type "
            "typically looks like in network traffic.",
            how_to=["Describe the traffic in your own words, e.g. *SSH brute force with many short connections*.",
                    "Optionally narrow the search by attack type or destination port. You can also write numeric "
                    "conditions in the description, e.g. *more than 10000 packets per second*.",
                    "Click **Search**. Select a row in the results to see that incident's metrics.",
                    "Use **Analyse this incident** to open it in Flow Analysis.",
                    "The **Attack-type profiles** tab compares an attack's typical metrics with normal traffic."],
            icon=":material/manage_search:")

search_tab, profile_tab = st.tabs(["Find similar incidents", "Attack-type profiles"])

with search_tab:
    with st.form("search_form"):
        query = st.text_input("Describe the traffic or incident",
                              placeholder="e.g. SSH brute force with many short connections")
        c1, c2, c3 = st.columns([2, 1, 1])
        attack_type = c1.selectbox("Attack type (optional)", attack_type_names(), index=None, placeholder="Any type")
        port_text = c2.text_input("Destination port (optional)", placeholder="e.g. 22")
        top_k = c3.slider("Number of results", 1, 20, 5)
        submitted = st.form_submit_button("Search", type="primary", icon=":material/search:")
    st.caption("Search combines keyword matching with meaning-based (semantic) matching. Protocol, IP address "
               "and time filters are not available because this dataset release does not include those fields.")

    if submitted:
        port = None
        if port_text.strip():
            if not port_text.strip().isdigit() or not 0 <= int(port_text) <= 65535:
                st.error("The destination port must be a number from 0 to 65535.")
                st.stop()
            port = int(port_text)
        if not query.strip():
            st.error("Describe the traffic you are looking for.")
            st.stop()
        with st.spinner("Searching… (the first search can take a little longer)"):
            ss.search_result = call(lambda: get_client().search(query.strip(), attack_type, port, top_k), "Search")

    result = ss.search_result
    if result:
        incidents = result.get("incidents") or []
        filters = {**(result.get("filters_inferred_from_query") or {}), **(result.get("filters_applied") or {})}
        chips = []
        if filters.get("label"):
            chips.append(f":blue-badge[Attack type: {filters['label']}]")
        if filters.get("dst_port") is not None:
            chips.append(f":blue-badge[Port {filters['dst_port']}]")
        for condition in result.get("numeric_conditions") or []:
            name = FEATURE_NAMES.get(condition.get("feature"), condition.get("feature"))
            unit = f" {condition['unit']}" if condition.get("unit") else ""
            value = condition.get("value")
            value = f"{value:,.0f}" if isinstance(value, (int, float)) and float(value).is_integer() else value
            chips.append(f":violet-badge[{name} {OPERATORS.get(condition.get('operator'), condition.get('operator'))} "
                         f"{value}{unit}]")
        st.markdown(f"**{len(incidents)} matching incident{'s' if len(incidents) != 1 else ''}** "
                    f"(from {result.get('candidates_after_filters') or 0:,} candidates) " + " ".join(chips))
        if not incidents:
            st.info("No matching incidents. Try a broader description or fewer filters.", icon=":material/info:")
        else:
            frame = pd.DataFrame(incidents)[["incident_id", "attack_type", "dst_port", "relevance_score", "summary"]]
            selection = st.dataframe(frame, hide_index=True, on_select="rerun", selection_mode="single-row",
                                     key="search_table", column_config={
                                         "incident_id": "Incident",
                                         "attack_type": "Attack type",
                                         "dst_port": st.column_config.NumberColumn("Port", format="%d"),
                                         "relevance_score": st.column_config.NumberColumn("Match score", format="%.3f"),
                                         "summary": st.column_config.TextColumn("Summary", width="large"),
                                     })
            rows = selection.selection.rows if selection else []
            if rows:
                incident_id = frame.iloc[rows[0]]["incident_id"]
                details = call(lambda: get_client().incident(incident_id), "Incident lookup")
                if details:
                    with st.container(border=True):
                        head, action = st.columns([3, 1], vertical_alignment="center")
                        head.markdown(f"#### {incident_id}")
                        head.caption(f"Recorded as **{details['historical_label']}** · destination port "
                                     f"{details.get('dst_port')}")
                        if action.button("Analyse this incident", icon=":material/query_stats:", type="primary",
                                         width="stretch"):
                            ss.flow_prefill_incident = incident_id
                            st.switch_page("views/flow_analyzer.py")
                        feedback = details.get("analyst_feedback")
                        if feedback:
                            st.info(f"Analysts corrected this incident's attack type: {feedback.get('corrected_labels')}",
                                    icon=":material/person_check:")
                        metrics = pd.DataFrame(details["metrics"].items(), columns=["Metric", "Value"])
                        st.dataframe(metrics, hide_index=True, height=300)
            else:
                st.caption("Select a row to see the incident's details.")

with profile_tab:
    st.caption("How an attack type differs from normal traffic: the metrics that matter most to the classifier, "
               "with their typical (median) values.")
    name = st.selectbox("Attack type", [n for n in attack_type_names() if n != "BENIGN"], index=None,
                        placeholder="Choose an attack type", key="profile_type")
    if name:
        profile = call(lambda: get_client().attack_profile(name), "Profile")
        if profile:
            st.markdown(f"**{profile['attack_type']}** · severity `{profile['severity']}` · example incident "
                        f"`{fmt.EXAMPLE_INCIDENTS.get(profile['attack_type'], '–')}`")
            st.markdown(profile.get("summary", ""))
            features = profile.get("top_features") or []
            if features:
                labels = [f["feature"] for f in features]
                fig = go.Figure([
                    go.Bar(name=profile["attack_type"], x=labels, y=[f["attack_median"] for f in features],
                           marker_color="#DC2626"),
                    go.Bar(name="Normal traffic", x=labels, y=[f["normal_median"] for f in features],
                           marker_color="#2563EB"),
                ])
                fig.update_layout(barmode="group", yaxis_type="log", yaxis_title="Typical value (log scale)",
                                  height=380, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
                st.plotly_chart(fig, key="profile_chart")
                st.dataframe(pd.DataFrame(features), hide_index=True, column_config={
                    "feature": "Metric",
                    "direction_vs_normal": "Compared with normal",
                    "attack_median": st.column_config.NumberColumn("Typical in attack", format="%.2f"),
                    "normal_median": st.column_config.NumberColumn("Typical in normal traffic", format="%.2f"),
                    "unit": "Unit",
                    "importance_share": st.column_config.NumberColumn("Importance", format="percent"),
                })
