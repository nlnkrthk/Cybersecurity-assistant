"""
Incident search: hybrid keyword + semantic retrieval over historical incidents with
metadata filters, incident details, and attack-type traffic profiles.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import formatting as fmt
from components import attack_type_names, call, get_client

ss = st.session_state
ss.setdefault("search_result", None)

st.title("🔎 Incident search")
search_tab, profile_tab = st.tabs(["Similar incidents", "Attack-type profiles"])

with search_tab:
    st.caption("Hybrid search (BM25 keywords + sentence embeddings) over ~97,000 CIC-IDS2017 incidents. "
               "Numeric conditions in the query are applied as filters, e.g. *more than 10000 packets per second*.")
    with st.form("search_form"):
        query = st.text_input("Describe the traffic or incident",
                              placeholder="SSH brute force with many short connections")
        c1, c2, c3 = st.columns([2, 1, 1])
        attack_type = c1.selectbox("Attack type", attack_type_names(), index=None, placeholder="Any")
        port_text = c2.text_input("Destination port", placeholder="Any, e.g. 22")
        top_k = c3.slider("Results", 1, 20, 5)
        submitted = st.form_submit_button("Search", type="primary")
    st.caption("The dataset version used here has no protocol, IP address or timestamp values, so those "
               "metadata filters are not available; attack type, destination port and numeric conditions are.")

    if submitted:
        port = None
        if port_text.strip():
            if not port_text.strip().isdigit() or not 0 <= int(port_text) <= 65535:
                st.error("The destination port must be a number from 0 to 65535.")
                st.stop()
            port = int(port_text)
        if not query.strip():
            st.error("Enter a description to search for.")
            st.stop()
        with st.spinner("Searching (the first search can take a while if the index is still loading)…"):
            ss.search_result = call(lambda: get_client().search(query.strip(), attack_type, port, top_k), "Search")

    result = ss.search_result
    if result:
        incidents = result.get("incidents") or []
        notes = [f"{len(incidents)} result(s) from {result.get('candidates_after_filters')} candidates"]
        if result.get("filters_applied"):
            notes.append(f"filters `{result['filters_applied']}`")
        if result.get("filters_inferred_from_query"):
            notes.append(f"inferred from the query `{result['filters_inferred_from_query']}`")
        for condition in result.get("numeric_conditions") or []:
            notes.append(f"condition `{condition.get('feature')} {condition.get('operator')} "
                         f"{condition.get('value')}{(' ' + condition['unit']) if condition.get('unit') else ''}`")
        st.markdown(" · ".join(notes))
        if not incidents:
            st.info("No matching incidents. Try fewer filters or a broader description.")
        else:
            frame = pd.DataFrame(incidents)[["incident_id", "attack_type", "dst_port", "relevance_score", "summary"]]
            selection = st.dataframe(frame, hide_index=True, on_select="rerun", selection_mode="single-row",
                                     key="search_table", column_config={
                                         "relevance_score": st.column_config.NumberColumn("relevance", format="%.3f"),
                                         "summary": st.column_config.TextColumn(width="large"),
                                     })
            rows = selection.selection.rows if selection else []
            if rows:
                incident_id = frame.iloc[rows[0]]["incident_id"]
                details = call(lambda: get_client().incident(incident_id), "Incident lookup")
                if details:
                    st.subheader(incident_id)
                    st.markdown(f"Historical label `{details['historical_label']}` · port {details.get('dst_port')}")
                    feedback = details.get("analyst_feedback")
                    if feedback:
                        st.info(f"Analyst corrections: {feedback.get('corrected_labels')}", icon="🧑‍💻")
                    metrics = pd.DataFrame(details["metrics"].items(), columns=["metric", "value"])
                    st.dataframe(metrics, hide_index=True, height=300)
                    if st.button("Analyse in the flow analyzer", icon="🔬"):
                        ss.flow_prefill_incident = incident_id
                        st.switch_page("views/flow_analyzer.py")
            else:
                st.caption("Select a row to see the incident's metrics.")

with profile_tab:
    st.caption("Typical traffic of each attack type: the metrics that differ most from normal traffic "
               "(medians, ranked by SHAP importance).")
    name = st.selectbox("Attack type", [n for n in attack_type_names() if n != "BENIGN"], index=None,
                        placeholder="Pick an attack type", key="profile_type")
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
                           marker_color="#d62728"),
                    go.Bar(name="Normal traffic", x=labels, y=[f["normal_median"] for f in features],
                           marker_color="#1f77b4"),
                ])
                fig.update_layout(barmode="group", yaxis_type="log", yaxis_title="median (log scale)",
                                  height=380, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, key="profile_chart")
                st.dataframe(pd.DataFrame(features), hide_index=True, column_config={
                    "importance_share": st.column_config.NumberColumn("importance", format="percent"),
                })
