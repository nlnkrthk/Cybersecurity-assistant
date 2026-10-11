"""
Flow Analysis: attack type, intrusion risk, unusualness and priority of one flow, with the
metrics behind each prediction (POST /ml/explain; machine learning only, no LLM).
"""
import streamlit as st

import formatting as fmt
from api_client import ApiError
from components import call, flow_result, get_client, page_header

TAB_INCIDENT, TAB_METRICS = "Use a historical incident", "Enter flow metrics"
ESSENTIAL = fmt.FLOW_FIELDS[:5]   # duration, packets and bytes in both directions
OPTIONAL_GROUPS = {
    "Rates": fmt.FLOW_FIELDS[5:8],
    "Packet sizes": fmt.FLOW_FIELDS[8:14],
    "TCP flags": fmt.FLOW_FIELDS[14:19],
    "Other": fmt.FLOW_FIELDS[19:24],
}

ss = st.session_state
ss.setdefault("flow_result", None)

# The incident search page can send an incident here.
if ss.get("flow_prefill_incident"):
    ss.flow_tab = TAB_INCIDENT
    ss.flow_incident_id = ss.pop("flow_prefill_incident")
    ss.flow_result = call(lambda: get_client().explain({"incident_id": ss.flow_incident_id}), "Analysis")


def fill_from_incident() -> None:
    incident_id = (ss.get("fill_incident_id") or "").strip()
    try:
        incident = get_client().incident(incident_id)
    except ApiError as exc:
        ss.fill_message = ("error", exc.message)
        return
    values = fmt.metrics_to_form(incident.get("metrics", {}))
    for key, _, _ in fmt.FLOW_FIELDS:
        ss[f"flow_{key}"] = values.get(key, "")
    ss.fill_message = ("ok", f"Loaded the metrics of {incident['incident_id']} (recorded as "
                             f"{incident['historical_label']}). Change any value, then click Analyse flow.")


def clear_metrics() -> None:
    for key, _, _ in fmt.FLOW_FIELDS:
        ss[f"flow_{key}"] = ""
    ss.flow_result = None
    ss.pop("fill_message", None)


def pick_example() -> None:
    label = ss.get("example_type")
    if label:
        ss.flow_incident_id = fmt.EXAMPLE_INCIDENTS[label]


def analyse(flow) -> None:
    with st.spinner("Running the models…"):
        ss.flow_result = call(lambda: get_client().explain(flow), "Analysis")


page_header("Flow Analysis",
            "Get a fast, data-only verdict on one network flow: what kind of traffic it is, how risky and how "
            "unusual it is, how urgent it is, and which metrics drove the result.",
            how_to=["Choose a tab: analyse a **historical incident** by its ID, or **enter the metrics** of a flow "
                    "from your own logs (NetFlow, Zeek, firewall).",
                    "For your own flow, the five essential metrics are enough; more metrics give a more reliable "
                    "result. Units such as `2 s`, `1.2 KB` or `43k` are understood.",
                    "Click **Analyse**. The coloured banner gives the verdict and recommended action.",
                    "The charts under **Why the models decided this** show which metrics mattered most."],
            icon=":material/query_stats:")

incident_tab, metrics_tab = st.tabs([TAB_INCIDENT, TAB_METRICS], key="flow_tab", default=TAB_INCIDENT)

with incident_tab:
    st.caption("Every incident in the dataset has an ID such as INC-000035. Not sure which to try? Pick an "
               "attack type to load an example.")
    left, right = st.columns([2, 3])
    left.selectbox("Load an example by attack type", list(fmt.EXAMPLE_INCIDENTS), index=None, key="example_type",
                   on_change=pick_example, placeholder="Choose an attack type")
    with right.form("incident_form", border=False):
        incident_id = st.text_input("Incident ID", key="flow_incident_id", placeholder="INC-000035")
        if st.form_submit_button("Analyse incident", type="primary", icon=":material/play_arrow:"):
            if not fmt.INCIDENT_ID.match(incident_id.strip()):
                st.error("Enter an incident ID in the form INC-000035.")
            else:
                analyse({"incident_id": incident_id.strip()})

with metrics_tab:
    with st.container(border=True):
        st.markdown("**Optional: start from a real incident's values**")
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        c1.text_input("Incident ID to copy from", key="fill_incident_id", placeholder="INC-000006")
        c2.button("Load values", on_click=fill_from_incident, icon=":material/download:", width="stretch")
        c3.button("Clear all", on_click=clear_metrics, icon=":material/delete:", width="stretch")
        message = ss.get("fill_message")
        if message:
            (st.success if message[0] == "ok" else st.error)(message[1])

    with st.form("metrics_form"):
        st.markdown("**Essential metrics**")
        cols = st.columns(5)
        for col, (key, label, hint) in zip(cols, ESSENTIAL):
            col.text_input(label, key=f"flow_{key}", placeholder=hint)
        with st.expander("Additional metrics (optional, improves accuracy)", icon=":material/tune:"):
            for group, fields in OPTIONAL_GROUPS.items():
                st.caption(group)
                group_cols = st.columns(5)
                for i, (key, label, hint) in enumerate(fields):
                    group_cols[i % 5].text_input(label, key=f"flow_{key}", placeholder=hint)
        if st.form_submit_button("Analyse flow", type="primary", icon=":material/play_arrow:"):
            flow = fmt.parse_flow_inputs({key: ss.get(f"flow_{key}", "") for key, _, _ in fmt.FLOW_FIELDS})
            if not flow:
                st.error("Enter at least the essential metrics, or load values from an incident.")
            else:
                analyse(flow)

if ss.flow_result:
    st.divider()
    flow_result(ss.flow_result)
