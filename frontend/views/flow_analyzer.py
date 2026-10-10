"""
Flow analyzer: attack type, intrusion risk, anomaly and priority of one flow, with the
SHAP features behind each prediction (POST /ml/explain, no LLM).
"""
import streamlit as st

import formatting as fmt
from api_client import ApiError
from components import call, flow_result, get_client

ss = st.session_state
ss.setdefault("flow_result", None)
ss.setdefault("flow_mode", "Historical incident")

# Another page (incident search) can send an incident here.
if ss.get("flow_prefill_incident"):
    ss.flow_mode = "Historical incident"
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
    ss.fill_message = ("ok", f"Filled with the metrics of {incident['incident_id']} "
                             f"(historical label {incident['historical_label']}). Edit any value and analyse.")


def clear_metrics() -> None:
    for key, _, _ in fmt.FLOW_FIELDS:
        ss[f"flow_{key}"] = ""
    ss.flow_result = None
    ss.pop("fill_message", None)


def pick_example() -> None:
    label = ss.get("example_type")
    if label:
        ss.flow_incident_id = fmt.EXAMPLE_INCIDENTS[label]


st.title("🔬 Flow analyzer")
st.caption("Runs the attack-type classifier, intrusion-risk model, anomaly detector and prioritiser on one "
           "network flow and explains each prediction with SHAP. No LLM involved.")

mode = st.segmented_control("Input", ["Historical incident", "Enter metrics"], key="flow_mode", required=True)

if mode == "Historical incident":
    left, right = st.columns([2, 3])
    left.selectbox("Example per attack type", list(fmt.EXAMPLE_INCIDENTS), index=None, key="example_type",
                   on_change=pick_example, placeholder="Pick an attack type")
    with right.form("incident_form", border=False):
        incident_id = st.text_input("Incident ID", key="flow_incident_id", placeholder="INC-000035")
        if st.form_submit_button("Analyse incident", type="primary"):
            if not fmt.INCIDENT_ID.match(incident_id.strip()):
                st.error("Use an incident ID such as INC-000035.")
            else:
                with st.spinner("Analysing…"):
                    ss.flow_result = call(lambda: get_client().explain({"incident_id": incident_id.strip()}),
                                          "Analysis")
else:
    st.markdown("Enter the metrics you know. Numbers or values with units (`2 s`, `1.2 KB`, `43k`) are accepted; "
                "missing metrics are derived or filled with typical values, and the result says how reliable it is.")
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    c1.text_input("Start from a historical incident", key="fill_incident_id", placeholder="INC-000006")
    c2.button("Fill metrics", on_click=fill_from_incident, width="stretch")
    c3.button("Clear", on_click=clear_metrics, width="stretch")
    message = ss.get("fill_message")
    if message:
        (st.success if message[0] == "ok" else st.error)(message[1])

    with st.form("metrics_form"):
        for group, fields in fmt.FIELD_GROUPS.items():
            st.markdown(f"**{group}**")
            cols = st.columns(4)
            for i, (key, label, hint) in enumerate(fields):
                cols[i % 4].text_input(label, key=f"flow_{key}", placeholder=hint)
        if st.form_submit_button("Analyse flow", type="primary"):
            flow = fmt.parse_flow_inputs({key: ss.get(f"flow_{key}", "") for key, _, _ in fmt.FLOW_FIELDS})
            if not flow:
                st.error("Enter at least a few metrics, or start from a historical incident.")
            else:
                with st.spinner("Analysing…"):
                    ss.flow_result = call(lambda: get_client().explain(flow), "Analysis")

if ss.flow_result:
    st.divider()
    flow_result(ss.flow_result)
