"""
Snapshot of one assistant turn for the feedback store, built from the conversation state
(never from client-supplied text): question, answer, route, model prediction, incident or
flow, input quality, judge verdict.
"""
from typing import Any, Dict, List, Optional

# Tools whose result describes one flow's predicted attack type.
PREDICTION_TOOLS = ("explain_flow", "classify_attack_type", "prioritize_flow")


def _single_flow_prediction(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prediction details of the flow analysed this turn (None values if no single flow)."""
    found: Dict[str, Any] = {"predicted_label": None, "prediction_confidence": None, "incident_id": None,
                             "historical_label": None, "input_quality": None}
    for tool in PREDICTION_TOOLS:
        for item in evidence:
            result = item.get("result") or {}
            if item.get("tool") != tool or result.get("status") != "ok":
                continue
            attack = result.get("attack_type") or {}
            incident = result.get("incident") or {}
            found.update({
                "predicted_label": attack.get("prediction"),
                "prediction_confidence": attack.get("confidence"),
                "incident_id": incident.get("incident_id"),
                "historical_label": incident.get("historical_label"),
                "input_quality": (result.get("input") or {}).get("input_quality"),
            })
            return found
    # A lookup without a prediction still identifies the incident.
    for item in evidence:
        result = item.get("result") or {}
        if item.get("tool") == "lookup_incident" and result.get("status") == "ok":
            found.update({"incident_id": result.get("incident_id"), "historical_label": result.get("historical_label")})
            break
    return found


def build_snapshot(state: Dict[str, Any], question: str, llm_model: Optional[str] = None) -> Dict[str, Any]:
    evidence = state.get("evidence") or []
    judgement = state.get("judgement") or {}
    snapshot = {
        "question": question,
        "answer": state.get("final_answer", ""),
        "route": state.get("route"),
        "agents_run": state.get("completed", []),
        "flow_args": state.get("flow_args"),
        "judge_verdict": judgement.get("verdict"),
        "judge_overall": judgement.get("overall"),
        "llm_model": llm_model,
        **_single_flow_prediction(evidence),
    }
    if not snapshot["incident_id"] and isinstance(snapshot["flow_args"], dict):
        snapshot["incident_id"] = snapshot["flow_args"].get("incident_id")
    return snapshot


def has_flow(snapshot: Dict[str, Any]) -> bool:
    """True when the turn analysed one flow or incident (a label correction makes sense)."""
    return bool(snapshot.get("incident_id") or snapshot.get("flow_args"))
