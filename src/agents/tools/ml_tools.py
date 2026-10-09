"""
LangGraph tools around the ML service and prioritization.

Every flow tool accepts EITHER flow metrics (any subset of the 24 features, numbers
or unit strings) OR an incident_id from the historical dataset. Missing metrics are
derived or filled and flagged by services.feature_input; if key metrics are missing
the tool returns status "needs_follow_up" with a question for the user.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from agents.tools.common import compact_input, compact_result, error_result, rounded, safe_tool
from services.feature_input import FEATURE_NAMES, FlowFeatures, NormalizedFlow, get_incident_features
from services.ml_service import get_ml_service
from services.prioritization import SEVERITY, prioritize, prioritize_batch

logger = logging.getLogger(__name__)

MAX_BATCH_FLOWS = 1000


# ============================================================================
# Input schemas
# ============================================================================
class FlowArgs(FlowFeatures):
    """Flow metrics (any subset) or a historical incident ID."""

    model_config = ConfigDict(extra="ignore")

    incident_id: Optional[str] = Field(
        default=None,
        description=(
            "ID of a historical incident, e.g. 'INC-000123'. When given, the incident's stored "
            "metrics are used and any metric fields are ignored."
        ),
    )


class BatchFlowArgs(BaseModel):
    flows: List[FlowArgs] = Field(default_factory=list, description="Flows to rank, each with its own metrics.")
    incident_ids: List[str] = Field(default_factory=list, description="Historical incident IDs to rank.")
    top_n: int = Field(default=10, ge=1, le=100, description="How many of the most urgent flows to return.")


class AttackTypeArgs(BaseModel):
    attack_type: str = Field(description="Attack type name, e.g. 'DDoS', 'PortScan', 'SSH-Patator', 'Web Attack - XSS'.")


class IncidentArgs(BaseModel):
    incident_id: str = Field(description="Historical incident ID, e.g. 'INC-000123'.")


# ============================================================================
# Helpers
# ============================================================================
def _resolve_flow(args: Dict[str, Any]) -> Tuple[Union[Dict[str, Any], NormalizedFlow], Optional[Dict[str, Any]]]:
    """Return (flow input for the ML service, incident context or None)."""
    incident_id = args.pop("incident_id", None)
    if incident_id:
        incident = get_incident_features(incident_id)
        context = {
            "incident_id": incident.incident_id,
            "historical_label": incident.historical_label,
            "dst_port": incident.dst_port,
        }
        return incident.flow, context
    metrics = {k: v for k, v in args.items() if v is not None and k in FEATURE_NAMES}
    if not metrics:
        raise ValueError("Provide at least some flow metrics or an incident_id.")
    return metrics, None


def _with_context(result: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = compact_result(result)
    if context:
        out = {"incident": context, **out}
    return out


def match_attack_type(name: str) -> str:
    """Case-insensitive exact or unique-substring match against the known attack types."""
    known = get_ml_service().list_attack_types()
    wanted = " ".join(str(name).replace("_", " ").split()).lower()
    for attack in known:
        if attack.lower() == wanted:
            return attack
    matches = [a for a in known if wanted and wanted in a.lower()]
    if len(matches) == 1:
        return matches[0]
    hint = f" Matches: {matches}." if matches else ""
    raise ValueError(f"Unknown or ambiguous attack type {name!r}.{hint} Known types: {known}")


# ============================================================================
# Tool functions
# ============================================================================
@safe_tool
def classify_attack_type(**kwargs) -> Dict[str, Any]:
    flow, context = _resolve_flow(kwargs)
    return _with_context(get_ml_service().classify(flow), context)


@safe_tool
def assess_intrusion_risk(**kwargs) -> Dict[str, Any]:
    flow, context = _resolve_flow(kwargs)
    service = get_ml_service()
    risk = service.predict_risk(flow)
    if risk["status"] != "ok":
        return _with_context(risk, context)
    anomaly = service.detect_anomaly(flow)
    return _with_context({**risk, "anomaly": anomaly["anomaly"]}, context)


@safe_tool
def explain_flow(**kwargs) -> Dict[str, Any]:
    flow, context = _resolve_flow(kwargs)
    result = get_ml_service().explain(flow)
    if result["status"] == "ok":
        result["priority"] = prioritize(result)
    return _with_context(result, context)


@safe_tool
def prioritize_single_flow(**kwargs) -> Dict[str, Any]:
    flow, context = _resolve_flow(kwargs)
    result = get_ml_service().predict(flow)
    result["priority"] = prioritize(result)
    return _with_context(result, context)


@safe_tool
def prioritize_flows(flows: List[FlowArgs] = None, incident_ids: List[str] = None, top_n: int = 10) -> Dict[str, Any]:
    flows, incident_ids = flows or [], incident_ids or []
    total = len(flows) + len(incident_ids)
    if total == 0:
        return error_result("Provide flows and/or incident_ids to prioritize.")
    if total > MAX_BATCH_FLOWS:
        return error_result(f"Too many flows ({total}); the limit is {MAX_BATCH_FLOWS} per call.")

    inputs: List[Any] = []
    labels: List[Dict[str, Any]] = []
    for i, flow in enumerate(flows):
        data = flow.model_dump() if isinstance(flow, BaseModel) else dict(flow)
        if data.get("incident_id"):
            incident_ids = incident_ids + [data["incident_id"]]
            continue
        inputs.append({k: v for k, v in data.items() if v is not None and k in FEATURE_NAMES})
        labels.append({"flow_index": i})
    for incident_id in incident_ids:
        incident = get_incident_features(incident_id)
        inputs.append(incident.flow)
        labels.append({"incident_id": incident.incident_id, "historical_label": incident.historical_label})

    ranked = prioritize_batch(inputs)
    counts: Dict[str, int] = {}
    items = []
    for item in ranked:
        priority = item["priority"]
        key = priority["priority"] or "needs_follow_up"
        counts[key] = counts.get(key, 0) + 1
        entry = {**labels[item["index"]], "priority": priority["priority"], "score": priority["score"]}
        if item["status"] == "ok":
            entry.update({
                "attack_type": item["attack_type"]["prediction"],
                "confidence": item["attack_type"]["confidence"],
                "risk_probability": item["risk"]["probability"],
                "anomaly_level": item["anomaly"]["level"],
                "reason": priority["reason"],
                "overrides": priority["overrides_applied"],
                "input_quality": item["input"]["input_quality"],
            })
        else:
            entry["follow_up_question"] = item["follow_up_question"]
        items.append(entry)

    return rounded({
        "status": "ok",
        "total_flows": total,
        "priority_counts": dict(sorted(counts.items())),
        "top": items[:top_n],
    })


@safe_tool
def get_attack_profile(attack_type: str) -> Dict[str, Any]:
    name = match_attack_type(attack_type)
    if name == "BENIGN":
        return error_result("BENIGN is normal traffic; profiles exist only for attack types.")
    profile = get_ml_service().attack_profile(name)
    severity, _ = SEVERITY.get(name, ("High", 0.8))
    return rounded({
        "status": "ok",
        "attack_type": name,
        "severity": severity,
        "summary": profile["summary"],
        "top_features": [
            {
                "feature": f["display_name"],
                "direction_vs_normal": f["direction_vs_normal"],
                "attack_median": f["attack_median"],
                "normal_median": f["benign_median"],
                "unit": f["unit"],
                "importance_share": f["shap_share"],
            }
            for f in profile["top_features"]
        ],
    })


@safe_tool
def list_attack_types() -> Dict[str, Any]:
    return {
        "status": "ok",
        "attack_types": [
            {"attack_type": name, "severity": SEVERITY.get(name, ("High", 0.8))[0]}
            for name in get_ml_service().list_attack_types()
        ],
    }


@safe_tool
def lookup_incident(incident_id: str) -> Dict[str, Any]:
    incident = get_incident_features(incident_id)
    service = get_ml_service()
    metrics = {
        service.reference["features"][name]["display_name"]: incident.flow.features[name]
        for name in FEATURE_NAMES
        if name not in incident.flow.filled
    }
    return rounded({
        "status": "ok",
        "incident_id": incident.incident_id,
        "historical_label": incident.historical_label,
        "dst_port": incident.dst_port,
        "input": compact_input(incident.flow.to_dict()),
        "metrics": metrics,
    })


# ============================================================================
# Tool objects
# ============================================================================
_FLOW_HINT = (
    " Pass the flow metrics you know (numbers or strings with units such as '2 s', '1.5 KB', '43k'), "
    "or an incident_id. Never invent metric values. If the result has status 'needs_follow_up', "
    "ask the user the returned follow_up_question."
)

classify_attack_type_tool = StructuredTool.from_function(
    func=classify_attack_type,
    name="classify_attack_type",
    description=(
        "Predict the attack type of a network flow (15 classes incl. BENIGN, DDoS, DoS variants, PortScan, "
        "SSH/FTP brute force, Web attacks, Bot, Infiltration, Heartbleed) with confidence and alternatives."
        + _FLOW_HINT
    ),
    args_schema=FlowArgs,
)

assess_intrusion_risk_tool = StructuredTool.from_function(
    func=assess_intrusion_risk,
    name="assess_intrusion_risk",
    description=(
        "Estimate how likely a flow is malicious (calibrated probability, risk level Low/Medium/High/Critical) "
        "and how unusual it is compared with normal traffic (anomaly level, most unusual metrics). "
        "Use for 'is this suspicious?' questions." + _FLOW_HINT
    ),
    args_schema=FlowArgs,
)

explain_flow_tool = StructuredTool.from_function(
    func=explain_flow,
    name="explain_flow",
    description=(
        "Full analysis of one flow: attack type, intrusion risk, anomaly, P1-P4 priority, and the SHAP reasons "
        "(which metrics drove each prediction) plus a summary. Use when the user wants to investigate or "
        "understand a flow." + _FLOW_HINT
    ),
    args_schema=FlowArgs,
)

prioritize_single_flow_tool = StructuredTool.from_function(
    func=prioritize_single_flow,
    name="prioritize_flow",
    description=(
        "Priority (P1 Critical .. P4 Low) for one flow with score, reason and any rule overrides, "
        "based on attack-type severity, intrusion risk and anomaly level." + _FLOW_HINT
    ),
    args_schema=FlowArgs,
)

prioritize_flows_tool = StructuredTool.from_function(
    func=prioritize_flows,
    name="prioritize_flows",
    description=(
        "Rank many flows and/or historical incidents by urgency and return the most urgent ones with "
        "priority counts. Use for 'which of these should I investigate first?'."
    ),
    args_schema=BatchFlowArgs,
)

get_attack_profile_tool = StructuredTool.from_function(
    func=get_attack_profile,
    name="get_attack_profile",
    description=(
        "Typical network-traffic characteristics of an attack type (which metrics are higher or lower than "
        "normal traffic, with medians), derived from the trained model's SHAP analysis, plus its severity."
    ),
    args_schema=AttackTypeArgs,
)

list_attack_types_tool = StructuredTool.from_function(
    func=list_attack_types,
    name="list_attack_types",
    description="List the attack types the models know, with their severity.",
)

lookup_incident_tool = StructuredTool.from_function(
    func=lookup_incident,
    name="lookup_incident",
    description="Get the stored metrics, destination port and historical label of an incident by ID.",
    args_schema=IncidentArgs,
)

ML_TOOLS = [
    classify_attack_type_tool,
    assess_intrusion_risk_tool,
    explain_flow_tool,
    prioritize_single_flow_tool,
    prioritize_flows_tool,
    get_attack_profile_tool,
    list_attack_types_tool,
    lookup_incident_tool,
]
