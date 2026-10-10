"""
Direct ML endpoints (no LLM): classification, risk, explanation, prioritization.

They call the same functions as the agent tools, so the API and the chat agree.
A flow with key metrics missing returns 200 with status "needs_follow_up" and the
question to ask; invalid or impossible values return 422.
"""
import io
import math
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from agents.tools import ml_tools
from api.deps import tool_response
from api.schemas import BatchRequest, FlowRequest
from services.feature_input import FEATURE_NAMES, _canonical_key

router = APIRouter(prefix="/ml", tags=["ml"])

MAX_CSV_BYTES = 5 * 1024 * 1024


def _flow_kwargs(body: FlowRequest) -> Dict[str, Any]:
    return body.model_dump(exclude_none=True)


@router.post("/classify", summary="Attack type of a flow")
def classify(body: FlowRequest) -> Dict[str, Any]:
    """Predicted attack type (15 classes incl. BENIGN), confidence and alternatives."""
    return tool_response(ml_tools.classify_attack_type(**_flow_kwargs(body)))


@router.post("/risk", summary="Intrusion risk and anomaly of a flow")
def risk(body: FlowRequest) -> Dict[str, Any]:
    """Calibrated intrusion probability, risk level, and anomaly level vs normal traffic."""
    return tool_response(ml_tools.assess_intrusion_risk(**_flow_kwargs(body)))


@router.post("/explain", summary="Full analysis with SHAP reasons")
def explain(body: FlowRequest) -> Dict[str, Any]:
    """Attack type, risk, anomaly, priority, and the metrics that drove each prediction (SHAP)."""
    return tool_response(ml_tools.explain_flow(**_flow_kwargs(body)))


@router.post("/prioritize", summary="Priority of one flow")
def prioritize(body: FlowRequest) -> Dict[str, Any]:
    """P1 (Critical) .. P4 (Low) with score, reason and rule overrides."""
    return tool_response(ml_tools.prioritize_single_flow(**_flow_kwargs(body)))


@router.post("/prioritize/batch", summary="Rank many flows / incidents")
def prioritize_batch(body: BatchRequest) -> Dict[str, Any]:
    """Rank flows and/or incident IDs by urgency; returns the top_n and counts per priority."""
    flows = [flow.model_dump(exclude_none=True) for flow in body.flows]
    return tool_response(ml_tools.prioritize_flows(flows=flows, incident_ids=body.incident_ids, top_n=body.top_n))


def _rows_from_csv(content: bytes) -> List[Dict[str, Any]]:
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(422, f"Could not read the CSV file: {exc}")
    if frame.empty:
        raise HTTPException(422, "The CSV file has no rows.")

    columns = {}
    for column in frame.columns:
        key = _canonical_key(str(column))
        if key in FEATURE_NAMES or key == "incident_id":
            columns[column] = key
    if not columns:
        raise HTTPException(
            422,
            "No known columns found. Use incident_id and/or flow metric columns such as "
            f"{', '.join(FEATURE_NAMES[:5])}, ...",
        )

    rows = []
    for record in frame[list(columns)].rename(columns=columns).to_dict(orient="records"):
        row = {}
        for key, value in record.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            row[key] = str(value).strip() if key == "incident_id" else value
        rows.append(row)
    return rows


@router.post("/prioritize/csv", summary="Rank flows from a CSV file")
async def prioritize_csv(
    file: UploadFile = File(description="CSV with an incident_id column and/or flow metric columns."),
    top_n: int = Query(default=50, ge=1, le=ml_tools.MAX_BATCH_FLOWS),
) -> Dict[str, Any]:
    """Upload a CSV (max 5 MB, 1000 rows). Each row is a flow; `flow_index` in the result is the row number."""
    content = await file.read(MAX_CSV_BYTES + 1)
    if len(content) > MAX_CSV_BYTES:
        raise HTTPException(413, "CSV file is larger than 5 MB.")
    rows = _rows_from_csv(content)
    if len(rows) > ml_tools.MAX_BATCH_FLOWS:
        raise HTTPException(422, f"Too many rows ({len(rows)}); the limit is {ml_tools.MAX_BATCH_FLOWS}.")
    result = await run_in_threadpool(ml_tools.prioritize_flows, flows=rows, incident_ids=[], top_n=top_n)
    return tool_response(result)
