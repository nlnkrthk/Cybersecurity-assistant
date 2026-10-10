"""
Retrieval and lookup endpoints (no LLM): hybrid incident search, incidents, attack types.
"""
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from agents.tools import ml_tools
from agents.tools.search_tools import search_similar_incidents
from api.deps import AppState, get_state, tool_response
from api.schemas import SearchRequest

router = APIRouter(tags=["search"])


@router.post("/search", summary="Find similar historical incidents")
def search(body: SearchRequest, state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Hybrid keyword + semantic search over ~97,000 incidents, with optional attack-type and
    port filters and numeric conditions in the query ("more than 10000 packets per second").
    Retrieval only: no generated answer.

    The first search waits for the indexes if they are still loading (see GET /health).
    """
    if state.search_index == "error":
        raise HTTPException(503, f"Search indexes are not available: {state.search_index_error}")
    return tool_response(search_similar_incidents(**body.model_dump(exclude_none=True)))


@router.get("/incidents/{incident_id}", summary="Look up a historical incident")
def incident(incident_id: str) -> Dict[str, Any]:
    """Stored metrics, destination port and historical label of an incident (e.g. INC-000035)."""
    return tool_response(ml_tools.lookup_incident(incident_id=incident_id))


@router.get("/attack-types", summary="List attack types")
def attack_types() -> Dict[str, Any]:
    """The 15 classes the models know (incl. BENIGN) with their severity."""
    return tool_response(ml_tools.list_attack_types())


@router.get("/attack-types/{name}", summary="Profile of an attack type")
def attack_type_profile(name: str) -> Dict[str, Any]:
    """Typical traffic of an attack type: which metrics differ from normal traffic, with medians."""
    return tool_response(ml_tools.get_attack_profile(attack_type=name))
