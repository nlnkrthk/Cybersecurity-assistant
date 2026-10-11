"""
Analyst feedback endpoints: rate answers, correct attack labels, see stats, export corrections.
"""
import io
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.deps import AppState, get_assistant, get_current_user, get_state, require_conversation_owner
from api.schemas import FeedbackRequest

router = APIRouter(prefix="/feedback", tags=["feedback"])


def _store():
    from feedback import get_feedback_store

    return get_feedback_store()


@router.post("", status_code=201, summary="Rate an answer")
def add_feedback(body: FeedbackRequest, user: Dict[str, Any] = Depends(get_current_user),
                 assistant=Depends(get_assistant), state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Thumbs up/down, an optional comment and an optional corrected attack type for an answer.

    The question, answer and model prediction are taken from the conversation itself. Rating
    the same turn again updates the earlier feedback. A corrected label is shown in later
    answers about the same incident and is used when the classifier is retrained.
    Only the owner of the conversation can rate it; the rating records their username.
    """
    from feedback.service import FeedbackError, record_feedback

    require_conversation_owner(state, body.thread_id, user)

    try:
        return record_feedback(
            assistant,
            thread_id=body.thread_id,
            turn_id=body.turn_id,
            rating=body.rating,
            comment=body.comment,
            corrected_label=body.corrected_label,
            llm_model=state.llm_model,
            username=user["username"],
            store=_store(),
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0]) if exc.args else "Not found")
    except FeedbackError as exc:
        raise HTTPException(422, str(exc))


@router.get("", summary="List feedback")
def list_feedback(
    rating: Optional[Literal["up", "down"]] = None,
    route: Optional[str] = None,
    label: Optional[str] = Query(default=None, description="Matches the model prediction or the correction."),
    corrected_only: bool = False,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> List[Dict[str, Any]]:
    """Stored feedback, newest first."""
    return _store().list(rating=rating, route=route, label=label, corrected_only=corrected_only,
                         limit=limit, offset=offset)


@router.get("/stats", summary="Feedback statistics")
def feedback_stats() -> Dict[str, Any]:
    """Approval rate (overall and per route), agreement between the LLM judge and analysts,
    the most common prediction -> correction pairs, and recent comments."""
    return _store().stats()


@router.get(
    "/export.csv",
    summary="Export label corrections",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/csv": {}}}},
)
def export_corrections():
    """Corrected flows for retraining: 24 features, corrected label (majority of analysts),
    source incident, votes. Corrections without usable features are left out."""
    from feedback.retraining import collect_corrections

    frame, _ = collect_corrections(_store())
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=feedback_corrections.csv"},
    )
