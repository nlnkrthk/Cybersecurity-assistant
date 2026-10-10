"""
Health and feedback endpoints.
"""
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.deps import AppState, get_state
from api.schemas import FeedbackRequest, HealthResponse

router = APIRouter(tags=["system"])

VERSIONED_PACKAGES = ("fastapi", "langgraph", "langchain-anthropic", "scikit-learn", "shap")


def _versions() -> Dict[str, str]:
    out = {}
    for package in VERSIONED_PACKAGES:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = "not installed"
    return out


@router.get("/health", response_model=HealthResponse, summary="Service status")
def health(state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Whether the ML models, the assistant (LLM) and the search indexes are ready.

    `status` is "ok" when the models and the assistant are ready and the indexes are
    not in error; otherwise "degraded" (the direct ML endpoints may still work).
    """
    ml_ready = state.ml_service is not None
    ok = ml_ready and state.assistant is not None and state.search_index != "error"
    return {
        "status": "ok" if ok else "degraded",
        "ml_models_loaded": ml_ready,
        "attack_types": len(state.ml_service.list_attack_types()) if ml_ready else 0,
        "assistant_ready": state.assistant is not None,
        "assistant_error": state.assistant_error,
        "llm_model": state.llm_model,
        "search_index": state.search_index,
        "search_index_error": state.search_index_error,
        "versions": _versions(),
    }


@router.post("/feedback", status_code=501, summary="Rate an answer (coming soon)")
def feedback(body: FeedbackRequest) -> JSONResponse:
    """Thumbs up/down and an optional corrected attack type for an answer. Not stored yet."""
    return JSONResponse(status_code=501, content={"detail": "Feedback storage is not implemented yet."})
