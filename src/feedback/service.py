"""
Record analyst feedback for an assistant turn (used by the API and the chat script).
"""
from typing import Any, Dict, Optional

from feedback import get_feedback_store
from feedback.snapshot import build_snapshot, has_flow


class FeedbackError(ValueError):
    """Invalid feedback (e.g. a label correction for an answer that analysed no flow)."""


def record_feedback(
    assistant,
    thread_id: str,
    rating: str,
    turn_id: Optional[str] = None,
    comment: Optional[str] = None,
    corrected_label: Optional[str] = None,
    llm_model: Optional[str] = None,
    store=None,
) -> Dict[str, Any]:
    """Store feedback for a turn (default: the conversation's latest turn).

    The question, answer and prediction are read from the conversation store, so feedback
    always refers to an answer the assistant really gave. Raises KeyError for an unknown
    conversation or turn and FeedbackError for invalid feedback.
    """
    from agents.judge import last_question
    from agents.tools.ml_tools import match_attack_type

    if llm_model is None:
        try:
            from agents.llm import llm_settings

            settings = llm_settings()
            llm_model = f"{settings['provider']}:{settings['model']}"
        except Exception:
            llm_model = None

    state = assistant.turn_state(thread_id, turn_id)
    snapshot = build_snapshot(state, last_question(state.get("messages", [])), llm_model)

    label = None
    if corrected_label:
        if not has_flow(snapshot):
            raise FeedbackError("This answer analysed no flow or incident, so there is no attack type to correct.")
        try:
            label = match_attack_type(corrected_label)
        except ValueError as exc:
            raise FeedbackError(str(exc)) from exc

    return (store or get_feedback_store()).add(
        thread_id=thread_id,
        turn_id=state.get("turn_id") or "legacy",
        rating=rating,
        snapshot=snapshot,
        comment=comment,
        corrected_label=label,
    )
