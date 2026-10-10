"""
LLM-as-Judge endpoint: grade an answer against the evidence it was written from.
"""
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from agents.judge import judge_answer
from api.deps import get_assistant
from api.routes.chat import is_llm_provider_error, llm_error_detail
from api.schemas import JudgeRequest, JudgeResponse

router = APIRouter(tags=["assistant"])


def _judge_model_name() -> str:
    try:
        from agents.llm import judge_model_settings

        settings = judge_model_settings()
        return f"{settings['provider']}:{settings['model']}"
    except Exception:
        return "custom"


@router.post("/judge", response_model=JudgeResponse, summary="Grade an answer (LLM-as-Judge)")
def judge(body: JudgeRequest, assistant=Depends(get_assistant)) -> Dict[str, Any]:
    """Score an answer 1-5 on faithfulness, correctness, relevance, actionability and safety,
    list unsupported claims, and give a verdict (pass / needs_review / fail).

    Send a `thread_id` to grade that conversation's latest answer with its real evidence, or a
    `question` + `answer` (optionally with `findings` and `evidence`) to grade any answer.
    """
    try:
        if body.thread_id:
            try:
                result = assistant.judge_last_turn(body.thread_id)
            except KeyError as exc:
                raise HTTPException(404, str(exc.args[0]))
            except ValueError as exc:
                raise HTTPException(422, str(exc))
        else:
            if not body.question or not body.answer:
                raise HTTPException(422, "Provide a thread_id, or both question and answer.")
            result = judge_answer(
                assistant.judge_llm(),
                question=body.question,
                answer=body.answer,
                route=body.route,
                findings=body.findings,
                evidence=body.evidence,
            )
    except HTTPException:
        raise
    except Exception as exc:
        if is_llm_provider_error(exc):
            raise HTTPException(502, llm_error_detail(exc))
        raise
    return {**result, "judge_model": _judge_model_name()}
