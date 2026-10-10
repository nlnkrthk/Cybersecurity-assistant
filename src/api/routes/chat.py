"""
Assistant endpoints: ask a question (whole answer or live trace stream), read or
reset a conversation.

Handlers are plain `def` functions, so FastAPI runs the blocking graph in its
thread pool and a long investigation does not block other requests.
"""
import json
import logging
import re
import time
from typing import Any, Dict, Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.deps import AppState, get_assistant, get_state
from api.schemas import ChatRequest, ChatResponse, HistoryMessage, HistoryResponse, THREAD_ID_PATTERN

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["assistant"])

LLM_ERROR_MODULES = {
    "anthropic", "langchain_anthropic", "google", "langchain_google_genai", "openai", "langchain_openai",
}


def is_llm_provider_error(exc: Exception) -> bool:
    """Errors raised by an LLM provider SDK or its LangChain integration (auth, billing, quota, outage)."""
    return type(exc).__module__.split(".")[0] in LLM_ERROR_MODULES


def llm_error_detail(exc: Exception) -> str:
    message = getattr(exc, "message", None) or str(exc)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        message = body.get("error", {}).get("message", message)
    return f"The language model provider returned an error: {message}"


def _thread_id(request: ChatRequest, assistant) -> str:
    return request.thread_id or assistant.new_thread_id()


@router.post("", response_model=ChatResponse, summary="Ask the assistant")
def chat(body: ChatRequest, assistant=Depends(get_assistant), state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Answer one message. The response includes the agent trace and guardrail results.

    Pass the returned `thread_id` with the next message to continue the conversation
    (follow-up answers, "what about its risk?").
    """
    thread_id = _thread_id(body, assistant)
    started = time.perf_counter()
    with state.thread_lock(thread_id):
        try:
            result = assistant.ask(body.message, thread_id)
        except Exception as exc:
            if is_llm_provider_error(exc):
                raise HTTPException(502, llm_error_detail(exc))
            raise
    return {**result, "elapsed_ms": round((time.perf_counter() - started) * 1000)}


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post(
    "/stream",
    summary="Ask the assistant (live trace)",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Server-Sent Events"}},
)
def chat_stream(body: ChatRequest, assistant=Depends(get_assistant), state: AppState = Depends(get_state)):
    """Answer one message as Server-Sent Events.

    - `event: trace`  one per step (guardrail, route, agent start/end, tool result, final answer)
    - `event: result` the full response (same shape as POST /chat), sent last
    - `event: error`  if the run fails part-way
    """
    thread_id = _thread_id(body, assistant)

    def events() -> Iterator[str]:
        started = time.perf_counter()
        with state.thread_lock(thread_id):
            try:
                for kind, payload in assistant.stream(body.message, thread_id):
                    if kind == "event":
                        yield _sse("trace", payload)
                    else:
                        payload = {**payload, "elapsed_ms": round((time.perf_counter() - started) * 1000)}
                        yield _sse("result", payload)
            except Exception as exc:
                logger.exception("Streaming chat failed")
                detail = llm_error_detail(exc) if is_llm_provider_error(exc) else f"The assistant failed: {exc}"
                yield _sse("error", {"detail": detail, "thread_id": thread_id})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Thread-ID": thread_id},
    )


def _config(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _check_thread_id(thread_id: str) -> None:
    if not re.fullmatch(THREAD_ID_PATTERN, thread_id):
        raise HTTPException(422, "Invalid thread_id")


@router.get("/{thread_id}/history", response_model=HistoryResponse, summary="Conversation history")
def history(thread_id: str, assistant=Depends(get_assistant)) -> Dict[str, Any]:
    _check_thread_id(thread_id)
    values = assistant.graph.get_state(_config(thread_id)).values
    if not values.get("messages"):
        raise HTTPException(404, f"Conversation {thread_id} not found")
    messages = [
        HistoryMessage(role="user" if m.type == "human" else "assistant", content=m.text)
        for m in values["messages"]
    ]
    return {"thread_id": thread_id, "messages": messages}


@router.delete("/{thread_id}", status_code=204, summary="Delete a conversation")
def reset(thread_id: str, assistant=Depends(get_assistant)) -> None:
    _check_thread_id(thread_id)
    assistant.graph.checkpointer.delete_thread(thread_id)
