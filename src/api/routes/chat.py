"""
Assistant endpoints: ask a question (whole answer or live trace stream), list, read,
rename or delete your conversations. All of them need a signed-in user, and a
conversation can only be used by the user who started it.

Handlers are plain `def` functions, so FastAPI runs the blocking graph in its
thread pool and a long investigation does not block other requests.
"""
import json
import logging
import re
import time
from typing import Any, Dict, Iterator, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.deps import AppState, get_assistant, get_current_user, get_state, require_conversation_owner
from api.schemas import (
    THREAD_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    ConversationSummary,
    HistoryMessage,
    HistoryResponse,
    RenameRequest,
)
from auth import make_title

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


def _messages(assistant, thread_id: str) -> list:
    return assistant.graph.get_state(_config(thread_id)).values.get("messages") or []


def _open_conversation(body: ChatRequest, assistant, state: AppState, user: Dict[str, Any]) -> str:
    """The conversation to answer in: a new one, or one of the user's own (else 404)."""
    store = state.auth_store
    if body.thread_id is None:
        thread_id = assistant.new_thread_id()
        store.claim_conversation(thread_id, user["id"])
        return thread_id
    thread_id = body.thread_id
    owner = store.conversation_owner(thread_id)
    if owner == user["id"]:
        return thread_id
    # A client-chosen new ID is fine, as long as nobody has used it yet.
    if owner is None and not _messages(assistant, thread_id) and store.claim_conversation(thread_id, user["id"]):
        return thread_id
    raise HTTPException(404, f"Conversation {thread_id} not found")


def _after_turn(state: AppState, assistant, thread_id: str) -> None:
    """Mark the conversation as used; the first question (after secret redaction) becomes its title."""
    try:
        first = next((m.text for m in _messages(assistant, thread_id) if m.type == "human"), None)
        state.auth_store.touch_conversation(thread_id, make_title(first) if first else None)
    except Exception:
        logger.exception("Could not update the conversation")


@router.post("", response_model=ChatResponse, summary="Ask the assistant")
def chat(body: ChatRequest, user: Dict[str, Any] = Depends(get_current_user), assistant=Depends(get_assistant),
         state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Answer one message. The response includes the agent trace and guardrail results.

    Pass the returned `thread_id` with the next message to continue the conversation
    (follow-up answers, "what about its risk?").
    """
    thread_id = _open_conversation(body, assistant, state, user)
    started = time.perf_counter()
    with state.thread_lock(thread_id):
        try:
            result = assistant.ask(body.message, thread_id, judge=body.judge)
        except Exception as exc:
            if is_llm_provider_error(exc):
                raise HTTPException(502, llm_error_detail(exc))
            raise
        finally:
            _after_turn(state, assistant, thread_id)
    return {**result, "elapsed_ms": round((time.perf_counter() - started) * 1000)}


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post(
    "/stream",
    summary="Ask the assistant (live trace)",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Server-Sent Events"}},
)
def chat_stream(body: ChatRequest, user: Dict[str, Any] = Depends(get_current_user),
                assistant=Depends(get_assistant), state: AppState = Depends(get_state)):
    """Answer one message as Server-Sent Events.

    - `event: trace`  one per step (guardrail, route, agent start/end, tool result, final answer)
    - `event: result` the full response (same shape as POST /chat), sent last
    - `event: error`  if the run fails part-way
    """
    thread_id = _open_conversation(body, assistant, state, user)

    def events() -> Iterator[str]:
        started = time.perf_counter()
        with state.thread_lock(thread_id):
            try:
                for kind, payload in assistant.stream(body.message, thread_id, judge=body.judge):
                    if kind == "event":
                        yield _sse("trace", payload)
                    else:
                        payload = {**payload, "elapsed_ms": round((time.perf_counter() - started) * 1000)}
                        yield _sse("result", payload)
            except Exception as exc:
                logger.exception("Streaming chat failed")
                detail = llm_error_detail(exc) if is_llm_provider_error(exc) else f"The assistant failed: {exc}"
                yield _sse("error", {"detail": detail, "thread_id": thread_id})
            finally:
                _after_turn(state, assistant, thread_id)

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


@router.get("/conversations", response_model=List[ConversationSummary], summary="Your conversations")
def conversations(user: Dict[str, Any] = Depends(get_current_user),
                  state: AppState = Depends(get_state)) -> List[Dict[str, Any]]:
    """Your saved conversations, most recently used first."""
    return state.auth_store.list_conversations(user["id"])


@router.get("/{thread_id}/history", response_model=HistoryResponse, summary="Conversation history")
def history(thread_id: str, user: Dict[str, Any] = Depends(get_current_user), assistant=Depends(get_assistant),
            state: AppState = Depends(get_state)) -> Dict[str, Any]:
    _check_thread_id(thread_id)
    require_conversation_owner(state, thread_id, user)
    values = assistant.graph.get_state(_config(thread_id)).values
    if not values.get("messages"):
        raise HTTPException(404, f"Conversation {thread_id} not found")
    messages = [
        HistoryMessage(role="user" if m.type == "human" else "assistant", content=m.text)
        for m in values["messages"]
    ]
    return {"thread_id": thread_id, "messages": messages}


@router.patch("/{thread_id}", response_model=ConversationSummary, summary="Rename a conversation")
def rename(thread_id: str, body: RenameRequest, user: Dict[str, Any] = Depends(get_current_user),
           state: AppState = Depends(get_state)) -> Dict[str, Any]:
    _check_thread_id(thread_id)
    require_conversation_owner(state, thread_id, user)
    if not body.title.strip():
        raise HTTPException(422, "The title must not be empty.")
    state.auth_store.rename_conversation(thread_id, body.title)
    return state.auth_store.conversation(thread_id)


@router.delete("/{thread_id}", status_code=204, summary="Delete a conversation")
def reset(thread_id: str, user: Dict[str, Any] = Depends(get_current_user), assistant=Depends(get_assistant),
          state: AppState = Depends(get_state)) -> None:
    _check_thread_id(thread_id)
    require_conversation_owner(state, thread_id, user)
    with state.thread_lock(thread_id):
        assistant.graph.checkpointer.delete_thread(thread_id)
        state.auth_store.delete_conversation(thread_id)
