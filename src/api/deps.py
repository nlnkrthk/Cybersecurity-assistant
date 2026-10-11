"""
Shared objects of the running API: the assistant graph, the ML service, the user store and
the search-index loading state. Created once at startup (see api.main.lifespan).
"""
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

TOOL_NOT_FOUND = ("not found", "unknown or ambiguous attack type")


@dataclass
class AppState:
    assistant: Any = None                     # agents.graph.Assistant, None if the LLM is not configured
    assistant_error: Optional[str] = None
    llm_model: Optional[str] = None
    ml_service: Any = None
    search_index: str = "not_loaded"          # not_loaded | loading | ready | error
    search_index_error: Optional[str] = None
    auth_store: Any = None                    # auth.AuthStore: users, sessions, conversation owners
    login_limiter: Any = None                 # auth.LoginLimiter
    _thread_locks: Dict[str, threading.Lock] = field(default_factory=lambda: defaultdict(threading.Lock))
    _locks_guard: threading.Lock = field(default_factory=threading.Lock)

    def thread_lock(self, thread_id: str) -> threading.Lock:
        """One request at a time per conversation, so turns are not interleaved."""
        with self._locks_guard:
            return self._thread_locks[thread_id]

    def load_search_index(self) -> None:
        """Load the BM25 index, Chroma store and embedder (slow; run in a background thread)."""
        from agents.tools.search_tools import _get_searcher

        self.search_index = "loading"
        try:
            _get_searcher()
            self.search_index = "ready"
            logger.info("Search indexes ready")
        except Exception as exc:
            self.search_index, self.search_index_error = "error", str(exc)
            logger.exception("Search indexes failed to load")


def get_state(request: Request) -> AppState:
    return request.app.state.services


def get_assistant(request: Request):
    state = get_state(request)
    if state.assistant is None:
        raise HTTPException(503, f"The assistant is not available: {state.assistant_error or 'not initialised'}")
    return state.assistant


bearer = HTTPBearer(auto_error=False, description="Token from POST /auth/login or /auth/signup.")


def get_optional_user(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)
) -> Optional[Dict[str, Any]]:
    """The signed-in user, or None when no valid token was sent."""
    return get_state(request).auth_store.user_for_token(credentials.credentials) if credentials else None


def get_current_user(user: Optional[Dict[str, Any]] = Depends(get_optional_user)) -> Dict[str, Any]:
    """The signed-in user (Authorization: Bearer <token>); 401 when missing, expired or logged out."""
    if user is None:
        raise HTTPException(401, "Sign in required (missing, expired or invalid token).",
                            headers={"WWW-Authenticate": "Bearer"})
    return user


def require_conversation_owner(state: AppState, thread_id: str, user: Dict[str, Any]) -> None:
    """404 unless the conversation belongs to the user (the same answer as for an unknown
    conversation, so other users' conversation IDs cannot be discovered)."""
    if state.auth_store.conversation_owner(thread_id) != user["id"]:
        raise HTTPException(404, f"Conversation {thread_id} not found")


def tool_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a tool result, turning {"status": "error"} into the matching HTTP error."""
    if result.get("status") != "error":
        return result
    message = str(result.get("error", "Unknown error"))
    lowered = message.lower()
    if lowered.startswith("internal error"):
        raise HTTPException(500, message)
    if any(marker in lowered for marker in TOOL_NOT_FOUND):
        raise HTTPException(404, message)
    raise HTTPException(422, message)
