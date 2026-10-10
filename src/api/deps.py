"""
Shared objects of the running API: the assistant graph, the ML service and the
search-index loading state. Created once at startup (see api.main.lifespan).
"""
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

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
