"""
Analyst feedback: store (SQLite), turn snapshots, and corrections for retraining.

    get_feedback_store()        shared store (path: FEEDBACK_DB_PATH env var or config.FEEDBACK_DB_FILE)
    corrections_for_incident()  analyst-corrected labels for an incident, without creating a database
"""
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FEEDBACK_DB_FILE  # noqa: E402
from feedback.store import FeedbackStore  # noqa: E402

_store: Optional[FeedbackStore] = None
_pinned = False  # set_feedback_store() was used: keep that store regardless of the path
_lock = threading.Lock()


def feedback_db_path() -> Path:
    return Path(os.getenv("FEEDBACK_DB_PATH") or FEEDBACK_DB_FILE)


def get_feedback_store() -> FeedbackStore:
    """The shared store; created (with its database file) on first use."""
    global _store
    path = feedback_db_path()
    with _lock:
        if _store is None or (not _pinned and _store.path != path):
            _store = FeedbackStore(path)
        return _store


def set_feedback_store(store: Optional[FeedbackStore]) -> None:
    """Use this store everywhere (tests); None goes back to the configured path."""
    global _store, _pinned
    with _lock:
        _store, _pinned = store, store is not None


def corrections_for_incident(incident_id: str) -> Dict[str, int]:
    """{label: count} of analyst corrections; {} when there is no feedback database yet."""
    if _store is None and not feedback_db_path().exists():
        return {}
    try:
        return get_feedback_store().corrections_for(incident_id)
    except Exception:
        return {}


__all__ = ["FeedbackStore", "get_feedback_store", "set_feedback_store", "corrections_for_incident", "feedback_db_path"]
