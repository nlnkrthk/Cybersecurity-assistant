"""
User accounts: sign-up, login sessions and ownership of assistant conversations.

    auth_db_path()   AUTH_DB_PATH env var or config.AUTH_DB_FILE
    chat_db_path()   CHAT_DB_PATH env var or config.CHAT_DB_FILE (saved conversations)
    open_checkpointer(path)  LangGraph SQLite checkpointer for saved conversations
"""
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import AUTH_DB_FILE, CHAT_DB_FILE  # noqa: E402
from auth.security import AuthError, LoginLimiter  # noqa: E402
from auth.store import AuthStore, UsernameTaken, make_title  # noqa: E402


def auth_db_path() -> Path:
    return Path(os.getenv("AUTH_DB_PATH") or AUTH_DB_FILE)


def chat_db_path() -> Path:
    return Path(os.getenv("CHAT_DB_PATH") or CHAT_DB_FILE)


def open_checkpointer(path: Path):
    """A SQLite checkpointer, so conversations survive API restarts. One connection shared by the
    API's worker threads (the saver locks around each call; each conversation is also locked)."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    saver.setup()
    return saver


__all__ = ["AuthError", "AuthStore", "LoginLimiter", "UsernameTaken", "auth_db_path", "chat_db_path",
           "make_title", "open_checkpointer"]
