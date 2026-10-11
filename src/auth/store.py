"""
SQLite store for user accounts, login sessions and conversation ownership.

    users          username (unique, case-insensitive) and scrypt password hash
    sessions       SHA-256 digest of each login token, with its expiry (the token itself is never stored)
    conversations  which user owns which assistant conversation (thread_id), with a title

The conversation messages themselves live in the LangGraph checkpointer (agents.graph).
"""
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from auth.security import (
    DUMMY_HASH,
    AuthError,
    hash_password,
    new_token,
    token_digest,
    validate_password,
    validate_username,
    verify_password,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    thread_id  TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations (user_id, updated_at);
"""

NEW_CONVERSATION_TITLE = "New conversation"
TITLE_MAX = 80


class UsernameTaken(AuthError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="microseconds")


def make_title(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return NEW_CONVERSATION_TITLE
    return text if len(text) <= TITLE_MAX else text[:TITLE_MAX - 1].rstrip() + "…"


class AuthStore:
    """Thread-safe SQLite store (one connection guarded by a lock, like the feedback store)."""

    def __init__(self, path: Path, session_ttl_hours: float = 24 * 7) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _user(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        return {"id": row["id"], "username": row["username"], "created_at": row["created_at"]} if row else None

    # ------------------------------------------------------------------ users
    def create_user(self, username: str, password: str) -> Dict[str, Any]:
        username, password = validate_username(username), validate_password(password)
        password_hash = hash_password(password)  # slow: outside the lock
        with self._lock:
            try:
                cursor = self._conn.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, password_hash, _iso(_now())),
                )
            except sqlite3.IntegrityError:
                raise UsernameTaken("This username is already taken.") from None
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return self._user(row)

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """The user when the password is right, else None (same work for unknown usernames)."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE username = ?", ((username or "").strip(),)).fetchone()
        ok = verify_password(password or "", row["password_hash"] if row else DUMMY_HASH)
        return self._user(row) if row and ok else None

    # --------------------------------------------------------------- sessions
    def create_session(self, user_id: int) -> Dict[str, Any]:
        token, now = new_token(), _now()
        expires = now + self.session_ttl
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (_iso(now),))
            self._conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token_digest(token), user_id, _iso(now), _iso(expires)),
            )
            self._conn.commit()
        return {"token": token, "expires_at": _iso(expires)}

    def user_for_token(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id "
                "WHERE sessions.token_hash = ? AND sessions.expires_at > ?",
                (token_digest(token), _iso(_now())),
            ).fetchone()
        return self._user(row)

    def delete_session(self, token: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_digest(token),))
            self._conn.commit()

    # ---------------------------------------------------------- conversations
    def claim_conversation(self, thread_id: str, user_id: int) -> bool:
        """Record a new conversation for this user. False when it already belongs to someone."""
        now = _iso(_now())
        with self._lock:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO conversations (thread_id, user_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (thread_id, user_id, NEW_CONVERSATION_TITLE, now, now),
            )
            self._conn.commit()
        return cursor.rowcount == 1

    def conversation_owner(self, thread_id: str) -> Optional[int]:
        with self._lock:
            row = self._conn.execute("SELECT user_id FROM conversations WHERE thread_id = ?", (thread_id,)).fetchone()
        return row["user_id"] if row else None

    def conversation(self, thread_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT thread_id, title, created_at, updated_at FROM conversations WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return dict(row) if row else None

    def touch_conversation(self, thread_id: str, title: Optional[str] = None) -> None:
        """Mark the conversation as just used; set the title while it is still the default one."""
        with self._lock:
            self._conn.execute(
                "UPDATE conversations SET updated_at = ?, "
                "title = CASE WHEN title = ? AND ? IS NOT NULL THEN ? ELSE title END WHERE thread_id = ?",
                (_iso(_now()), NEW_CONVERSATION_TITLE, title, title, thread_id),
            )
            self._conn.commit()

    def rename_conversation(self, thread_id: str, title: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE conversations SET title = ? WHERE thread_id = ?", (make_title(title), thread_id))
            self._conn.commit()

    def delete_conversation(self, thread_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
            self._conn.commit()

    def list_conversations(self, user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Newest (most recently used) first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT thread_id, title, created_at, updated_at FROM conversations WHERE user_id = ? "
                "ORDER BY updated_at DESC, rowid DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
