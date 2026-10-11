"""
Password hashing, login tokens, input rules and the login rate limit.

    hash_password / verify_password   scrypt (standard library) with a random salt per user
    new_token / token_digest          random bearer token; only its SHA-256 digest is stored
    LoginLimiter                      locks a username after repeated failed logins
"""
import hashlib
import hmac
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_LEN = 2 ** 14, 8, 1, 32
USERNAME_PATTERN = r"^[A-Za-z0-9_.\-]{3,32}$"
PASSWORD_MIN, PASSWORD_MAX = 8, 128


class AuthError(ValueError):
    """Invalid sign-up or login input."""


def validate_username(username: str) -> str:
    username = (username or "").strip()
    if not re.fullmatch(USERNAME_PATTERN, username):
        raise AuthError("Username must be 3-32 characters: letters, digits, '.', '_' or '-'.")
    return username


def validate_password(password: str) -> str:
    if not isinstance(password, str) or not PASSWORD_MIN <= len(password) <= PASSWORD_MAX:
        raise AuthError(f"Password must be {PASSWORD_MIN}-{PASSWORD_MAX} characters.")
    if password.strip() != password or not password.strip():
        raise AuthError("Password must not start or end with spaces.")
    return password


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int, length: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=length, maxmem=64 * 1024 * 1024)


def hash_password(password: str) -> str:
    """'scrypt$n$r$p$salt$hash' (hex), so the cost can be raised later without breaking old hashes."""
    salt = secrets.token_bytes(16)
    digest = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_LEN)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = bytes.fromhex(digest)
        actual = _scrypt(password, bytes.fromhex(salt), int(n), int(r), int(p), len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# Checked when the username does not exist, so a failed login takes the same time either way.
DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class LoginLimiter:
    """At most `max_failures` failed logins per username in `window` seconds, then locked
    until the oldest failure leaves the window. A successful login clears the count."""

    def __init__(self, max_failures: int = 5, window: float = 300.0, clock=time.monotonic) -> None:
        self.max_failures, self.window, self._clock = max_failures, window, clock
        self._failures: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _recent(self, key: str) -> Deque[float]:
        failures, now = self._failures[key], self._clock()
        while failures and now - failures[0] >= self.window:
            failures.popleft()
        return failures

    def retry_after(self, username: str) -> Optional[int]:
        """Seconds until the next attempt is allowed, or None when not locked."""
        key = username.strip().lower()
        with self._lock:
            failures = self._recent(key)
            if len(failures) < self.max_failures:
                return None
            return max(1, int(self.window - (self._clock() - failures[0])) + 1)

    def failed(self, username: str) -> None:
        key = username.strip().lower()
        with self._lock:
            self._recent(key).append(self._clock())

    def succeeded(self, username: str) -> None:
        with self._lock:
            self._failures.pop(username.strip().lower(), None)
