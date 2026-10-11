"""
Test helpers for the signed-in API: create a throwaway account and send its token.
"""
import uuid
from typing import Optional

from fastapi.testclient import TestClient

TEST_PASSWORD = "test-passphrase-123"


def sign_up(client: TestClient, username: Optional[str] = None, password: str = TEST_PASSWORD) -> dict:
    """Create an account (a unique name by default) and return the /auth/signup response."""
    username = username or f"user_{uuid.uuid4().hex[:10]}"
    response = client.post("/auth/signup", json={"username": username, "password": password})
    assert response.status_code == 201, response.text
    return response.json()


def auth_header(session: dict) -> dict:
    return {"Authorization": f"Bearer {session['token']}"}


class SignedInClient(TestClient):
    """A TestClient that signs up a fresh user on entering and sends its token with every request."""

    def __enter__(self):
        super().__enter__()
        self.session = sign_up(self)
        self.headers.update(auth_header(self.session))
        return self
