"""
Account endpoints: sign up, log in, log out, who am I.

Login returns an opaque bearer token (only its SHA-256 digest is stored, so it can be
revoked by logging out and expires after SESSION_TTL_HOURS). Repeated failed logins for
a username are rate limited.
"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from api.deps import AppState, bearer, get_current_user, get_state
from api.schemas import Credentials, SessionResponse, UserResponse
from auth import AuthError, UsernameTaken

router = APIRouter(prefix="/auth", tags=["accounts"])

LOGIN_FAILED = "Wrong username or password."


@router.post("/signup", response_model=SessionResponse, status_code=201, summary="Create an account")
def signup(body: Credentials, state: AppState = Depends(get_state)) -> Dict[str, Any]:
    """Create an account and sign in. Username: 3-32 letters, digits, '.', '_' or '-'
    (not case-sensitive). Password: 8-128 characters."""
    try:
        user = state.auth_store.create_user(body.username, body.password)
    except UsernameTaken as exc:
        raise HTTPException(409, str(exc))
    except AuthError as exc:
        raise HTTPException(422, str(exc))
    return {**state.auth_store.create_session(user["id"]), "user": user}


@router.post("/login", response_model=SessionResponse, summary="Sign in")
def login(body: Credentials, state: AppState = Depends(get_state)) -> Dict[str, Any]:
    retry = state.login_limiter.retry_after(body.username)
    if retry:
        raise HTTPException(429, f"Too many failed attempts. Try again in {retry} seconds.",
                            headers={"Retry-After": str(retry)})
    user = state.auth_store.authenticate(body.username, body.password)
    if user is None:
        state.login_limiter.failed(body.username)
        raise HTTPException(401, LOGIN_FAILED)
    state.login_limiter.succeeded(body.username)
    return {**state.auth_store.create_session(user["id"]), "user": user}


@router.post("/logout", status_code=204, summary="Sign out")
def logout(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
           state: AppState = Depends(get_state)) -> None:
    """Revoke the token. Always succeeds (logging out twice is harmless)."""
    if credentials:
        state.auth_store.delete_session(credentials.credentials)


@router.get("/me", response_model=UserResponse, summary="Current user")
def me(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return user
