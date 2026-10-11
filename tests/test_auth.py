"""
User accounts: password hashing, tokens, login rate limit, the account store, the /auth
endpoints, conversation ownership on every conversation endpoint, and conversations that
survive an API restart (SQLite checkpointer).
"""
import pytest
from fastapi.testclient import TestClient

from agents.graph import Assistant
from api.main import create_app
from auth import AuthError, AuthStore, LoginLimiter, UsernameTaken, open_checkpointer
from auth.security import hash_password, token_digest, validate_password, validate_username, verify_password
from auth.store import NEW_CONVERSATION_TITLE, make_title
from auth_utils import TEST_PASSWORD, SignedInClient, auth_header, sign_up
from test_agents import FakeLLM, route, text, tool_call


@pytest.fixture
def store(tmp_path):
    s = AuthStore(tmp_path / "users.db")
    yield s
    s.close()


def general_llm(*replies):
    return FakeLLM(routes=[route("general")] * len(replies), final_replies=list(replies))


def make_app(llm=None, assistant=None, auth_store=None):
    return create_app(assistant=assistant or Assistant(llm=llm or FakeLLM()), load_search_index=False,
                      auth_store=auth_store)


# ---------------------------------------------------------------- security
def test_password_hash_and_verify():
    stored = hash_password("correct horse battery")
    assert stored.startswith("scrypt$") and "correct horse" not in stored
    assert verify_password("correct horse battery", stored)
    assert not verify_password("correct horse batterY", stored)
    assert hash_password("correct horse battery") != stored  # random salt
    assert not verify_password("x", "not-a-hash") and not verify_password("x", "bcrypt$1$2$3$aa$bb")


def test_token_digest_is_stable_and_not_the_token():
    assert token_digest("abc") == token_digest("abc") != "abc" and len(token_digest("abc")) == 64


@pytest.mark.parametrize("username", ["ab", "a" * 33, "has space", "bad/char", ""])
def test_invalid_usernames(username):
    with pytest.raises(AuthError):
        validate_username(username)


@pytest.mark.parametrize("password", ["short", "x" * 129, " leading-space", "trailing-space "])
def test_invalid_passwords(password):
    with pytest.raises(AuthError):
        validate_password(password)


def test_login_limiter_locks_and_unlocks():
    now = [0.0]
    limiter = LoginLimiter(max_failures=3, window=60, clock=lambda: now[0])
    for _ in range(3):
        assert limiter.retry_after("Alice") is None
        limiter.failed("alice")
    assert limiter.retry_after("ALICE") == 61  # case-insensitive, like usernames
    assert limiter.retry_after("bob") is None
    now[0] = 61
    assert limiter.retry_after("alice") is None  # failures left the window
    limiter.failed("alice")
    limiter.succeeded("alice")
    assert limiter._failures.get("alice") is None


# ---------------------------------------------------------------- store
def test_store_users_and_authentication(store):
    user = store.create_user("Analyst1", TEST_PASSWORD)
    assert user["username"] == "Analyst1" and user["id"]
    with pytest.raises(UsernameTaken):
        store.create_user("analyst1", "another-password")  # usernames are case-insensitive
    assert store.authenticate("ANALYST1", TEST_PASSWORD)["id"] == user["id"]
    assert store.authenticate("analyst1", "wrong-password") is None
    assert store.authenticate("nobody", TEST_PASSWORD) is None


def test_store_sessions_expire_and_revoke(store, tmp_path):
    user = store.create_user("analyst1", TEST_PASSWORD)
    session = store.create_session(user["id"])
    assert store.user_for_token(session["token"])["username"] == "analyst1"
    raw = store._conn.execute("SELECT token_hash FROM sessions").fetchone()[0]
    assert raw != session["token"]  # only the digest is stored
    store.delete_session(session["token"])
    assert store.user_for_token(session["token"]) is None
    assert store.user_for_token("") is None

    expired = AuthStore(tmp_path / "users.db", session_ttl_hours=0)
    token = expired.create_session(user["id"])["token"]
    assert expired.user_for_token(token) is None
    expired.close()


def test_store_conversations(store):
    alice, bob = store.create_user("alice", TEST_PASSWORD), store.create_user("bob", TEST_PASSWORD)
    assert store.claim_conversation("t1", alice["id"])
    assert not store.claim_conversation("t1", bob["id"])  # already owned
    assert store.conversation_owner("t1") == alice["id"] and store.conversation_owner("t9") is None
    assert store.conversation("t1")["title"] == NEW_CONVERSATION_TITLE

    store.touch_conversation("t1", "first question")
    store.touch_conversation("t1", "second question")  # the title is only set once
    assert store.conversation("t1")["title"] == "first question"
    store.claim_conversation("t2", alice["id"])
    store.touch_conversation("t1")
    assert [c["thread_id"] for c in store.list_conversations(alice["id"])] == ["t1", "t2"]
    assert store.list_conversations(bob["id"]) == []

    store.rename_conversation("t2", "  My   SSH case ")
    assert store.conversation("t2")["title"] == "My SSH case"
    store.delete_conversation("t2")
    assert store.conversation("t2") is None


def test_make_title():
    assert make_title("  hello\n world ") == "hello world"
    assert make_title("") == NEW_CONVERSATION_TITLE
    long = make_title("x" * 200)
    assert len(long) == 80 and long.endswith("…")


# ---------------------------------------------------------------- /auth
def test_signup_login_me_logout():
    with TestClient(make_app()) as c:
        session = sign_up(c, "Analyst_7")
        assert session["user"]["username"] == "Analyst_7" and session["token"] and session["expires_at"]
        assert c.get("/auth/me", headers=auth_header(session)).json()["username"] == "Analyst_7"

        login = c.post("/auth/login", json={"username": "analyst_7", "password": TEST_PASSWORD})
        assert login.status_code == 200 and login.json()["token"] != session["token"]

        assert c.post("/auth/logout", headers=auth_header(session)).status_code == 204
        assert c.get("/auth/me", headers=auth_header(session)).status_code == 401
        assert c.get("/auth/me", headers=auth_header(login.json())).status_code == 200  # other session still valid
        assert c.post("/auth/logout").status_code == 204


def test_signup_errors():
    with TestClient(make_app()) as c:
        sign_up(c, "taken_name")
        assert c.post("/auth/signup", json={"username": "TAKEN_NAME", "password": TEST_PASSWORD}).status_code == 409
        r = c.post("/auth/signup", json={"username": "ok_name", "password": "short"})
        assert r.status_code == 422 and "Password" in r.json()["detail"]
        assert c.post("/auth/signup", json={"username": "a b", "password": TEST_PASSWORD}).status_code == 422


def test_login_failures_are_generic_and_rate_limited():
    with TestClient(make_app()) as c:
        sign_up(c, "victim")
        wrong = c.post("/auth/login", json={"username": "victim", "password": "wrong-password"})
        unknown = c.post("/auth/login", json={"username": "nobody_here", "password": "wrong-password"})
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json()["detail"] == unknown.json()["detail"]  # does not reveal which usernames exist
        for _ in range(4):
            c.post("/auth/login", json={"username": "victim", "password": "wrong-password"})
        locked = c.post("/auth/login", json={"username": "victim", "password": TEST_PASSWORD})
        assert locked.status_code == 429 and int(locked.headers["Retry-After"]) > 0


def test_protected_endpoints_need_a_token():
    with TestClient(make_app()) as c:
        for method, path, body in [("post", "/chat", {"message": "hi"}), ("post", "/chat/stream", {"message": "hi"}),
                                   ("get", "/chat/conversations", None), ("get", "/chat/t1/history", None),
                                   ("delete", "/chat/t1", None), ("patch", "/chat/t1", {"title": "x"}),
                                   ("post", "/feedback", {"thread_id": "t1", "rating": "up"}),
                                   ("post", "/judge", {"thread_id": "t1"}), ("get", "/auth/me", None)]:
            r = c.request(method.upper(), path, json=body)
            assert r.status_code == 401, (method, path, r.status_code)
            assert r.headers.get("WWW-Authenticate") == "Bearer"
        assert c.get("/chat/conversations", headers={"Authorization": "Bearer not-a-token"}).status_code == 401
        # Data-only endpoints stay open.
        assert c.get("/health").status_code == 200
        assert c.post("/ml/classify", json={"incident_id": "INC-000035"}).status_code == 200
        assert "/auth/login" in c.get("/openapi.json").json()["paths"]


# ---------------------------------------------------------------- conversations
def test_conversations_are_listed_titled_renamed_and_deleted():
    with SignedInClient(make_app(general_llm("hi there", "still here", "new one"))) as c:
        first = c.post("/chat", json={"message": "hello, what can you do?"}).json()["thread_id"]
        c.post("/chat", json={"message": "and more?", "thread_id": first})
        second = c.post("/chat/stream", json={"message": "second chat"}).headers["X-Thread-ID"]

        listed = c.get("/chat/conversations").json()
        assert [x["thread_id"] for x in listed] == [second, first]
        assert listed[1]["title"] == "hello, what can you do?" and listed[0]["title"] == "second chat"

        renamed = c.patch(f"/chat/{first}", json={"title": "My first case"})
        assert renamed.status_code == 200 and renamed.json()["title"] == "My first case"
        assert c.patch(f"/chat/{first}", json={"title": "   "}).status_code == 422

        assert c.delete(f"/chat/{first}").status_code == 204
        assert [x["thread_id"] for x in c.get("/chat/conversations").json()] == [second]
        assert c.get(f"/chat/{first}/history").status_code == 404


def test_title_never_contains_a_redacted_secret():
    with SignedInClient(make_app(general_llm("ok"))) as c:
        c.post("/chat", json={"message": "my password is Winter2024!x, is that safe?"})
        title = c.get("/chat/conversations").json()[0]["title"]
    assert "Winter2024" not in title and "[REDACTED_" in title


def test_users_cannot_touch_each_others_conversations():
    llm = FakeLLM(
        routes=[route("classify")],
        agent_replies=[tool_call("classify_attack_type", {"incident_id": "INC-004393"}),
                       text("INC-004393 is Infiltration.")],
    )
    app = make_app(llm)
    with SignedInClient(app) as alice:
        chat = alice.post("/chat", json={"message": "What attack is INC-004393?"}).json()
        thread = chat["thread_id"]
        bob = auth_header(sign_up(alice))
        not_found = [
            alice.get(f"/chat/{thread}/history", headers=bob),
            alice.delete(f"/chat/{thread}", headers=bob),
            alice.patch(f"/chat/{thread}", json={"title": "mine now"}, headers=bob),
            alice.post("/chat", json={"message": "and its risk?", "thread_id": thread}, headers=bob),
            alice.post("/chat/stream", json={"message": "and its risk?", "thread_id": thread}, headers=bob),
            alice.post("/feedback", json={"thread_id": thread, "rating": "down"}, headers=bob),
            alice.post("/judge", json={"thread_id": thread}, headers=bob),
        ]
        assert [r.status_code for r in not_found] == [404] * len(not_found)
        assert alice.get("/chat/conversations", headers=bob).json() == []

        # Alice still has everything, and her rating records her name.
        assert alice.get(f"/chat/{thread}/history").status_code == 200
        row = alice.post("/feedback", json={"thread_id": thread, "turn_id": chat["turn_id"], "rating": "up"}).json()
        assert row["username"] == alice.session["user"]["username"]
        assert alice.get("/chat/conversations").json()[0]["title"] == "What attack is INC-004393?"


def test_client_chosen_thread_id_is_claimed_once():
    with SignedInClient(make_app(general_llm("one", "two"))) as alice:
        assert alice.post("/chat", json={"message": "hi", "thread_id": "my-own-id"}).json()["thread_id"] == "my-own-id"
        bob = auth_header(sign_up(alice))
        assert alice.post("/chat", json={"message": "hi", "thread_id": "my-own-id"}, headers=bob).status_code == 404


def test_conversations_survive_an_api_restart(tmp_path):
    users = AuthStore(tmp_path / "users.db")

    def assistant(*replies):
        return Assistant(llm=general_llm(*replies), checkpointer=open_checkpointer(tmp_path / "chats.db"))

    with TestClient(make_app(assistant=assistant("first answer"), auth_store=users)) as c:
        session = sign_up(c, "persistent")
        thread = c.post("/chat", json={"message": "remember me"}, headers=auth_header(session)).json()["thread_id"]

    # New app, new assistant and checkpointer on the same files: as after a restart.
    with TestClient(make_app(assistant=assistant("second answer"), auth_store=users)) as c:
        login = c.post("/auth/login", json={"username": "persistent", "password": TEST_PASSWORD}).json()
        headers = auth_header(login)
        assert c.get("/chat/conversations", headers=headers).json()[0]["thread_id"] == thread
        history = c.get(f"/chat/{thread}/history", headers=headers).json()["messages"]
        assert [m["content"] for m in history] == ["remember me", "first answer"]
        c.post("/chat", json={"message": "still there?", "thread_id": thread}, headers=headers)
        assert len(c.get(f"/chat/{thread}/history", headers=headers).json()["messages"]) == 4
        assert c.delete(f"/chat/{thread}", headers=headers).status_code == 204
        assert c.get(f"/chat/{thread}/history", headers=headers).status_code == 404
    users.close()


def test_sqlite_checkpointer_keeps_tool_evidence_and_turn_history(tmp_path):
    """Real tool results (numpy values, nested dicts) round-trip through the SQLite store,
    and earlier turns can still be found for feedback."""
    llm = FakeLLM(
        routes=[route("classify"), route("general")],
        agent_replies=[tool_call("classify_attack_type", {"incident_id": "INC-004393"}),
                       text("INC-004393 is Infiltration.")],
        final_replies=["You're welcome."],
    )
    bot = Assistant(llm=llm, checkpointer=open_checkpointer(tmp_path / "chats.db"))
    first = bot.ask("What attack is INC-004393?", "t1")
    bot.ask("thanks", "t1")
    earlier = bot.turn_state("t1", first["turn_id"])
    assert earlier["evidence"][0]["result"]["attack_type"]["prediction"] == first["evidence"][0]["result"]["attack_type"]["prediction"]
    assert earlier["final_answer"] == "INC-004393 is Infiltration."
