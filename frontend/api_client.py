"""
HTTP client for the FastAPI backend. The front end talks to the assistant only through
this module: no agent, ML or retrieval code runs in the Streamlit process.

    client = ApiClient()                      # API_BASE_URL env var, default http://localhost:8000
    client.health()
    session = client.login("analyst1", "...")
    client = client.with_token(session["token"])   # signed in: chat, conversations, feedback
    for kind, data in client.chat_stream("Investigate INC-000035"):
        ...                                   # ("trace", event) ... ("result", response)

Every failed call raises ApiError with the HTTP status and a message fit for the UI.
"""
import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30.0
CHAT_TIMEOUT = 600.0  # an investigation runs several agents and LLM calls

STATUS_MESSAGES = {
    401: "Not signed in",
    404: "Not found",
    409: "Already exists",
    413: "The file is too large",
    422: "Invalid input",
    429: "Too many attempts",
    502: "The language model provider returned an error",
    503: "The service is not available",
}


class ApiError(Exception):
    """A failed API call. `status` is None when the API could not be reached."""

    def __init__(self, status: Optional[int], detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail

    @property
    def message(self) -> str:
        if self.status is None:
            return f"Cannot reach the API: {self.detail}"
        title = STATUS_MESSAGES.get(self.status, f"API error {self.status}")
        if self.detail and self.detail != title:
            return f"{title}: {self.detail}"
        return title


def base_url() -> str:
    return (os.getenv("API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def error_detail(response: httpx.Response) -> str:
    """The API's error message: FastAPI `detail` (text or validation list) or the raw body."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return response.text[:500] or response.reason_phrase
    if isinstance(detail, list):  # pydantic validation errors
        parts = []
        for item in detail:
            where = ".".join(str(p) for p in item.get("loc", [])[1:]) or "input"
            parts.append(f"{where}: {item.get('msg', '')}")
        return "; ".join(parts)
    return str(detail) if detail else response.reason_phrase


def parse_sse(lines: Iterable[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Turn Server-Sent Event lines into (event, data) pairs; data is parsed JSON."""
    event, data = "message", []
    for line in lines:
        line = line.rstrip("\r")
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith(":"):
            continue  # comment / keep-alive
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event, json.loads("\n".join(data))


class ApiClient:
    def __init__(self, url: Optional[str] = None, client: Optional[httpx.Client] = None,
                 timeout: float = DEFAULT_TIMEOUT, token: Optional[str] = None):
        self.url = (url or base_url()).rstrip("/")
        self._client = client or httpx.Client(base_url=self.url, timeout=timeout)
        self.token = token

    def with_token(self, token: Optional[str]) -> "ApiClient":
        """A client for one signed-in user that shares this client's connection pool."""
        return ApiClient(self.url, client=self._client, token=token)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    # ------------------------------------------------------------------ core
    def _request(self, method: str, path: str, timeout: Optional[float] = None, **kwargs) -> httpx.Response:
        if timeout is not None:
            kwargs["timeout"] = timeout
        kwargs["headers"] = {**self._headers(), **kwargs.get("headers", {})}
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException:
            raise ApiError(None, "the request timed out")
        except httpx.TransportError as exc:
            raise ApiError(None, f"{exc.__class__.__name__} at {self.url}")
        if response.status_code >= 400:
            raise ApiError(response.status_code, error_detail(response))
        return response

    def _json(self, method: str, path: str, **kwargs) -> Any:
        response = self._request(method, path, **kwargs)
        return response.json() if response.content else None

    # ---------------------------------------------------------------- system
    def health(self) -> Dict[str, Any]:
        return self._json("GET", "/health", timeout=10)

    # -------------------------------------------------------------- accounts
    def signup(self, username: str, password: str) -> Dict[str, Any]:
        return self._json("POST", "/auth/signup", json={"username": username, "password": password})

    def login(self, username: str, password: str) -> Dict[str, Any]:
        return self._json("POST", "/auth/login", json={"username": username, "password": password})

    def logout(self) -> None:
        self._request("POST", "/auth/logout")

    def me(self) -> Dict[str, Any]:
        return self._json("GET", "/auth/me")

    # ------------------------------------------------------------------ chat
    @staticmethod
    def _chat_body(message: str, thread_id: Optional[str], judge: Optional[bool]) -> Dict[str, Any]:
        body: Dict[str, Any] = {"message": message}
        if thread_id:
            body["thread_id"] = thread_id
        if judge is not None:
            body["judge"] = judge
        return body

    def chat(self, message: str, thread_id: Optional[str] = None, judge: Optional[bool] = None) -> Dict[str, Any]:
        return self._json("POST", "/chat", json=self._chat_body(message, thread_id, judge), timeout=CHAT_TIMEOUT)

    def chat_stream(self, message: str, thread_id: Optional[str] = None,
                    judge: Optional[bool] = None) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Yield ("trace", event) for each agent step, then ("result", response).

        An `error` event from the server is raised as ApiError(502 or 500, detail).
        """
        body = self._chat_body(message, thread_id, judge)
        try:
            with self._client.stream("POST", "/chat/stream", json=body, headers=self._headers(),
                                     timeout=CHAT_TIMEOUT) as response:
                if response.status_code >= 400:
                    response.read()
                    raise ApiError(response.status_code, error_detail(response))
                for kind, data in parse_sse(response.iter_lines()):
                    if kind == "error":
                        detail = data.get("detail", "The assistant failed")
                        raise ApiError(502 if "provider" in detail else 500, detail)
                    yield kind, data
        except httpx.TimeoutException:
            raise ApiError(None, "the request timed out")
        except httpx.TransportError as exc:
            raise ApiError(None, f"{exc.__class__.__name__} at {self.url}")

    def history(self, thread_id: str) -> Dict[str, Any]:
        return self._json("GET", f"/chat/{thread_id}/history")

    def delete_thread(self, thread_id: str) -> None:
        self._request("DELETE", f"/chat/{thread_id}")

    def conversations(self) -> List[Dict[str, Any]]:
        return self._json("GET", "/chat/conversations")

    def rename_thread(self, thread_id: str, title: str) -> Dict[str, Any]:
        return self._json("PATCH", f"/chat/{thread_id}", json={"title": title})

    # -------------------------------------------------------------------- ML
    def explain(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        return self._json("POST", "/ml/explain", json=flow)

    def classify(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        return self._json("POST", "/ml/classify", json=flow)

    def risk(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        return self._json("POST", "/ml/risk", json=flow)

    def prioritize(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        return self._json("POST", "/ml/prioritize", json=flow)

    def prioritize_batch(self, incident_ids: List[str] = (), flows: List[Dict[str, Any]] = (),
                         top_n: int = 10) -> Dict[str, Any]:
        body = {"incident_ids": list(incident_ids), "flows": list(flows), "top_n": top_n}
        return self._json("POST", "/ml/prioritize/batch", json=body, timeout=120)

    def prioritize_csv(self, content: bytes, filename: str = "flows.csv", top_n: int = 50) -> Dict[str, Any]:
        files = {"file": (filename, content, "text/csv")}
        return self._json("POST", "/ml/prioritize/csv", params={"top_n": top_n}, files=files, timeout=120)

    # ---------------------------------------------------------------- search
    def search(self, query: str, attack_type: Optional[str] = None, dst_port: Optional[int] = None,
               top_k: int = 5) -> Dict[str, Any]:
        body: Dict[str, Any] = {"query": query, "top_k": top_k}
        if attack_type:
            body["attack_type"] = attack_type
        if dst_port is not None:
            body["dst_port"] = dst_port
        return self._json("POST", "/search", json=body, timeout=300)  # first search may wait for the index

    def incident(self, incident_id: str) -> Dict[str, Any]:
        return self._json("GET", f"/incidents/{incident_id}")

    def attack_types(self) -> List[Dict[str, Any]]:
        return self._json("GET", "/attack-types")["attack_types"]

    def attack_profile(self, name: str) -> Dict[str, Any]:
        return self._json("GET", f"/attack-types/{name}")

    # ----------------------------------------------------- judge and feedback
    def judge(self, thread_id: Optional[str] = None, question: Optional[str] = None,
              answer: Optional[str] = None, route: Optional[str] = None) -> Dict[str, Any]:
        body = {k: v for k, v in {"thread_id": thread_id, "question": question, "answer": answer,
                                  "route": route}.items() if v}
        return self._json("POST", "/judge", json=body, timeout=CHAT_TIMEOUT)

    def send_feedback(self, thread_id: str, rating: str, turn_id: Optional[str] = None,
                      comment: Optional[str] = None, corrected_label: Optional[str] = None) -> Dict[str, Any]:
        body = {k: v for k, v in {"thread_id": thread_id, "turn_id": turn_id, "rating": rating,
                                  "comment": comment, "corrected_label": corrected_label}.items() if v}
        return self._json("POST", "/feedback", json=body)

    def feedback_list(self, **filters) -> List[Dict[str, Any]]:
        params = {k: v for k, v in filters.items() if v not in (None, "", False)}
        return self._json("GET", "/feedback", params=params)

    def feedback_stats(self) -> Dict[str, Any]:
        return self._json("GET", "/feedback/stats")

    def feedback_export_csv(self) -> bytes:
        return self._request("GET", "/feedback/export.csv").content
