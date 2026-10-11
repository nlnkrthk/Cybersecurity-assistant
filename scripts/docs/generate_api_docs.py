#!/usr/bin/env python3
"""
Generate docs/API_Documentation.md by calling every endpoint of the real API.

    python scripts/docs/generate_api_docs.py          # then: python scripts/docs/build_docs.py api

The app runs in-process (FastAPI TestClient) with the LLM configured in .env, the real ML models
and search indexes, and throwaway account / conversation / feedback databases (data/ is never
touched). For each endpoint the document lists the method, path, authentication, request and
response fields (from the OpenAPI schema), and a sample request with the response it really
returned, trimmed for length.
"""
import json
import os
import sys
import tempfile
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "API_Documentation.md"

_tmp = Path(tempfile.mkdtemp(prefix="api_docs_"))
os.environ.update({"AUTH_DB_PATH": str(_tmp / "users.db"), "CHAT_DB_PATH": str(_tmp / "chats.db"),
                   "FEEDBACK_DB_PATH": str(_tmp / "feedback.db"), "TQDM_DISABLE": "1", "JUDGE_MODE": "off"})
os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402

TAG_TITLES = {"system": "System", "accounts": "Accounts", "assistant": "Assistant", "feedback": "Feedback",
              "ml": "Machine-learning models", "search": "Incident search"}
TAG_ORDER = ["system", "accounts", "assistant", "feedback", "ml", "search"]
FLOW = {"flow_duration": "2 s", "total_fwd_packets": 10, "total_bwd_packets": 8,
        "total_len_fwd": "1.2 KB", "total_len_bwd": "4 KB"}
DEMO_USER = {"username": "analyst_demo", "password": "demo-passphrase-2026"}


# ============================================================================
# Trimming and formatting
# ============================================================================
def trim(value: Any, depth: int = 0, list_items: int = 2, text: int = 260) -> Any:
    if isinstance(value, dict):
        if depth >= 5:
            return "{…}"
        return {k: ("<token>" if k == "token" else trim(v, depth + 1, list_items, text)) for k, v in value.items()}
    if isinstance(value, list):
        items = [trim(v, depth + 1, list_items, text) for v in value[:list_items]]
        if len(value) > list_items:
            items.append(f"… {len(value) - list_items} more")
        return items
    if isinstance(value, str) and len(value) > text:
        return value[:text].rstrip() + " …"
    return value


def code(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text}\n```"


def as_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def json_type(schema: Dict[str, Any], spec: Dict[str, Any]) -> str:
    schema = resolve(schema, spec)
    if "anyOf" in schema:
        kinds = [json_type(s, spec) for s in schema["anyOf"] if s.get("type") != "null"]
        return " or ".join(dict.fromkeys(kinds)) or "null"
    if "enum" in schema:
        return " \\| ".join(f"`{v}`" for v in schema["enum"])
    if "const" in schema:
        return f"`{schema['const']}`"
    kind = schema.get("type", "object")
    if kind == "array":
        return f"array of {json_type(schema.get('items', {}), spec)}"
    return kind


def resolve(schema: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    while "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        schema = spec["components"]["schemas"][name]
    return schema


def field_table(schema: Dict[str, Any], spec: Dict[str, Any]) -> str:
    schema = resolve(schema, spec)
    props = schema.get("properties") or {}
    if not props:
        return ""
    required = set(schema.get("required") or [])
    rows = ["| Field | Type | Required | Description |", "|---|---|---|---|"]
    for name, prop in props.items():
        info = resolve(prop, spec)
        limits = []
        for key, label in (("minLength", "min length"), ("maxLength", "max length"), ("minimum", "≥"),
                           ("maximum", "≤"), ("maxItems", "max items")):
            if key in info:
                limits.append(f"{label} {info[key]}")
        if "default" in info and info["default"] not in (None, [], {}):
            limits.append(f"default {json.dumps(info['default'])}")
        description = (info.get("description") or prop.get("description") or "").replace("\n", " ")
        if limits:
            description = (description + f" ({', '.join(limits)})").strip()
        rows.append(f"| `{name}` | {json_type(prop, spec)} | {'yes' if name in required else 'no'} | {description} |")
    return "\n".join(rows)


def response_fields(sample: Any) -> str:
    if not isinstance(sample, dict) or not sample:
        return ""
    rows = ["| Field | Type |", "|---|---|"]
    for key, value in sample.items():
        kind = ("object" if isinstance(value, dict) else "array" if isinstance(value, list) else
                "boolean" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else
                "null" if value is None else "string")
        rows.append(f"| `{key}` | {kind} |")
    return "\n".join(rows)


# ============================================================================
# Calling the API
# ============================================================================
class Recorder:
    def __init__(self, client: TestClient):
        self.client, self.calls = client, {}

    def call(self, method: str, path: str, template: Optional[str] = None, *, token: Optional[str] = None,
             json_body: Any = None, params: Optional[Dict[str, Any]] = None, files=None, note: str = "",
             list_items: int = 2, text: int = 260):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        started = time.perf_counter()
        response = self.client.request(method, path, json=json_body, params=params, files=files, headers=headers)
        elapsed = time.perf_counter() - started
        content_type = response.headers.get("content-type", "")
        if not response.content:
            body = ""
        elif "application/json" in content_type:
            body = response.json()
        else:
            body = response.text
        key = (method, template or path)
        self.calls[key] = {"path": path, "json": json_body, "params": params, "files": files, "token": token,
                           "status": response.status_code, "body": body, "content_type": content_type,
                           "note": note, "list_items": list_items, "text": text, "elapsed": elapsed}
        print(f"{method:6} {path:45} {response.status_code} ({elapsed:.1f}s)")
        return body


def run_calls(client: TestClient) -> Recorder:
    r = Recorder(client)
    for _ in range(90):  # the search indexes load in the background
        if client.get("/health").json().get("search_index") in ("ready", "error"):
            break
        time.sleep(1)
    r.call("GET", "/health", list_items=8)

    session = r.call("POST", "/auth/signup", json_body=DEMO_USER)
    token = session["token"]
    r.call("POST", "/auth/login", json_body=DEMO_USER)
    r.call("GET", "/auth/me", token=token)

    chat = r.call("POST", "/chat", token=token, json_body={"message": "What attack is incident INC-004393?"},
                  list_items=1, text=700)
    thread, turn = chat["thread_id"], chat["turn_id"]
    stream = r.call("POST", "/chat/stream", token=token,
                    json_body={"message": "And how risky is it?", "thread_id": thread})
    r.calls[("POST", "/chat/stream")]["body"] = trim_sse(stream)
    r.call("GET", "/chat/conversations", token=token)
    r.call("GET", f"/chat/{thread}/history", "/chat/{thread_id}/history", token=token, text=400)
    r.call("PATCH", f"/chat/{thread}", "/chat/{thread_id}", token=token, json_body={"title": "Infiltration case"})

    r.call("POST", "/judge", token=token, json_body={"thread_id": thread}, text=300)
    r.call("POST", "/feedback", token=token, json_body={
        "thread_id": thread, "turn_id": turn, "rating": "down", "comment": "Should be Bot",
        "corrected_label": "Bot"}, text=200)
    r.call("GET", "/feedback", params={"limit": 5}, list_items=1, text=160)
    r.call("GET", "/feedback/stats")
    r.call("GET", "/feedback/export.csv")

    r.call("POST", "/ml/classify", json_body={"incident_id": "INC-000035"})
    r.call("POST", "/ml/risk", json_body=FLOW)
    r.call("POST", "/ml/explain", json_body={"incident_id": "INC-000035"})
    r.call("POST", "/ml/prioritize", json_body={"incident_id": "INC-002545"})
    r.call("POST", "/ml/prioritize/batch",
           json_body={"incident_ids": ["INC-000001", "INC-000035", "INC-002545", "INC-000005"], "top_n": 3})
    csv = b"incident_id\nINC-000001\nINC-000035\nINC-002545\n"
    r.call("POST", "/ml/prioritize/csv", params={"top_n": 3}, files={"file": ("flows.csv", csv, "text/csv")})

    r.call("POST", "/search", json_body={"query": "SSH brute force with many short connections",
                                         "attack_type": "SSH-Patator", "top_k": 3}, text=220)
    r.call("GET", "/incidents/INC-000035", "/incidents/{incident_id}", list_items=30)
    r.call("GET", "/attack-types", list_items=3)
    r.call("GET", "/attack-types/DDoS", "/attack-types/{name}")

    r.call("DELETE", f"/chat/{thread}", "/chat/{thread_id}", token=token)
    r.call("POST", "/auth/logout", token=token)
    return r


def trim_sse(text: str, keep: int = 4) -> str:
    blocks = [b for b in text.strip().split("\n\n") if b.strip()]
    shown = []
    for block in blocks[:keep] + blocks[-1:]:
        lines = []
        for line in block.splitlines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                line = "data: " + json.dumps(trim(data, list_items=1, text=120), ensure_ascii=False)
                if len(line) > 600:
                    line = line[:600] + " …"
            lines.append(line)
        shown.append("\n".join(lines))
    middle = f"\n\n… {len(blocks) - keep - 1} more trace events …\n\n" if len(blocks) > keep + 1 else "\n\n"
    return "\n\n".join(shown[:-1]) + middle + shown[-1]


# ============================================================================
# Document
# ============================================================================
def sample_request(method: str, call: Dict[str, Any]) -> str:
    query = ""
    if call["params"]:
        query = "?" + "&".join(f"{k}={v}" for k, v in call["params"].items())
    lines = [f"{method} {call['path']}{query} HTTP/1.1", "Host: localhost:8000"]
    if call["token"]:
        lines.append("Authorization: Bearer <token>")
    if call["files"]:
        name, content, _ = call["files"]["file"]
        lines += ["Content-Type: multipart/form-data", "", f"file = {name}:", content.decode()]
    elif call["json"] is not None:
        lines += ["Content-Type: application/json", "", as_json(call["json"])]
    return code("\n".join(lines), "http")


def sample_response(call: Dict[str, Any]) -> str:
    head = f"HTTP/1.1 {call['status']}"
    body = call["body"]
    if isinstance(body, (dict, list)):
        body = as_json(trim(body, list_items=call["list_items"], text=call["text"]))
    elif not body:
        body = "(no body)"
    elif "text/csv" in call["content_type"]:
        lines = body.strip().splitlines()
        header = lines[0].split(",")
        body = ",".join(header[:6]) + f",… ({len(header)} columns)\n" + "\n".join(
            ",".join(row.split(",")[:6]) + ",…" for row in lines[1:3])
    return code(f"{head}\n\n{body}", "json" if call["content_type"].startswith("application/json") else "")


def endpoint_section(method: str, path: str, op: Dict[str, Any], spec: Dict[str, Any], call: Dict[str, Any]) -> str:
    out = [f"### {method} {path}", "", f"**{op.get('summary', '')}**", ""]
    description = (op.get("description") or "").strip()
    if description:
        out += [description, ""]
    protected = any("HTTPBearer" in s for s in (op.get("security") or [{}])[0]) if op.get("security") else False
    auth = "Bearer token required" if protected else "None"
    if path == "/judge":
        auth = "Bearer token required when grading a conversation (`thread_id`)"
    if path == "/auth/logout":
        auth = "Bearer token (the session to end)"
    out += ["| Property | Value |", "|---|---|", f"| Method | `{method}` |", f"| Path | `{path}` |", f"| Authentication | {auth} |"]
    params = op.get("parameters") or []
    if params:
        out.append("| Parameters | " + "; ".join(
            f"`{p['name']}` ({p['in']}{', required' if p.get('required') else ''})" for p in params) + " |")
    out.append("")

    out += ["#### Request format", ""]
    body = (op.get("requestBody") or {}).get("content", {})
    if "application/json" in body:
        schema = body["application/json"]["schema"]
        name = resolve(schema, spec).get("title", "")
        if name == "FlowRequest":
            out += ["A flow (see [Flow input](#flow-input)): any subset of the 24 metrics, or an `incident_id`.", ""]
        else:
            table = field_table(schema, spec)
            out += [table or "JSON body.", ""]
    elif "multipart/form-data" in body:
        out += ["`multipart/form-data` with a `file` field: a CSV of flows (≤ 5 MB, ≤ 1,000 rows). Columns are "
                "matched by metric name, alias or raw CIC-IDS2017 header; an `incident_id` column is also accepted.", ""]
    else:
        out += ["No request body.", ""]

    out += ["#### Response format", ""]
    responses = op.get("responses", {})
    ok = next((code_ for code_ in responses if code_.startswith("2")), "200")
    content = responses.get(ok, {}).get("content", {})
    media = ", ".join(content) or "no body"
    out += [f"`{ok}` · {media}", ""]
    schema = content.get("application/json", {}).get("schema", {})
    table = field_table(schema, spec) if schema.get("$ref") else ""
    if not table and isinstance(call["body"], dict):
        table = response_fields(call["body"])
    if table:
        out += [table, ""]
    if call["note"]:
        out += [call["note"], ""]
    out += ["#### Sample request", "", sample_request(method, call), "", "#### Sample response", "",
            sample_response(call), ""]
    return "\n".join(out)


def build_document(spec: Dict[str, Any], recorder: Recorder, llm_model: str) -> str:
    groups: Dict[str, List[str]] = {tag: [] for tag in TAG_ORDER}
    index_rows = ["| Method | Path | Purpose | Auth |", "|---|---|---|---|"]
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            method = method.upper()
            call = recorder.calls[(method, path)]
            tag = (op.get("tags") or ["system"])[0]
            groups[tag].append(endpoint_section(method, path, op, spec, call))
            protected = bool(op.get("security"))
            index_rows.append(f"| `{method}` | `{path}` | {op.get('summary', '')} | {'🔒' if protected else ''} |")

    flow_schema = spec["components"]["schemas"]["FlowRequest"]
    parts = [
        "---",
        'title: "AI-Powered Cybersecurity Incident Response Assistant"',
        'subtitle: "API Documentation"',
        'tagline: "REST and Server-Sent Events API of the FastAPI backend: accounts, assistant, feedback, '
        'machine-learning models and incident search"',
        'project: "Capstone Project 8 · CIC-IDS2017"',
        f'version: "{spec["info"]["version"]}"',
        f'date: "{date.today():%B %Y}"',
        "---",
        "",
        "# 1. Overview",
        "",
        "| Property | Value |",
        "|---|---|",
        "| Base URL | `http://localhost:8000` (start with `python scripts/api/run_api.py`) |",
        "| Format | JSON requests and responses; `/chat/stream` returns Server-Sent Events; one endpoint takes a CSV upload |",
        "| Interactive docs | Swagger UI at `/docs`, OpenAPI schema at `/openapi.json` |",
        "| Authentication | `Authorization: Bearer <token>` from `/auth/signup` or `/auth/login`; needed for the assistant, conversation and feedback-writing endpoints (🔒) |",
        "| Request IDs | Every response has an `X-Request-ID` header (taken from the request or generated) |",
        "| CORS | Only the Streamlit origin (`localhost:8501`) by default; `API_CORS_ORIGINS` changes it |",
        "",
        "All samples in this document are **real responses** of the API, recorded on "
        f"{date.today():%d %B %Y} with the assistant running `{llm_model}` and the project's trained models and "
        "search indexes. Long lists and texts are shortened (`… N more`, ` …`).",
        "",
        "## 1.1 Endpoints",
        "",
        "\n".join(index_rows),
        "",
        "🔒 = bearer token required.",
        "",
        "## 1.2 Status codes",
        "",
        "| Code | When |",
        "|---|---|",
        "| `200` / `201` / `204` | Success (`201` for sign-up and new feedback, `204` for logout and delete) |",
        "| `200` with `\"status\": \"needs_follow_up\"` | ML endpoints: key metrics (duration, packets, bytes) are missing; the response says which |",
        "| `200` with `\"route\": \"blocked\"` | Assistant: the input guard blocked the message; fixed reply |",
        "| `401` | Missing, expired or revoked token; wrong username or password |",
        "| `404` | Unknown incident, attack type or conversation, or another user's conversation |",
        "| `409` | Username already taken |",
        "| `413` | CSV larger than 5 MB |",
        "| `422` | Invalid input (validation details in `detail`) or impossible flow values |",
        "| `429` | Too many failed logins for a username (`Retry-After` header) |",
        "| `502` | The LLM provider returned an error (quota, billing, outage); `detail` has its message |",
        "| `503` | Assistant not configured (no LLM key), or the search indexes failed to load |",
        "| `500` | Unexpected error; the body has a `request_id` to find it in the server log |",
        "",
        "Errors have the shape `{\"detail\": \"...\"}` (validation errors: a list of `{loc, msg, type}`).",
        "",
        "## 1.3 Flow input",
        "",
        "The `/ml` endpoints and the assistant accept a network flow as **any subset of the 24 metrics** "
        "below, or an `incident_id` of a historical incident. Values can be numbers or strings with units "
        "(`\"2 s\"`, `\"1.5 KB\"`, `\"300 ms\"`). Missing metrics are derived where possible, otherwise filled "
        "with normal-traffic medians and reported in `input_quality`; if duration, packets or bytes are all "
        "missing, the response asks for them (`needs_follow_up`). Impossible values are rejected with `422`.",
        "",
        field_table(flow_schema, spec),
        "",
    ]
    number = 2
    for tag in TAG_ORDER:
        if not groups[tag]:
            continue
        parts += [f"# {number}. {TAG_TITLES[tag]}", ""] + groups[tag]
        number += 1
    return "\n".join(parts) + "\n"


def main() -> None:
    app = create_app(load_search_index=True)
    with TestClient(app) as client:
        state = app.state.services
        if state.assistant is None:
            sys.exit(f"The assistant is not available ({state.assistant_error}); configure the LLM in .env first.")
        recorder = run_calls(client)
        spec = app.openapi()
        failed = [f"{m} {p}: {c['status']}" for (m, p), c in recorder.calls.items() if c["status"] >= 400]
        if failed:
            sys.exit("Calls failed:\n" + "\n".join(failed))
        OUTPUT.write_text(build_document(spec, recorder, state.llm_model), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
