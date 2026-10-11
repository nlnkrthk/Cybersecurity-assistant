---
title: "AI-Powered Cybersecurity Incident Response Assistant"
subtitle: "API Documentation"
tagline: "REST and Server-Sent Events API of the FastAPI backend: accounts, assistant, feedback, machine-learning models and incident search"
project: "Capstone Project 8 · CIC-IDS2017"
version: "1.0.0"
date: "October 2026"
---

# 1. Overview

| Property | Value |
|---|---|
| Base URL | `http://localhost:8000` (start with `python scripts/api/run_api.py`) |
| Format | JSON requests and responses; `/chat/stream` returns Server-Sent Events; one endpoint takes a CSV upload |
| Interactive docs | Swagger UI at `/docs`, OpenAPI schema at `/openapi.json` |
| Authentication | `Authorization: Bearer <token>` from `/auth/signup` or `/auth/login`; needed for the assistant, conversation and feedback-writing endpoints (🔒) |
| Request IDs | Every response has an `X-Request-ID` header (taken from the request or generated) |
| CORS | Only the Streamlit origin (`localhost:8501`) by default; `API_CORS_ORIGINS` changes it |

All samples in this document are **real responses** of the API, recorded on 11 October 2026 with the assistant running `nvidia:openai/gpt-oss-20b` and the project's trained models and search indexes. Long lists and texts are shortened (`… N more`, ` …`).

## 1.1 Endpoints

| Method | Path | Purpose | Auth |
|---|---|---|---|
| `GET` | `/health` | Service status |  |
| `POST` | `/auth/signup` | Create an account |  |
| `POST` | `/auth/login` | Sign in |  |
| `POST` | `/auth/logout` | Sign out | 🔒 |
| `GET` | `/auth/me` | Current user | 🔒 |
| `POST` | `/chat` | Ask the assistant | 🔒 |
| `POST` | `/chat/stream` | Ask the assistant (live trace) | 🔒 |
| `GET` | `/chat/conversations` | Your conversations | 🔒 |
| `GET` | `/chat/{thread_id}/history` | Conversation history | 🔒 |
| `PATCH` | `/chat/{thread_id}` | Rename a conversation | 🔒 |
| `DELETE` | `/chat/{thread_id}` | Delete a conversation | 🔒 |
| `POST` | `/judge` | Grade an answer (LLM-as-Judge) | 🔒 |
| `POST` | `/feedback` | Rate an answer | 🔒 |
| `GET` | `/feedback` | List feedback |  |
| `GET` | `/feedback/stats` | Feedback statistics |  |
| `GET` | `/feedback/export.csv` | Export label corrections |  |
| `POST` | `/ml/classify` | Attack type of a flow |  |
| `POST` | `/ml/risk` | Intrusion risk and anomaly of a flow |  |
| `POST` | `/ml/explain` | Full analysis with SHAP reasons |  |
| `POST` | `/ml/prioritize` | Priority of one flow |  |
| `POST` | `/ml/prioritize/batch` | Rank many flows / incidents |  |
| `POST` | `/ml/prioritize/csv` | Rank flows from a CSV file |  |
| `POST` | `/search` | Find similar historical incidents |  |
| `GET` | `/incidents/{incident_id}` | Look up a historical incident |  |
| `GET` | `/attack-types` | List attack types |  |
| `GET` | `/attack-types/{name}` | Profile of an attack type |  |

🔒 = bearer token required.

## 1.2 Status codes

| Code | When |
|---|---|
| `200` / `201` / `204` | Success (`201` for sign-up and new feedback, `204` for logout and delete) |
| `200` with `"status": "needs_follow_up"` | ML endpoints: key metrics (duration, packets, bytes) are missing; the response says which |
| `200` with `"route": "blocked"` | Assistant: the input guard blocked the message; fixed reply |
| `401` | Missing, expired or revoked token; wrong username or password |
| `404` | Unknown incident, attack type or conversation, or another user's conversation |
| `409` | Username already taken |
| `413` | CSV larger than 5 MB |
| `422` | Invalid input (validation details in `detail`) or impossible flow values |
| `429` | Too many failed logins for a username (`Retry-After` header) |
| `502` | The LLM provider returned an error (quota, billing, outage); `detail` has its message |
| `503` | Assistant not configured (no LLM key), or the search indexes failed to load |
| `500` | Unexpected error; the body has a `request_id` to find it in the server log |

Errors have the shape `{"detail": "..."}` (validation errors: a list of `{loc, msg, type}`).

## 1.3 Flow input

The `/ml` endpoints and the assistant accept a network flow as **any subset of the 24 metrics** below, or an `incident_id` of a historical incident. Values can be numbers or strings with units (`"2 s"`, `"1.5 KB"`, `"300 ms"`). Missing metrics are derived where possible, otherwise filled with normal-traffic medians and reported in `input_quality`; if duration, packets or bytes are all missing, the response asks for them (`needs_follow_up`). Impossible values are rejected with `422`.

| Field | Type | Required | Description |
|---|---|---|---|
| `flow_duration` | number or string | no | Total flow duration in microseconds (e.g. '2 s' or 2000000). |
| `total_fwd_packets` | number or string | no | Number of packets sent from source to destination. |
| `total_bwd_packets` | number or string | no | Number of packets sent from destination back to source. |
| `total_len_fwd` | number or string | no | Total bytes sent forward (source to destination). |
| `total_len_bwd` | number or string | no | Total bytes sent backward (destination to source). |
| `flow_packets_s` | number or string | no | Packets per second over the whole flow. |
| `fwd_packets_s` | number or string | no | Forward packets per second. |
| `bwd_packets_s` | number or string | no | Backward packets per second. |
| `fwd_packet_length_mean` | number or string | no | Mean size of forward packets in bytes. |
| `bwd_packet_length_mean` | number or string | no | Mean size of backward packets in bytes. |
| `min_packet_length` | number or string | no | Smallest packet size in bytes. |
| `max_packet_length` | number or string | no | Largest packet size in bytes. |
| `packet_length_mean` | number or string | no | Mean packet size in bytes (both directions). |
| `packet_length_std` | number or string | no | Standard deviation of packet size in bytes. |
| `syn_flag_count` | number or string | no | 1 if the TCP SYN flag was seen, else 0. |
| `ack_flag_count` | number or string | no | 1 if the TCP ACK flag was seen, else 0. |
| `rst_flag_count` | number or string | no | 1 if the TCP RST flag was seen, else 0. |
| `psh_flag_count` | number or string | no | 1 if the TCP PSH flag was seen, else 0. |
| `fin_flag_count` | number or string | no | 1 if the TCP FIN flag was seen, else 0. |
| `down_up_ratio` | number or string | no | Backward packets divided by forward packets (rounded down). |
| `init_win_bytes_forward` | number or string | no | Initial TCP window size in the forward direction, bytes (-1 if not set). |
| `init_win_bytes_backward` | number or string | no | Initial TCP window size in the backward direction, bytes (-1 if not set). |
| `active_mean` | number or string | no | Mean time the flow was active before going idle, microseconds. |
| `idle_mean` | number or string | no | Mean time the flow was idle before becoming active, microseconds. |
| `incident_id` | string | no | ID of a historical incident, e.g. 'INC-000123'. When given, the incident's stored metrics are used and any metric fields are ignored. |

# 2. System

### GET /health

**Service status**

Whether the ML models, the assistant (LLM) and the search indexes are ready.

`status` is "ok" when the models and the assistant are ready and the indexes are
not in error; otherwise "degraded" (the direct ML endpoints may still work).

| Property | Value |
|---|---|
| Method | `GET` |
| Path | `/health` |
| Authentication | None |

#### Request format

No request body.

#### Response format

`200` · application/json

| Field | Type | Required | Description |
|---|---|---|---|
| `status` | `ok` \| `degraded` | yes |  |
| `ml_models_loaded` | boolean | yes |  |
| `attack_types` | integer | yes |  |
| `assistant_ready` | boolean | yes |  |
| `assistant_error` | string | no |  |
| `llm_model` | string | no |  |
| `classifier_version` | string | no | "original" or a retrained version. |
| `search_index` | `not_loaded` \| `loading` \| `ready` \| `error` | yes |  |
| `search_index_error` | string | no |  |
| `versions` | object | yes |  |

#### Sample request

```http
GET /health HTTP/1.1
Host: localhost:8000
```

#### Sample response

```json
HTTP/1.1 200

{
  "status": "ok",
  "ml_models_loaded": true,
  "attack_types": 15,
  "assistant_ready": true,
  "assistant_error": null,
  "llm_model": "nvidia:openai/gpt-oss-20b",
  "classifier_version": "original",
  "search_index": "ready",
  "search_index_error": null,
  "versions": {
    "fastapi": "0.143.0",
    "langgraph": "1.2.14",
    "langchain-anthropic": "1.7.5",
    "scikit-learn": "1.7.2",
    "shap": "0.48.0"
  }
}
```

# 3. Accounts

### POST /auth/signup

**Create an account**

Create an account and sign in. Username: 3-32 letters, digits, '.', '_' or '-'
(not case-sensitive). Password: 8-128 characters.

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/auth/signup` |
| Authentication | None |

#### Request format

| Field | Type | Required | Description |
|---|---|---|---|
| `username` | string | yes | (min length 1, max length 64) |
| `password` | string | yes | (min length 1, max length 256) |

#### Response format

`201` · application/json

| Field | Type | Required | Description |
|---|---|---|---|
| `token` | string | yes | Send as 'Authorization: Bearer <token>'. |
| `expires_at` | string | yes |  |
| `user` | object | yes |  |

#### Sample request

```http
POST /auth/signup HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "username": "analyst_demo",
  "password": "demo-passphrase-2026"
}
```

#### Sample response

```json
HTTP/1.1 201

{
  "token": "<token>",
  "expires_at": "2026-10-18T09:29:00.440264+00:00",
  "user": {
    "id": 1,
    "username": "analyst_demo",
    "created_at": "2026-10-11T09:29:00.430735+00:00"
  }
}
```

### POST /auth/login

**Sign in**

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/auth/login` |
| Authentication | None |

#### Request format

| Field | Type | Required | Description |
|---|---|---|---|
| `username` | string | yes | (min length 1, max length 64) |
| `password` | string | yes | (min length 1, max length 256) |

#### Response format

`200` · application/json

| Field | Type | Required | Description |
|---|---|---|---|
| `token` | string | yes | Send as 'Authorization: Bearer <token>'. |
| `expires_at` | string | yes |  |
| `user` | object | yes |  |

#### Sample request

```http
POST /auth/login HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "username": "analyst_demo",
  "password": "demo-passphrase-2026"
}
```

#### Sample response

```json
HTTP/1.1 200

{
  "token": "<token>",
  "expires_at": "2026-10-18T09:29:00.507677+00:00",
  "user": {
    "id": 1,
    "username": "analyst_demo",
    "created_at": "2026-10-11T09:29:00.430735+00:00"
  }
}
```

### POST /auth/logout

**Sign out**

Revoke the token. Always succeeds (logging out twice is harmless).

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/auth/logout` |
| Authentication | Bearer token (the session to end) |

#### Request format

No request body.

#### Response format

`204` · no body

#### Sample request

```http
POST /auth/logout HTTP/1.1
Host: localhost:8000
Authorization: Bearer <token>
```

#### Sample response

```json
HTTP/1.1 204

(no body)
```

### GET /auth/me

**Current user**

| Property | Value |
|---|---|
| Method | `GET` |
| Path | `/auth/me` |
| Authentication | Bearer token required |

#### Request format

No request body.

#### Response format

`200` · application/json

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | integer | yes |  |
| `username` | string | yes |  |
| `created_at` | string | yes |  |

#### Sample request

```http
GET /auth/me HTTP/1.1
Host: localhost:8000
Authorization: Bearer <token>
```

#### Sample response

```json
HTTP/1.1 200

{
  "id": 1,
  "username": "analyst_demo",
  "created_at": "2026-10-11T09:29:00.430735+00:00"
}
```

# 4. Assistant

### POST /chat

**Ask the assistant**

Answer one message. The response includes the agent trace and guardrail results.

Pass the returned `thread_id` with the next message to continue the conversation
(follow-up answers, "what about its risk?").

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/chat` |
| Authentication | Bearer token required |

#### Request format

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | yes | The user's question. (min length 1, max length 4000) |
| `thread_id` | string | no | Conversation ID from an earlier response. Omit to start a new conversation. |
| `judge` | boolean | no | Grade the answer with LLM-as-Judge (one extra LLM call). Omit to use JUDGE_MODE from .env. |

#### Response format

`200` · application/json

| Field | Type | Required | Description |
|---|---|---|---|
| `thread_id` | string | yes |  |
| `answer` | string | yes |  |
| `route` | string | yes | Supervisor route, or 'blocked' when the input guardrail stopped it. |
| `route_reason` | string | no |  |
| `agents_run` | array of string | no |  |
| `needs_follow_up` | boolean | yes | True when the answer is a question asking for missing metrics. |
| `findings` | object | no | Each agent's findings, used for the final answer. |
| `evidence` | array of object | no | Every tool call this turn: agent, tool, args, result. |
| `trace` | array of object | no | Guardrail, route, agent, tool and final-answer events. |
| `guardrails` | object | no |  |
| `turn_id` | string | no | ID of this turn; send it with POST /feedback. |
| `judgement` | object | no | LLM-as-Judge result when requested: verdict, overall, scores, reasons, unsupported claims. |
| `elapsed_ms` | integer | yes |  |

#### Sample request

```http
POST /chat HTTP/1.1
Host: localhost:8000
Authorization: Bearer <token>
Content-Type: application/json

{
  "message": "What attack is incident INC-004393?"
}
```

#### Sample response

```json
HTTP/1.1 200

{
  "thread_id": "6ce55595a2714d46ba1d3a5122f2d806",
  "answer": "**Assessment**\n\nIncident INC-004393 has been classified as **Infiltration** with 90 % prediction confidence. The historical database also labels it as Infiltration, providing strong agreement.\n\n**Why this label?**\n\n- Very long, low-and-slow flow (≈ 68 s) with only 40 packets, indicating a stealth reconnaissance or C&C channel.  \n- Numerous idle periods (≈ 33 s each) and tiny packets (average ≈ 95 B) are typical of low-bandwidth intrusion attempts.  \n- Only a SYN/ACK handshake is visible; no data exfiltration or reset packets.  \n- Port 444 is commonly used for custom communication channels by threat actors.  \n\n**Similar past incidents**\n\n- No prior incident IDs are cited in the findings; the …",
  "route": "classify",
  "route_reason": "User asks to identify the attack type of a specific incident.",
  "agents_run": [
    "classification"
  ],
  "needs_follow_up": false,
  "findings": {
    "classification": "**Incident INC‑004393**\n\n**Model Prediction**\n- **Attack type:** *Infiltration*  \n- **Confidence:** 90 % (high)  \n- **Alternative labels:** BENIGN (9 %), DoS Slowhttptest (1 %)\n\n**Historical Label** (from the incident database) – **Infiltration** – matches the prediction.\n\n**Key Metrics (used in the prediction)**\n- Duration: 67 917 167 µs (≈68 s) – relatively long flow with low activity  \n- Total packets: 20 FWD + 20 BWD (≈0.59 packets/s) – sparse traffic  \n- Forward/Backward byte size: 3 738 B / 120 B – very small overall payload  \n- Mean forward packet size: 186.9 B; mean backward packet size: 6 B (very few tiny packets)  \n- SYN/ACK flags present (1 each), no RST/PSH/FIN – classic TCP hand …"
  },
  "evidence": [
    {
      "agent": "classification",
      "tool": "lookup_incident",
      "args": {
        "incident_id": "INC-004393"
      },
      "result": {
        "status": "ok",
        "incident_id": "INC-004393",
        "historical_label": "Infiltration",
        "dst_port": 444,
        "input": {
          "input_quality": "full",
          "known_metrics": "24/24",
          "derived": [],
          "filled_with_normal_values": [],
          "warnings": []
        },
        "metrics": {
          "Flow duration": 67917167.0,
          "Forward packets": 20.0,
          "Backward packets": 20.0,
          "Forward bytes": 3738.0,
          "Backward bytes": 120.0,
          "Flow packet rate": 0.589,
          "Forward packet rate": 0.2945,
          "Backward packet rate": 0.2945,
          "Mean forward packet size": 186.9,
          "Mean backward packet size": 6.0,
          "Minimum packet size": 6.0,
          "Maximum packet size": 705.0,
          "Mean packet size": 94.9756,
          "Packet size std. dev.": 181.0974,
          "SYN flag": 1.0,
          "ACK flag": 1.0,
          "RST flag": 0.0,
          "PSH flag": 0.0,
          "FIN flag": 0.0,
          "Download/upload ratio": 1.0,
          "Initial TCP window (forward)": 255.0,
          "Initial TCP window (backward)": 1452.0,
          "Mean active time": 566477.5,
          "Mean idle time": 33100000.0
        }
      }
    },
    "… 1 more"
  ],
  "trace": [
    {
      "type": "guard_input",
      "action": "allowed",
      "category": null,
      "matched": null,
      "redactions": [],
      "truncated": false,
      "elapsed_ms": 0
    },
    "… 9 more"
  ],
  "guardrails": {
    "input": {
      "action": "allowed",
      "category": null,
      "matched": null,
      "redactions": [],
      "truncated": false
    },
    "output": {
      "issues": [],
      "changed": false
    }
  },
  "turn_id": "1418ee75519f4518a35e64bb48945262",
  "judgement": null,
  "elapsed_ms": 44957
}
```

### POST /chat/stream

**Ask the assistant (live trace)**

Answer one message as Server-Sent Events.

- `event: trace`  one per step (guardrail, route, agent start/end, tool result, final answer)
- `event: result` the full response (same shape as POST /chat), sent last
- `event: error`  if the run fails part-way

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/chat/stream` |
| Authentication | Bearer token required |

#### Request format

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | yes | The user's question. (min length 1, max length 4000) |
| `thread_id` | string | no | Conversation ID from an earlier response. Omit to start a new conversation. |
| `judge` | boolean | no | Grade the answer with LLM-as-Judge (one extra LLM call). Omit to use JUDGE_MODE from .env. |

#### Response format

`200` · text/event-stream

#### Sample request

```http
POST /chat/stream HTTP/1.1
Host: localhost:8000
Authorization: Bearer <token>
Content-Type: application/json

{
  "message": "And how risky is it?",
  "thread_id": "6ce55595a2714d46ba1d3a5122f2d806"
}
```

#### Sample response

```
HTTP/1.1 200

event: trace
data: {"type": "guard_input", "action": "allowed", "category": null, "matched": null, "redactions": [], "truncated": false, "elapsed_ms": 0}

event: trace
data: {"type": "route", "route": "risk", "reason": "User asks about the risk level of a previously identified incident", "plan": ["risk_priority"], "elapsed_ms": 1576}

event: trace
data: {"type": "agent_start", "agent": "risk_priority"}

event: trace
data: {"type": "tool_call", "agent": "risk_priority", "tool": "assess_intrusion_risk", "args": {"incident_id": "INC-004393"}}

… 4 more trace events …

event: result
data: {"thread_id": "6ce55595a2714d46ba1d3a5122f2d806", "answer": "**Risk Assessment**\n\n- **Malicious-intent probability:** **99.82 %**  \n- **Risk level:** **Critical** (P1 priority)\n\n**W …", "route": "risk", "route_reason": "User asks about the risk level of a previously identified incident", "agents_run": ["risk_priority"], "needs_follow_up": false, "findings": {"risk_priority": "**Intrusion probability and risk level**\n- Probability of malicious intent: **99.82 %**  \n- Risk level: **Critical**\n\n** …"}, "evidence": [{"agent": "risk_priority", "tool": "assess_intrusion_risk", "args": …
```

### GET /chat/conversations

**Your conversations**

Your saved conversations, most recently used first.

| Property | Value |
|---|---|
| Method | `GET` |
| Path | `/chat/conversations` |
| Authentication | Bearer token required |

#### Request format

No request body.

#### Response format

`200` · application/json

#### Sample request

```http
GET /chat/conversations HTTP/1.1
Host: localhost:8000
Authorization: Bearer <token>
```

#### Sample response

```json
HTTP/1.1 200

[
  {
    "thread_id": "6ce55595a2714d46ba1d3a5122f2d806",
    "title": "What attack is incident INC-004393?",
    "created_at": "2026-10-11T09:29:00.529149+00:00",
    "updated_at": "2026-10-11T09:30:17.875300+00:00"
  }
]
```

### GET /chat/{thread_id}/history

**Conversation history**

| Property | Value |
|---|---|
| Method | `GET` |
| Path | `/chat/{thread_id}/history` |
| Authentication | Bearer token required |
| Parameters | `thread_id` (path, required) |

#### Request format

No request body.

#### Response format

`200` · application/json

| Field | Type | Required | Description |
|---|---|---|---|
| `thread_id` | string | yes |  |
| `messages` | array of object | yes |  |

#### Sample request

```http
GET /chat/6ce55595a2714d46ba1d3a5122f2d806/history HTTP/1.1
Host: localhost:8000
Authorization: Bearer <token>
```

#### Sample response

```json
HTTP/1.1 200

{
  "thread_id": "6ce55595a2714d46ba1d3a5122f2d806",
  "messages": [
    {
      "role": "user",
      "content": "What attack is incident INC-004393?"
    },
    {
      "role": "assistant",
      "content": "**Assessment**\n\nIncident INC-004393 has been classified as **Infiltration** with 90 % prediction confidence. The historical database also labels it as Infiltration, providing strong agreement.\n\n**Why this label?**\n\n- Very long, low-and-slow flow (≈ 68 s) with only 40 packets, indicating a stealth reconnaissance or C&C channel.  \n- Numerous idle periods (≈ 33 s each) and tiny packets (average ≈ 95 …"
    },
    "… 2 more"
  ]
}
```

### PATCH /chat/{thread_id}

**Rename a conversation**

| Property | Value |
|---|---|
| Method | `PATCH` |
| Path | `/chat/{thread_id}` |
| Authentication | Bearer token required |
| Parameters | `thread_id` (path, required) |

#### Request format

| Field | Type | Required | Description |
|---|---|---|---|
| `title` | string | yes | (min length 1, max length 200) |

#### Response format

`200` · application/json

| Field | Type | Required | Description |
|---|---|---|---|
| `thread_id` | string | yes |  |
| `title` | string | yes |  |
| `created_at` | string | yes |  |
| `updated_at` | string | yes |  |

#### Sample request

```http
PATCH /chat/6ce55595a2714d46ba1d3a5122f2d806 HTTP/1.1
Host: localhost:8000
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "Infiltration case"
}
```

#### Sample response

```json
HTTP/1.1 200

{
  "thread_id": "6ce55595a2714d46ba1d3a5122f2d806",
  "title": "Infiltration case",
  "created_at": "2026-10-11T09:29:00.529149+00:00",
  "updated_at": "2026-10-11T09:30:17.875300+00:00"
}
```

### DELETE /chat/{thread_id}

**Delete a conversation**

| Property | Value |
|---|---|
| Method | `DELETE` |
| Path | `/chat/{thread_id}` |
| Authentication | Bearer token required |
| Parameters | `thread_id` (path, required) |

#### Request format

No request body.

#### Response format

`204` · no body

#### Sample request

```http
DELETE /chat/6ce55595a2714d46ba1d3a5122f2d806 HTTP/1.1
Host: localhost:8000
Authorization: Bearer <token>
```

#### Sample response

```json
HTTP/1.1 204

(no body)
```

### POST /judge

**Grade an answer (LLM-as-Judge)**

Score an answer 1-5 on faithfulness, correctness, relevance, actionability and safety,
list unsupported claims, and give a verdict (pass / needs_review / fail).

Send a `thread_id` to grade that conversation's latest answer with its real evidence, or a
`question` + `answer` (optionally with `findings` and `evidence`) to grade any answer.
Grading a conversation needs its owner to be signed in.

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/judge` |
| Authentication | Bearer token required when grading a conversation (`thread_id`) |

#### Request format

| Field | Type | Required | Description |
|---|---|---|---|
| `thread_id` | string | no |  |
| `question` | string | no |  |
| `answer` | string | no |  |
| `route` | string | no | Route of the answer (actionability is scored for recommend / investigate). |
| `findings` | object | no | Agent findings the answer was written from. |
| `evidence` | array of object | no | Tool calls: agent, tool, args, result. |

#### Response format

`200` · application/json

| Field | Type | Required | Description |
|---|---|---|---|
| `verdict` | `pass` \| `needs_review` \| `fail` \| `error` | yes |  |
| `overall` | number | no |  |
| `scores` | object | no |  |
| `reasons` | object | no |  |
| `applicable` | array of string | no |  |
| `unsupported_claims` | array of string | no |  |
| `error` | string | no |  |
| `judge_model` | string | no |  |

#### Sample request

```http
POST /judge HTTP/1.1
Host: localhost:8000
Authorization: Bearer <token>
Content-Type: application/json

{
  "thread_id": "6ce55595a2714d46ba1d3a5122f2d806"
}
```

#### Sample response

```json
HTTP/1.1 200

{
  "verdict": "pass",
  "overall": 5.0,
  "scores": {
    "faithfulness": 5,
    "correctness": 5,
    "relevance": 5,
    "actionability": 5,
    "safety": 5
  },
  "reasons": {
    "faithfulness": "All factual claims directly match the evidence and findings.",
    "correctness": "Risk probability, level, anomaly, and priority all match the tool output and findings.",
    "relevance": "Answer addresses risk assessment and covers key findings of the evidence.",
    "actionability": "Steps are specific, prioritized, and actionable for an analyst.",
    "safety": "No harmful instructions; uncertainties and caveats are stated."
  },
  "applicable": [
    "faithfulness",
    "correctness",
    "… 2 more"
  ],
  "unsupported_claims": [],
  "error": null,
  "judge_model": "nvidia:openai/gpt-oss-20b"
}
```

# 5. Feedback

### POST /feedback

**Rate an answer**

Thumbs up/down, an optional comment and an optional corrected attack type for an answer.

The question, answer and model prediction are taken from the conversation itself. Rating
the same turn again updates the earlier feedback. A corrected label is shown in later
answers about the same incident and is used when the classifier is retrained.
Only the owner of the conversation can rate it; the rating records their username.

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/feedback` |
| Authentication | Bearer token required |

#### Request format

| Field | Type | Required | Description |
|---|---|---|---|
| `thread_id` | string | yes |  |
| `turn_id` | string | no | Turn to rate (from the chat response). Omit to rate the conversation's latest answer. |
| `rating` | `up` \| `down` | yes |  |
| `comment` | string | no |  |
| `corrected_label` | string | no | Correct attack type, if the prediction was wrong (one of the 15 types; case-insensitive). |

#### Response format

`201` · application/json

| Field | Type |
|---|---|
| `id` | number |
| `created_at` | string |
| `updated_at` | string |
| `thread_id` | string |
| `turn_id` | string |
| `rating` | string |
| `comment` | string |
| `corrected_label` | string |
| `question` | string |
| `answer` | string |
| `route` | string |
| `agents_run` | array |
| `predicted_label` | string |
| `prediction_confidence` | number |
| `incident_id` | string |
| `historical_label` | string |
| `flow_args` | object |
| `input_quality` | string |
| `judge_verdict` | null |
| `judge_overall` | null |
| `llm_model` | string |
| `username` | string |

#### Sample request

```http
POST /feedback HTTP/1.1
Host: localhost:8000
Authorization: Bearer <token>
Content-Type: application/json

{
  "thread_id": "6ce55595a2714d46ba1d3a5122f2d806",
  "turn_id": "1418ee75519f4518a35e64bb48945262",
  "rating": "down",
  "comment": "Should be Bot",
  "corrected_label": "Bot"
}
```

#### Sample response

```json
HTTP/1.1 201

{
  "id": 1,
  "created_at": "2026-10-11T09:30:25+00:00",
  "updated_at": "2026-10-11T09:30:25+00:00",
  "thread_id": "6ce55595a2714d46ba1d3a5122f2d806",
  "turn_id": "1418ee75519f4518a35e64bb48945262",
  "rating": "down",
  "comment": "Should be Bot",
  "corrected_label": "Bot",
  "question": "What attack is incident INC-004393?",
  "answer": "**Assessment**\n\nIncident INC-004393 has been classified as **Infiltration** with 90 % prediction confidence. The historical database also labels it as Infiltration, providing strong agreement.\n\n**Why …",
  "route": "classify",
  "agents_run": [
    "classification"
  ],
  "predicted_label": "Infiltration",
  "prediction_confidence": 0.9,
  "incident_id": "INC-004393",
  "historical_label": "Infiltration",
  "flow_args": {
    "flow_duration": 67917167.0,
    "total_fwd_packets": 20.0,
    "total_bwd_packets": 20.0,
    "total_len_fwd": 3738.0,
    "total_len_bwd": 120.0,
    "flow_packets_s": 0.589,
    "fwd_packets_s": 0.2945,
    "bwd_packets_s": 0.2945,
    "fwd_packet_length_mean": 186.9,
    "bwd_packet_length_mean": 6.0,
    "min_packet_length": 6.0,
    "max_packet_length": 705.0,
    "packet_length_mean": 94.9756,
    "packet_length_std": 181.0974,
    "syn_flag_count": 1.0,
    "ack_flag_count": 1.0,
    "rst_flag_count": 0.0,
    "psh_flag_count": 0.0,
    "fin_flag_count": 0.0,
    "down_up_ratio": 1.0,
    "init_win_bytes_forward": 255.0,
    "init_win_bytes_backward": 1452.0,
    "active_mean": 566477.5,
    "idle_mean": 33100000.0
  },
  "input_quality": "full",
  "judge_verdict": null,
  "judge_overall": null,
  "llm_model": "nvidia:openai/gpt-oss-20b",
  "username": "analyst_demo"
}
```

### GET /feedback

**List feedback**

Stored feedback, newest first.

| Property | Value |
|---|---|
| Method | `GET` |
| Path | `/feedback` |
| Authentication | None |
| Parameters | `rating` (query); `route` (query); `label` (query); `corrected_only` (query); `limit` (query); `offset` (query) |

#### Request format

No request body.

#### Response format

`200` · application/json

#### Sample request

```http
GET /feedback?limit=5 HTTP/1.1
Host: localhost:8000
```

#### Sample response

```json
HTTP/1.1 200

[
  {
    "id": 1,
    "created_at": "2026-10-11T09:30:25+00:00",
    "updated_at": "2026-10-11T09:30:25+00:00",
    "thread_id": "6ce55595a2714d46ba1d3a5122f2d806",
    "turn_id": "1418ee75519f4518a35e64bb48945262",
    "rating": "down",
    "comment": "Should be Bot",
    "corrected_label": "Bot",
    "question": "What attack is incident INC-004393?",
    "answer": "**Assessment**\n\nIncident INC-004393 has been classified as **Infiltration** with 90 % prediction confidence. The historical database also labels it as Infiltrat …",
    "route": "classify",
    "agents_run": [
      "classification"
    ],
    "predicted_label": "Infiltration",
    "prediction_confidence": 0.9,
    "incident_id": "INC-004393",
    "historical_label": "Infiltration",
    "flow_args": {
      "flow_duration": 67917167.0,
      "total_fwd_packets": 20.0,
      "total_bwd_packets": 20.0,
      "total_len_fwd": 3738.0,
      "total_len_bwd": 120.0,
      "flow_packets_s": 0.589,
      "fwd_packets_s": 0.2945,
      "bwd_packets_s": 0.2945,
      "fwd_packet_length_mean": 186.9,
      "bwd_packet_length_mean": 6.0,
      "min_packet_length": 6.0,
      "max_packet_length": 705.0,
      "packet_length_mean": 94.9756,
      "packet_length_std": 181.0974,
      "syn_flag_count": 1.0,
      "ack_flag_count": 1.0,
      "rst_flag_count": 0.0,
      "psh_flag_count": 0.0,
      "fin_flag_count": 0.0,
      "down_up_ratio": 1.0,
      "init_win_bytes_forward": 255.0,
      "init_win_bytes_backward": 1452.0,
      "active_mean": 566477.5,
      "idle_mean": 33100000.0
    },
    "input_quality": "full",
    "judge_verdict": null,
    "judge_overall": null,
    "llm_model": "nvidia:openai/gpt-oss-20b",
    "username": "analyst_demo"
  }
]
```

### GET /feedback/stats

**Feedback statistics**

Approval rate (overall and per route), agreement between the LLM judge and analysts,
the most common prediction -> correction pairs, and recent comments.

| Property | Value |
|---|---|
| Method | `GET` |
| Path | `/feedback/stats` |
| Authentication | None |

#### Request format

No request body.

#### Response format

`200` · application/json

| Field | Type |
|---|---|
| `total` | number |
| `up` | number |
| `down` | number |
| `approval_rate` | number |
| `by_route` | object |
| `judge_agreement` | object |
| `corrections` | object |
| `recent_comments` | array |

#### Sample request

```http
GET /feedback/stats HTTP/1.1
Host: localhost:8000
```

#### Sample response

```json
HTTP/1.1 200

{
  "total": 1,
  "up": 0,
  "down": 1,
  "approval_rate": 0.0,
  "by_route": {
    "classify": {
      "n": 1,
      "up": 0,
      "approval_rate": 0.0
    }
  },
  "judge_agreement": {
    "n": 0,
    "pass_up": 0,
    "pass_down": 0,
    "not_pass_up": 0,
    "not_pass_down": 0,
    "agreement_rate": null
  },
  "corrections": {
    "total": 1,
    "top_pairs": [
      {
        "predicted": "Infiltration",
        "corrected": "Bot",
        "count": 1
      }
    ]
  },
  "recent_comments": [
    {
      "rating": "down",
      "route": "classify",
      "comment": "Should be Bot",
      "created_at": "2026-10-11T09:30:25+00:00"
    }
  ]
}
```

### GET /feedback/export.csv

**Export label corrections**

Corrected flows for retraining: 24 features, corrected label (majority of analysts),
source incident, votes. Corrections without usable features are left out.

| Property | Value |
|---|---|
| Method | `GET` |
| Path | `/feedback/export.csv` |
| Authentication | None |

#### Request format

No request body.

#### Response format

`200` · text/csv

#### Sample request

```http
GET /feedback/export.csv HTTP/1.1
Host: localhost:8000
```

#### Sample response

```
HTTP/1.1 200

flow_duration,total_fwd_packets,total_bwd_packets,total_len_fwd,total_len_bwd,flow_packets_s,… (28 columns)
67917167.0,20.0,20.0,3738.0,120.0,0.588952716,…
```

# 6. Machine-learning models

### POST /ml/classify

**Attack type of a flow**

Predicted attack type (15 classes incl. BENIGN), confidence and alternatives.

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/ml/classify` |
| Authentication | None |

#### Request format

A flow (see [Flow input](#flow-input)): any subset of the 24 metrics, or an `incident_id`.

#### Response format

`200` · application/json

| Field | Type |
|---|---|
| `incident` | object |
| `status` | string |
| `input` | object |
| `attack_type` | object |
| `overall_confidence` | string |

#### Sample request

```http
POST /ml/classify HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "incident_id": "INC-000035"
}
```

#### Sample response

```json
HTTP/1.1 200

{
  "incident": {
    "incident_id": "INC-000035",
    "historical_label": "SSH-Patator",
    "dst_port": 22
  },
  "status": "ok",
  "input": {
    "input_quality": "full",
    "known_metrics": "24/24",
    "derived": [],
    "filled_with_normal_values": [],
    "warnings": []
  },
  "attack_type": {
    "prediction": "SSH-Patator",
    "confidence": 1.0,
    "alternatives": []
  },
  "overall_confidence": "high"
}
```

### POST /ml/risk

**Intrusion risk and anomaly of a flow**

Calibrated intrusion probability, risk level, and anomaly level vs normal traffic.

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/ml/risk` |
| Authentication | None |

#### Request format

A flow (see [Flow input](#flow-input)): any subset of the 24 metrics, or an `incident_id`.

#### Response format

`200` · application/json

| Field | Type |
|---|---|
| `status` | string |
| `input` | object |
| `risk` | object |
| `overall_confidence` | string |
| `anomaly` | object |

#### Sample request

```http
POST /ml/risk HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "flow_duration": "2 s",
  "total_fwd_packets": 10,
  "total_bwd_packets": 8,
  "total_len_fwd": "1.2 KB",
  "total_len_bwd": "4 KB"
}
```

#### Sample response

```json
HTTP/1.1 200

{
  "status": "ok",
  "input": {
    "input_quality": "reduced",
    "known_metrics": "12/24",
    "derived": [
      "flow_packets_s",
      "fwd_packets_s",
      "… 5 more"
    ],
    "filled_with_normal_values": [
      "min_packet_length",
      "max_packet_length",
      "… 10 more"
    ],
    "warnings": [
      "packet_length_mean was approximated from total bytes and packets",
      "12 feature(s) not provided were filled with normal-traffic medians"
    ]
  },
  "risk": {
    "probability": 0.0048,
    "is_intrusion": false,
    "level": "Low"
  },
  "overall_confidence": "medium",
  "anomaly": {
    "score": 0.5286,
    "benign_percentile": 81.8,
    "level": "Low",
    "most_unusual_features": []
  }
}
```

### POST /ml/explain

**Full analysis with SHAP reasons**

Attack type, risk, anomaly, priority, and the metrics that drove each prediction (SHAP).

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/ml/explain` |
| Authentication | None |

#### Request format

A flow (see [Flow input](#flow-input)): any subset of the 24 metrics, or an `incident_id`.

#### Response format

`200` · application/json

| Field | Type |
|---|---|
| `incident` | object |
| `status` | string |
| `input` | object |
| `attack_type` | object |
| `risk` | object |
| `anomaly` | object |
| `overall_confidence` | string |
| `explanation` | object |
| `priority` | object |

#### Sample request

```http
POST /ml/explain HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "incident_id": "INC-000035"
}
```

#### Sample response

```json
HTTP/1.1 200

{
  "incident": {
    "incident_id": "INC-000035",
    "historical_label": "SSH-Patator",
    "dst_port": 22
  },
  "status": "ok",
  "input": {
    "input_quality": "full",
    "known_metrics": "24/24",
    "derived": [],
    "filled_with_normal_values": [],
    "warnings": []
  },
  "attack_type": {
    "prediction": "SSH-Patator",
    "confidence": 1.0,
    "alternatives": []
  },
  "risk": {
    "probability": 1.0,
    "is_intrusion": true,
    "level": "Critical"
  },
  "anomaly": {
    "score": 0.504,
    "benign_percentile": 71.8,
    "level": "Low",
    "most_unusual_features": [
      {
        "feature": "total_bwd_packets",
        "value": 32.0,
        "benign_percentile": 96.7,
        "text": "Backward packets = 32 packets (higher than 97% of normal traffic)"
      }
    ]
  },
  "overall_confidence": "high",
  "explanation": {
    "attack_type_reasons": [
      {
        "feature": "total_bwd_packets",
        "value": 32.0,
        "shap": 0.1875,
        "effect": "toward SSH-Patator",
        "benign_percentile": 96.7,
        "text": "Backward packets = 32 packets (higher than 97% of normal traffic)"
      },
      {
        "feature": "total_fwd_packets",
        "value": 21.0,
        "shap": 0.1619,
        "effect": "toward SSH-Patator",
        "benign_percentile": 93.5,
        "text": "Forward packets = 21 packets (higher than 94% of normal traffic)"
      },
      "… 1 more"
    ],
    "risk_reasons": [
      {
        "feature": "init_win_bytes_forward",
        "value": 29200.0,
        "shap": 4.3939,
        "effect": "raises risk",
        "benign_percentile": 88.6,
        "text": "Initial TCP window (forward) = 29,200 bytes (higher than 89% of normal traffic)"
      },
      {
        "feature": "init_win_bytes_backward",
        "value": 247.0,
        "shap": 2.2831,
        "effect": "raises risk",
        "benign_percentile": 71.5,
        "text": "Initial TCP window (backward) = 247 bytes (higher than 72% of normal traffic)"
      },
      "… 1 more"
    ],
    "summary": "Classified as SSH-Patator (100% confidence). Intrusion risk 100% (Critical). Anomaly level Low. Main reasons: Backward packets = 32 packets (higher than 97% of normal traffic); Forward packets = 21 packets (higher than 94% of normal traffic); Forward bytes = 2 …"
  },
  "priority": {
    "priority": "P2",
    "score": 66.0,
    "label": "High",
    "action": "Investigate today",
    "reason": "SSH-Patator (100%, severity Medium), intrusion risk 100%, anomaly Low",
    "components": {
      "risk": 1.0,
      "severity": 0.6,
      "anomaly": 0.0
    },
    "overrides_applied": [],
    "confidence_note": null
  }
}
```

### POST /ml/prioritize

**Priority of one flow**

P1 (Critical) .. P4 (Low) with score, reason and rule overrides.

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/ml/prioritize` |
| Authentication | None |

#### Request format

A flow (see [Flow input](#flow-input)): any subset of the 24 metrics, or an `incident_id`.

#### Response format

`200` · application/json

| Field | Type |
|---|---|
| `incident` | object |
| `status` | string |
| `input` | object |
| `attack_type` | object |
| `risk` | object |
| `anomaly` | object |
| `overall_confidence` | string |
| `priority` | object |

#### Sample request

```http
POST /ml/prioritize HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "incident_id": "INC-002545"
}
```

#### Sample response

```json
HTTP/1.1 200

{
  "incident": {
    "incident_id": "INC-002545",
    "historical_label": "Heartbleed",
    "dst_port": 444
  },
  "status": "ok",
  "input": {
    "input_quality": "full",
    "known_metrics": "24/24",
    "derived": [],
    "filled_with_normal_values": [],
    "warnings": []
  },
  "attack_type": {
    "prediction": "Heartbleed",
    "confidence": 1.0,
    "alternatives": []
  },
  "risk": {
    "probability": 1.0,
    "is_intrusion": true,
    "level": "Critical"
  },
  "anomaly": {
    "score": 0.6566,
    "benign_percentile": 99.9,
    "level": "High",
    "most_unusual_features": [
      {
        "feature": "bwd_packet_length_mean",
        "value": 3808.3789,
        "benign_percentile": 100.1,
        "text": "Mean backward packet size = 3,808.38 bytes (higher than 99.9% of normal traffic)"
      },
      {
        "feature": "total_fwd_packets",
        "value": 2801.0,
        "benign_percentile": 100.0,
        "text": "Forward packets = 2,801 packets (higher than 99.9% of normal traffic)"
      },
      "… 1 more"
    ]
  },
  "overall_confidence": "high",
  "priority": {
    "priority": "P1",
    "score": 99.8,
    "label": "Critical",
    "action": "Investigate now",
    "reason": "Heartbleed (100%, severity Critical), intrusion risk 100%, anomaly High",
    "components": {
      "risk": 1.0,
      "severity": 1.0,
      "anomaly": 0.99
    },
    "overrides_applied": [],
    "confidence_note": null
  }
}
```

### POST /ml/prioritize/batch

**Rank many flows / incidents**

Rank flows and/or incident IDs by urgency; returns the top_n and counts per priority.

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/ml/prioritize/batch` |
| Authentication | None |

#### Request format

| Field | Type | Required | Description |
|---|---|---|---|
| `flows` | array of object | no | (max items 1000) |
| `incident_ids` | array of string | no | (max items 1000) |
| `top_n` | integer | no | How many of the most urgent flows to return. (≥ 1.0, ≤ 1000.0, default 10) |

#### Response format

`200` · application/json

| Field | Type |
|---|---|
| `status` | string |
| `total_flows` | number |
| `priority_counts` | object |
| `top` | array |

#### Sample request

```http
POST /ml/prioritize/batch HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "incident_ids": [
    "INC-000001",
    "INC-000035",
    "INC-002545",
    "INC-000005"
  ],
  "top_n": 3
}
```

#### Sample response

```json
HTTP/1.1 200

{
  "status": "ok",
  "total_flows": 4,
  "priority_counts": {
    "P1": 2,
    "P2": 1,
    "P3": 1
  },
  "top": [
    {
      "incident_id": "INC-002545",
      "historical_label": "Heartbleed",
      "priority": "P1",
      "score": 99.8,
      "attack_type": "Heartbleed",
      "confidence": 1.0,
      "risk_probability": 1.0,
      "anomaly_level": "High",
      "reason": "Heartbleed (100%, severity Critical), intrusion risk 100%, anomaly High",
      "overrides": [],
      "input_quality": "full"
    },
    {
      "incident_id": "INC-000001",
      "historical_label": "DoS Hulk",
      "priority": "P1",
      "score": 85.6,
      "attack_type": "DoS Hulk",
      "confidence": 1.0,
      "risk_probability": 1.0,
      "anomaly_level": "Medium",
      "reason": "DoS Hulk (100%, severity High), intrusion risk 100%, anomaly Medium",
      "overrides": [],
      "input_quality": "full"
    },
    "… 1 more"
  ]
}
```

### POST /ml/prioritize/csv

**Rank flows from a CSV file**

Upload a CSV (max 5 MB, 1000 rows). Each row is a flow; `flow_index` in the result is the row number.

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/ml/prioritize/csv` |
| Authentication | None |
| Parameters | `top_n` (query) |

#### Request format

`multipart/form-data` with a `file` field: a CSV of flows (≤ 5 MB, ≤ 1,000 rows). Columns are matched by metric name, alias or raw CIC-IDS2017 header; an `incident_id` column is also accepted.

#### Response format

`200` · application/json

| Field | Type |
|---|---|
| `status` | string |
| `total_flows` | number |
| `priority_counts` | object |
| `top` | array |

#### Sample request

```http
POST /ml/prioritize/csv?top_n=3 HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data

file = flows.csv:
incident_id
INC-000001
INC-000035
INC-002545

```

#### Sample response

```json
HTTP/1.1 200

{
  "status": "ok",
  "total_flows": 3,
  "priority_counts": {
    "P1": 2,
    "P2": 1
  },
  "top": [
    {
      "incident_id": "INC-002545",
      "historical_label": "Heartbleed",
      "priority": "P1",
      "score": 99.8,
      "attack_type": "Heartbleed",
      "confidence": 1.0,
      "risk_probability": 1.0,
      "anomaly_level": "High",
      "reason": "Heartbleed (100%, severity Critical), intrusion risk 100%, anomaly High",
      "overrides": [],
      "input_quality": "full"
    },
    {
      "incident_id": "INC-000001",
      "historical_label": "DoS Hulk",
      "priority": "P1",
      "score": 85.6,
      "attack_type": "DoS Hulk",
      "confidence": 1.0,
      "risk_probability": 1.0,
      "anomaly_level": "Medium",
      "reason": "DoS Hulk (100%, severity High), intrusion risk 100%, anomaly Medium",
      "overrides": [],
      "input_quality": "full"
    },
    "… 1 more"
  ]
}
```

# 7. Incident search

### POST /search

**Find similar historical incidents**

Hybrid keyword + semantic search over ~97,000 incidents, with optional attack-type and
port filters and numeric conditions in the query ("more than 10000 packets per second").
Retrieval only: no generated answer.

The first search waits for the indexes if they are still loading (see GET /health).

| Property | Value |
|---|---|
| Method | `POST` |
| Path | `/search` |
| Authentication | None |

#### Request format

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | Description of the traffic or incident. (min length 1, max length 1000) |
| `attack_type` | string | no | Only incidents of this attack type. |
| `dst_port` | integer | no | Only incidents on this port. |
| `top_k` | integer | no | (≥ 1.0, ≤ 20.0, default 5) |

#### Response format

`200` · application/json

| Field | Type |
|---|---|
| `status` | string |
| `query` | string |
| `filters_applied` | object |
| `filters_inferred_from_query` | object |
| `numeric_conditions` | array |
| `candidates_after_filters` | number |
| `evidence_available` | boolean |
| `incidents` | array |

#### Sample request

```http
POST /search HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "query": "SSH brute force with many short connections",
  "attack_type": "SSH-Patator",
  "top_k": 3
}
```

#### Sample response

```json
HTTP/1.1 200

{
  "status": "ok",
  "query": "SSH brute force with many short connections",
  "filters_applied": {
    "label": "SSH-Patator"
  },
  "filters_inferred_from_query": {},
  "numeric_conditions": [],
  "candidates_after_filters": 3149,
  "evidence_available": true,
  "incidents": [
    {
      "incident_id": "INC-002563",
      "attack_type": "SSH-Patator",
      "dst_port": 22,
      "relevance_score": 0.0164,
      "summary": "Attack: SSH-Patator. Destination port: 22. Flow duration: 50 microseconds. Forward packets: 1. Backward packets: 1. Forward bytes: 0. Backward bytes: 0. Flow packets per second: 40000. Forward packets per second: 20000. …"
    },
    {
      "incident_id": "INC-079784",
      "attack_type": "SSH-Patator",
      "dst_port": 22,
      "relevance_score": 0.0164,
      "summary": "Attack: SSH-Patator. Destination port: 22. Flow duration: 2138428 microseconds. Forward packets: 20. Backward packets: 18. Forward bytes: 2296. Backward bytes: 3069. Flow packets per second: 17.7701. Forward packets per …"
    },
    "… 1 more"
  ]
}
```

### GET /incidents/{incident_id}

**Look up a historical incident**

Stored metrics, destination port and historical label of an incident (e.g. INC-000035).

| Property | Value |
|---|---|
| Method | `GET` |
| Path | `/incidents/{incident_id}` |
| Authentication | None |
| Parameters | `incident_id` (path, required) |

#### Request format

No request body.

#### Response format

`200` · application/json

| Field | Type |
|---|---|
| `status` | string |
| `incident_id` | string |
| `historical_label` | string |
| `dst_port` | number |
| `input` | object |
| `metrics` | object |

#### Sample request

```http
GET /incidents/INC-000035 HTTP/1.1
Host: localhost:8000
```

#### Sample response

```json
HTTP/1.1 200

{
  "status": "ok",
  "incident_id": "INC-000035",
  "historical_label": "SSH-Patator",
  "dst_port": 22,
  "input": {
    "input_quality": "full",
    "known_metrics": "24/24",
    "derived": [],
    "filled_with_normal_values": [],
    "warnings": []
  },
  "metrics": {
    "Flow duration": 13903754.0,
    "Forward packets": 21.0,
    "Backward packets": 32.0,
    "Forward bytes": 2008.0,
    "Backward bytes": 2745.0,
    "Flow packet rate": 3.8119,
    "Forward packet rate": 1.5104,
    "Backward packet rate": 2.3015,
    "Mean forward packet size": 95.619,
    "Mean backward packet size": 85.7812,
    "Minimum packet size": 0.0,
    "Maximum packet size": 976.0,
    "Mean packet size": 88.0185,
    "Packet size std. dev.": 189.5903,
    "SYN flag": 0.0,
    "ACK flag": 0.0,
    "RST flag": 0.0,
    "PSH flag": 1.0,
    "FIN flag": 0.0,
    "Download/upload ratio": 1.0,
    "Initial TCP window (forward)": 29200.0,
    "Initial TCP window (backward)": 247.0,
    "Mean active time": 0.0,
    "Mean idle time": 0.0
  }
}
```

### GET /attack-types

**List attack types**

The 15 classes the models know (incl. BENIGN) with their severity.

| Property | Value |
|---|---|
| Method | `GET` |
| Path | `/attack-types` |
| Authentication | None |

#### Request format

No request body.

#### Response format

`200` · application/json

| Field | Type |
|---|---|
| `status` | string |
| `attack_types` | array |

#### Sample request

```http
GET /attack-types HTTP/1.1
Host: localhost:8000
```

#### Sample response

```json
HTTP/1.1 200

{
  "status": "ok",
  "attack_types": [
    {
      "attack_type": "BENIGN",
      "severity": "None"
    },
    {
      "attack_type": "Bot",
      "severity": "High"
    },
    {
      "attack_type": "DDoS",
      "severity": "High"
    },
    "… 12 more"
  ]
}
```

### GET /attack-types/{name}

**Profile of an attack type**

Typical traffic of an attack type: which metrics differ from normal traffic, with medians.

| Property | Value |
|---|---|
| Method | `GET` |
| Path | `/attack-types/{name}` |
| Authentication | None |
| Parameters | `name` (path, required) |

#### Request format

No request body.

#### Response format

`200` · application/json

| Field | Type |
|---|---|
| `status` | string |
| `attack_type` | string |
| `severity` | string |
| `summary` | string |
| `top_features` | array |

#### Sample request

```http
GET /attack-types/DDoS HTTP/1.1
Host: localhost:8000
```

#### Sample response

```json
HTTP/1.1 200

{
  "status": "ok",
  "attack_type": "DDoS",
  "severity": "High",
  "summary": "Initial TCP window (backward) typically higher than normal (median 229 vs 0); Backward bytes typically higher than normal (median 11,601 vs 163); Mean backward packet size typically higher than normal (median 1,934.50 vs 95); Forward bytes typically lower than …",
  "top_features": [
    {
      "feature": "Initial TCP window (backward)",
      "direction_vs_normal": "higher",
      "attack_median": 229.0,
      "normal_median": 0.0,
      "unit": "bytes",
      "importance_share": 0.1043
    },
    {
      "feature": "Backward bytes",
      "direction_vs_normal": "higher",
      "attack_median": 11601.0,
      "normal_median": 163.0,
      "unit": "bytes",
      "importance_share": 0.1042
    },
    "… 3 more"
  ]
}
```

