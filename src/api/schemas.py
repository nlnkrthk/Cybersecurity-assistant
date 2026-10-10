"""
Request and response models of the API.

Flow inputs reuse the agent tool schema (agents.tools.ml_tools.FlowArgs): any subset
of the 24 flow metrics as numbers or unit strings, or a historical incident_id.
ML results are returned as produced by the tools (documented by the examples).
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from agents.guardrails import MAX_INPUT_CHARS
from agents.tools.ml_tools import MAX_BATCH_FLOWS, FlowArgs

THREAD_ID_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"

FLOW_EXAMPLES = [
    {"incident_id": "INC-000035"},
    {
        "flow_duration": "2 s",
        "total_fwd_packets": 10,
        "total_bwd_packets": 8,
        "total_len_fwd": "1.2 KB",
        "total_len_bwd": "4 KB",
    },
]


# ============================================================================
# Chat
# ============================================================================
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_INPUT_CHARS, description="The user's question.")
    thread_id: Optional[str] = Field(
        default=None,
        pattern=THREAD_ID_PATTERN,
        description="Conversation ID from an earlier response. Omit to start a new conversation.",
    )
    judge: Optional[bool] = Field(
        default=None,
        description="Grade the answer with LLM-as-Judge (one extra LLM call). Omit to use JUDGE_MODE from .env.",
    )

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"message": "Investigate incident INC-000035 and tell me how to respond", "judge": True},
        {"message": "What about its risk?", "thread_id": "3f2a9c..."},
    ]})


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    route: Optional[str] = Field(description="Supervisor route, or 'blocked' when the input guardrail stopped it.")
    route_reason: Optional[str] = None
    agents_run: List[str] = []
    needs_follow_up: bool = Field(description="True when the answer is a question asking for missing metrics.")
    findings: Dict[str, str] = Field(default={}, description="Each agent's findings, used for the final answer.")
    evidence: List[Dict[str, Any]] = Field(default=[], description="Every tool call this turn: agent, tool, args, result.")
    trace: List[Dict[str, Any]] = Field(default=[], description="Guardrail, route, agent, tool and final-answer events.")
    guardrails: Dict[str, Any] = {}
    judgement: Optional[Dict[str, Any]] = Field(
        default=None, description="LLM-as-Judge result when requested: verdict, overall, scores, reasons, unsupported claims."
    )
    elapsed_ms: int


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: List[HistoryMessage]


# ============================================================================
# ML
# ============================================================================
class FlowRequest(FlowArgs):
    model_config = ConfigDict(extra="ignore", json_schema_extra={"examples": FLOW_EXAMPLES})


class BatchRequest(BaseModel):
    flows: List[FlowRequest] = Field(default_factory=list, max_length=MAX_BATCH_FLOWS)
    incident_ids: List[str] = Field(default_factory=list, max_length=MAX_BATCH_FLOWS)
    top_n: int = Field(default=10, ge=1, le=MAX_BATCH_FLOWS, description="How many of the most urgent flows to return.")

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"incident_ids": ["INC-000001", "INC-000035", "INC-002545", "INC-000005"], "top_n": 4},
    ]})


# ============================================================================
# Search
# ============================================================================
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000, description="Description of the traffic or incident.")
    attack_type: Optional[str] = Field(default=None, description="Only incidents of this attack type.")
    dst_port: Optional[int] = Field(default=None, ge=0, le=65535, description="Only incidents on this port.")
    top_k: int = Field(default=5, ge=1, le=20)

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"query": "SSH brute force with many short connections", "dst_port": 22},
        {"query": "flows with more than 10000 packets per second", "attack_type": "DDoS", "top_k": 10},
    ]})


# ============================================================================
# Judge
# ============================================================================
class JudgeRequest(BaseModel):
    """Either a thread_id (grade that conversation's latest answer) or a question and answer."""

    thread_id: Optional[str] = Field(default=None, pattern=THREAD_ID_PATTERN)
    question: Optional[str] = Field(default=None, max_length=MAX_INPUT_CHARS)
    answer: Optional[str] = Field(default=None, max_length=20000)
    route: Optional[str] = Field(default=None, description="Route of the answer (actionability is scored for recommend / investigate).")
    findings: Dict[str, str] = Field(default_factory=dict, description="Agent findings the answer was written from.")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls: agent, tool, args, result.")

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"thread_id": "3f2a9c..."},
        {
            "question": "What attack is INC-004393?",
            "answer": "INC-004393 is Infiltration (90% confidence), priority P1.",
            "route": "classify",
            "evidence": [{"agent": "classification", "tool": "classify_attack_type",
                          "args": {"incident_id": "INC-004393"},
                          "result": {"status": "ok", "attack_type": {"prediction": "Infiltration", "confidence": 0.9}}}],
        },
    ]})


class JudgeResponse(BaseModel):
    verdict: Literal["pass", "needs_review", "fail", "error"]
    overall: Optional[float] = None
    scores: Dict[str, int] = {}
    reasons: Dict[str, str] = {}
    applicable: List[str] = []
    unsupported_claims: List[str] = []
    error: Optional[str] = None
    judge_model: Optional[str] = None


# ============================================================================
# Feedback and health
# ============================================================================
class FeedbackRequest(BaseModel):
    thread_id: str = Field(pattern=THREAD_ID_PATTERN)
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(default=None, max_length=2000)
    corrected_label: Optional[str] = Field(default=None, description="Correct attack type, if the prediction was wrong.")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ml_models_loaded: bool
    attack_types: int
    assistant_ready: bool
    assistant_error: Optional[str] = None
    llm_model: Optional[str] = None
    search_index: Literal["not_loaded", "loading", "ready", "error"]
    search_index_error: Optional[str] = None
    versions: Dict[str, str]
