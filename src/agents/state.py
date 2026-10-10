"""
Shared state of the assistant graph and the route -> agents plan.
"""
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

CLASSIFICATION = "classification"
RISK_PRIORITY = "risk_priority"
INCIDENT_SEARCH = "incident_search"
RESPONSE = "response"

# Which agents run, in order, for each route chosen by the supervisor.
ROUTE_PLANS: Dict[str, List[str]] = {
    "classify": [CLASSIFICATION],
    "risk": [RISK_PRIORITY],
    "search": [INCIDENT_SEARCH],
    "recommend": [RESPONSE],
    "investigate": [CLASSIFICATION, RISK_PRIORITY, INCIDENT_SEARCH, RESPONSE],
    "general": [],
    "out_of_scope": [],
}
ROUTES = tuple(ROUTE_PLANS)


class AssistantState(TypedDict, total=False):
    # Conversation shown to the user: user messages and final answers / follow-up questions.
    messages: Annotated[List[AnyMessage], add_messages]

    # Set by the supervisor for the current turn.
    route: str
    route_reason: str
    plan: List[str]

    # Filled by the agents during the current turn.
    completed: List[str]
    findings: Dict[str, str]            # agent name -> findings text for the final answer
    evidence: List[Dict[str, Any]]      # every tool call: agent, tool, args, result
    trace: List[Dict[str, Any]]         # route, agent, tool and final events (see agents.trace)

    # Flow analysed this turn, and the one from earlier turns (for "what about its risk?").
    flow_args: Optional[Dict[str, Any]]
    previous_flow_args: Optional[Dict[str, Any]]

    # Carried across turns.
    follow_up: Optional[str]              # question asked to the user when key metrics are missing
    pending_follow_up: Optional[Dict[str, Any]]  # previous turn's question and route, if any

    final_answer: str

    # Guardrail results for the current turn: {"input": {...}, "output": {...}} (see agents.guardrails).
    guardrails: Dict[str, Any]

    # LLM-as-Judge for the current turn (see agents.judge): requested by the caller, and its result.
    judge_requested: bool
    judgement: Optional[Dict[str, Any]]
