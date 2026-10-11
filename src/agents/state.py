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

# Agents that can run at the same time. Classification and Risk & Priority only need the flow;
# Incident Search and Response use their findings (attack type for the search filter, priority for
# the plan) and do not depend on each other. Routes not listed run one agent per stage.
ROUTE_STAGES: Dict[str, List[List[str]]] = {
    "investigate": [[CLASSIFICATION, RISK_PRIORITY], [INCIDENT_SEARCH, RESPONSE]],
}


def plan_stages(route: Optional[str], plan: List[str], parallel: bool = True) -> List[List[str]]:
    """The plan as stages of agents that run together (one agent per stage when not parallel)."""
    stages = ROUTE_STAGES.get(route) if parallel else None
    if stages is None or sorted(a for stage in stages for a in stage) != sorted(plan):
        return [[agent] for agent in plan]
    return [list(stage) for stage in stages]


def next_stage(state: "AssistantState", parallel: bool = True) -> List[str]:
    """The agents to run next: the unfinished agents of the first unfinished stage."""
    done = set(state.get("completed", []))
    for stage in plan_stages(state.get("route"), state.get("plan", []), parallel):
        remaining = [agent for agent in stage if agent not in done]
        if remaining:
            return remaining
    return []


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

    # Unique ID of the current turn (feedback refers to it).
    turn_id: str

    # LLM-as-Judge for the current turn (see agents.judge): requested by the caller, and its result.
    judge_requested: bool
    judgement: Optional[Dict[str, Any]]
