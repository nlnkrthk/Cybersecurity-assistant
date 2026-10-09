"""
Tools the LangGraph agents can call.

    ML tools      classify_attack_type, assess_intrusion_risk, explain_flow, prioritize_flow,
                  prioritize_flows, get_attack_profile, list_attack_types, lookup_incident
    Search tools  search_similar_incidents

Tools return dicts and never raise on bad input ({"status": "error", ...}).
TOOLS_BY_AGENT lists which tools each agent gets.
"""
from agents.tools.ml_tools import (
    ML_TOOLS,
    assess_intrusion_risk_tool,
    classify_attack_type_tool,
    explain_flow_tool,
    get_attack_profile_tool,
    list_attack_types_tool,
    lookup_incident_tool,
    prioritize_flows_tool,
    prioritize_single_flow_tool,
)
from agents.tools.search_tools import SEARCH_TOOLS, search_similar_incidents_tool

ALL_TOOLS = ML_TOOLS + SEARCH_TOOLS

TOOLS_BY_AGENT = {
    "classification": [
        classify_attack_type_tool,
        explain_flow_tool,
        get_attack_profile_tool,
        list_attack_types_tool,
        lookup_incident_tool,
    ],
    "risk_priority": [
        assess_intrusion_risk_tool,
        prioritize_single_flow_tool,
        prioritize_flows_tool,
        lookup_incident_tool,
    ],
    "incident_search": [
        search_similar_incidents_tool,
        get_attack_profile_tool,
        lookup_incident_tool,
    ],
    "response": [
        search_similar_incidents_tool,
        get_attack_profile_tool,
    ],
}

__all__ = ["ALL_TOOLS", "ML_TOOLS", "SEARCH_TOOLS", "TOOLS_BY_AGENT"]
