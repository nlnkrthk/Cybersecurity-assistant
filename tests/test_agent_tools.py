import json

import pytest

from agents.tools import ALL_TOOLS, TOOLS_BY_AGENT
from config import CLEANED_INCIDENTS_FILE

TOOLS = {t.name: t for t in ALL_TOOLS}
KEY_METRICS = {
    "flow_duration": "2 s",
    "total_fwd_packets": 10,
    "total_bwd_packets": 8,
    "total_len_fwd": "1.2 KB",
    "total_len_bwd": "9.6 KB",
}
SEARCH_INDEX_READY = (CLEANED_INCIDENTS_FILE.parent / "bm25_index.pkl").exists() and (
    CLEANED_INCIDENTS_FILE.parent / "chroma"
).exists()


def invoke(name, args):
    result = TOOLS[name].invoke(args)
    json.dumps(result)  # tool output must be JSON-serializable for the LLM
    return result


# ---------------------------------------------------------------- registry
def test_registry():
    assert set(TOOLS) == {
        "classify_attack_type", "assess_intrusion_risk", "explain_flow", "prioritize_flow",
        "prioritize_flows", "get_attack_profile", "list_attack_types", "lookup_incident",
        "search_similar_incidents",
    }
    for agent, tools in TOOLS_BY_AGENT.items():
        assert tools, agent
        assert all(t.name in TOOLS for t in tools)


def test_every_tool_has_description_and_schema():
    for tool in ALL_TOOLS:
        assert len(tool.description) > 40, tool.name
        tool.tool_call_schema.model_json_schema()


def test_flow_tools_expose_all_metrics_and_incident_id():
    props = TOOLS["classify_attack_type"].tool_call_schema.model_json_schema()["properties"]
    assert "incident_id" in props and "flow_duration" in props
    assert len(props) == 25


# ---------------------------------------------------------------- ML tools
def test_classify_by_incident_id():
    result = invoke("classify_attack_type", {"incident_id": "INC-004393"})
    assert result["status"] == "ok"
    assert result["incident"]["historical_label"] == "Infiltration"
    assert result["attack_type"]["prediction"] == "Infiltration"


def test_classify_by_metrics_reports_input_quality():
    result = invoke("classify_attack_type", KEY_METRICS)
    assert result["status"] == "ok"
    assert result["input"]["input_quality"] == "reduced"
    assert result["input"]["known_metrics"] == "12/24"
    assert result["input"]["filled_with_normal_values"]


def test_assess_intrusion_risk_has_risk_and_anomaly_only():
    result = invoke("assess_intrusion_risk", KEY_METRICS)
    assert {"risk", "anomaly"} <= set(result)
    assert "attack_type" not in result


def test_explain_flow_includes_priority_and_reasons():
    result = invoke("explain_flow", {"incident_id": "INC-000035"})
    assert result["priority"]["priority"] == "P2"
    assert result["explanation"]["attack_type_reasons"]
    assert result["explanation"]["summary"]


def test_follow_up_result_is_compact():
    result = invoke("prioritize_flow", {"flow_duration": 5})
    assert result["status"] == "needs_follow_up"
    assert result["follow_up_question"]
    assert "filled_with_normal_values" not in result["input"]


def test_prioritize_flows_ranks_mixed_input():
    result = invoke("prioritize_flows", {
        "incident_ids": ["INC-000001", "INC-002545", "INC-000035"],
        "flows": [KEY_METRICS, {"flow_duration": 5}],
        "top_n": 10,
    })
    assert result["status"] == "ok"
    assert result["total_flows"] == 5
    assert result["top"][0]["incident_id"] == "INC-002545"  # Heartbleed first
    assert result["priority_counts"].get("needs_follow_up") == 1
    assert "follow_up_question" in result["top"][-1]


def test_prioritize_flows_top_n_and_empty():
    result = invoke("prioritize_flows", {"incident_ids": ["INC-000001", "INC-000002", "INC-000003"], "top_n": 1})
    assert len(result["top"]) == 1
    assert invoke("prioritize_flows", {})["status"] == "error"


@pytest.mark.parametrize("name, expected", [("portscan", "PortScan"), ("DDOS", "DDoS"), ("ssh", "SSH-Patator"), ("sql", "Web Attack - Sql Injection")])
def test_attack_profile_name_matching(name, expected):
    result = invoke("get_attack_profile", {"attack_type": name})
    assert result["status"] == "ok"
    assert result["attack_type"] == expected
    assert result["top_features"]


def test_attack_profile_errors():
    assert "Matches" in invoke("get_attack_profile", {"attack_type": "web attack"})["error"]
    assert invoke("get_attack_profile", {"attack_type": "BENIGN"})["status"] == "error"


def test_list_attack_types():
    types = invoke("list_attack_types", {})["attack_types"]
    assert len(types) == 15
    assert {"attack_type": "Heartbleed", "severity": "Critical"} in types


def test_lookup_incident():
    result = invoke("lookup_incident", {"incident_id": "42"})
    assert result["incident_id"] == "INC-000042"
    assert result["historical_label"] == "BENIGN"
    assert len(result["metrics"]) == 24


@pytest.mark.parametrize("args, message", [
    ({}, "Provide at least some flow metrics"),
    ({"flow_duration": -1, "total_fwd_packets": 1}, "must be >= 0"),
    ({"flow_duration": "abc"}, "could not read a number"),
    ({"incident_id": "INC-999999999"}, "not found"),
])
def test_bad_input_returns_error_instead_of_raising(args, message):
    result = invoke("classify_attack_type", args)
    assert result["status"] == "error"
    assert message in result["error"]


# ---------------------------------------------------------------- search tool
@pytest.mark.skipif(not SEARCH_INDEX_READY, reason="search indexes not built")
def test_search_with_filters():
    result = invoke("search_similar_incidents", {
        "query": "high packet rate flooding", "attack_type": "ddos", "dst_port": 80, "top_k": 3,
    })
    assert result["status"] == "ok"
    assert result["filters_applied"] == {"label": "DDoS", "dst_port": 80}
    assert 1 <= len(result["incidents"]) <= 3
    for incident in result["incidents"]:
        assert incident["attack_type"] == "DDoS" and incident["dst_port"] == 80
        assert incident["incident_id"].startswith("INC-")


@pytest.mark.skipif(not SEARCH_INDEX_READY, reason="search indexes not built")
def test_search_numeric_condition_and_errors():
    result = invoke("search_similar_incidents", {"query": "flows with more than 10000 packets per second", "top_k": 2})
    assert result["numeric_conditions"]
    assert invoke("search_similar_incidents", {"query": "  "})["status"] == "error"
    assert invoke("search_similar_incidents", {"query": "x", "attack_type": "martians"})["status"] == "error"
