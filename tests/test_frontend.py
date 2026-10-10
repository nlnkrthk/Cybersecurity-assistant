"""
Front-end tests: the API client against the real FastAPI app (TestClient, scripted
FakeLLM), the formatting helpers, and Streamlit smoke tests of every page.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
sys.path.insert(0, str(FRONTEND))

import formatting as fmt  # noqa: E402
from api_client import ApiClient, ApiError, error_detail, parse_sse  # noqa: E402

from agents.graph import Assistant  # noqa: E402
from api.main import create_app  # noqa: E402
from services.feature_input import FEATURE_NAMES  # noqa: E402
from test_agents import FakeLLM, route, text, tool_call  # noqa: E402

# The client passes per-call timeouts, which Starlette's TestClient ignores with a warning.
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")


def make_api(llm=None) -> TestClient:
    return TestClient(create_app(assistant=Assistant(llm=llm or FakeLLM()), load_search_index=False))


@pytest.fixture
def api():
    with make_api() as test_client:
        yield ApiClient(url="http://testserver", client=test_client)


def classify_llm():
    return FakeLLM(
        routes=[route("classify")],
        agent_replies=[tool_call("classify_attack_type", {"incident_id": "INC-000035"}), text("SSH-Patator")],
        final_replies=["INC-000035 is SSH-Patator."],
    )


# ============================================================================
# SSE parsing and errors
# ============================================================================
def test_parse_sse_events_comments_and_multiline_data():
    lines = [": keep-alive", "event: trace", 'data: {"type": "route"}', "", "event: result",
             'data: {"a":', 'data: 1}', "", 'data: {"tail": true}']
    assert list(parse_sse(lines)) == [("trace", {"type": "route"}), ("result", {"a": 1}), ("message", {"tail": True})]


def test_error_detail_formats_validation_errors():
    import httpx

    response = httpx.Response(422, json={"detail": [{"loc": ["body", "message"], "msg": "too short"}]})
    assert error_detail(response) == "message: too short"
    assert error_detail(httpx.Response(404, json={"detail": "Incident X not found"})) == "Incident X not found"
    assert error_detail(httpx.Response(500, text="boom")) == "boom"


def test_api_error_messages():
    assert ApiError(None, "ConnectError").message.startswith("Cannot reach the API")
    assert ApiError(502, "quota").message == "The language model provider returned an error: quota"
    assert ApiError(418, "teapot").message == "API error 418: teapot"


def test_unreachable_api_raises_api_error_without_status():
    client = ApiClient(url="http://127.0.0.1:9", timeout=2)
    with pytest.raises(ApiError) as info:
        client.health()
    assert info.value.status is None
    with pytest.raises(ApiError):
        list(client.chat_stream("hello"))


# ============================================================================
# API client against the real app
# ============================================================================
def test_health_and_attack_types(api):
    assert api.health()["assistant_ready"] is True
    types = api.attack_types()
    assert len(types) == 15 and {"attack_type", "severity"} <= set(types[0])
    assert api.attack_profile("ddos")["attack_type"] == "DDoS"


def test_explain_incident_and_follow_up(api):
    result = api.explain({"incident_id": "INC-000035"})
    assert result["attack_type"]["prediction"] == "SSH-Patator"
    assert result["explanation"]["attack_type_reasons"] and result["priority"]["priority"]
    partial = api.explain({"flow_duration": "2 s", "total_fwd_packets": 10})
    assert partial["status"] == "needs_follow_up" and partial["follow_up_question"]


def test_errors_map_to_status_codes(api):
    with pytest.raises(ApiError) as info:
        api.incident("INC-999999")
    assert info.value.status == 404
    with pytest.raises(ApiError) as info:
        api.explain({"total_fwd_packets": -5, "flow_duration": 10, "total_bwd_packets": 1,
                     "total_len_fwd": 1, "total_len_bwd": 1})
    assert info.value.status == 422


def test_prioritize_batch_and_csv(api):
    batch = api.prioritize_batch(incident_ids=["INC-000035", "INC-002545"], top_n=2)
    assert batch["top"][0]["incident_id"] == "INC-002545"
    csv = b"incident_id\nINC-000005\nINC-000001\n"
    ranked = api.prioritize_csv(csv, top_n=5)
    assert ranked["total_flows"] == 2 and ranked["top"][0]["incident_id"] == "INC-000001"


def test_chat_stream_then_feedback_and_stats():
    with make_api(classify_llm()) as test_client:
        api = ApiClient(url="http://testserver", client=test_client)
        events = list(api.chat_stream("What attack is INC-000035?", judge=False))
        kinds = [kind for kind, _ in events]
        assert kinds[-1] == "result" and set(kinds[:-1]) == {"trace"}
        result = events[-1][1]
        assert result["answer"] == "INC-000035 is SSH-Patator." and result["turn_id"]
        assert fmt.has_flow_analysis(result) and fmt.predicted_label(result) == "SSH-Patator"
        assert all(fmt.trace_line(data) or data["type"] == "tool_call" for _, data in events[:-1])

        row = api.send_feedback(result["thread_id"], "down", turn_id=result["turn_id"],
                                comment="wrong", corrected_label="ftp-patator")
        assert row["corrected_label"] == "FTP-Patator" and row["predicted_label"] == "SSH-Patator"
        stats = api.feedback_stats()
        assert stats["corrections"]["top_pairs"][0] == {"predicted": "SSH-Patator", "corrected": "FTP-Patator",
                                                        "count": 1}
        assert any(r["turn_id"] == result["turn_id"] for r in api.feedback_list(corrected_only=True))
        exported = api.feedback_export_csv().decode()
        assert "FTP-Patator" in exported and "init_win_bytes_forward" in exported
        assert api.history(result["thread_id"])["messages"][-1]["content"] == result["answer"]
        api.delete_thread(result["thread_id"])
        with pytest.raises(ApiError) as info:
            api.history(result["thread_id"])
        assert info.value.status == 404


def test_chat_stream_error_event_raises():
    with make_api(FakeLLM(routes=[route("general")], final_replies=[])) as test_client:
        api = ApiClient(url="http://testserver", client=test_client)
        with pytest.raises(ApiError) as info:
            list(api.chat_stream("hello"))
    assert info.value.status == 500


def test_feedback_correction_without_flow_is_rejected():
    llm = FakeLLM(routes=[route("general")], final_replies=["Hello!"])
    with make_api(llm) as test_client:
        api = ApiClient(url="http://testserver", client=test_client)
        result = api.chat("hello")
        with pytest.raises(ApiError) as info:
            api.send_feedback(result["thread_id"], "down", corrected_label="DDoS")
    assert info.value.status == 422


# ============================================================================
# Formatting helpers
# ============================================================================
def test_flow_fields_match_model_features(api):
    assert [key for key, _, _ in fmt.FLOW_FIELDS] == list(FEATURE_NAMES)
    assert sum(len(v) for v in fmt.FIELD_GROUPS.values()) == 24
    metrics = api.incident("INC-000035")["metrics"]
    form = fmt.metrics_to_form(metrics)
    assert set(form) == set(FEATURE_NAMES)  # every display name maps back to a field
    assert form["init_win_bytes_forward"] == "29200"


def test_example_incidents_have_their_labels(api):
    for label, incident_id in fmt.EXAMPLE_INCIDENTS.items():
        assert api.incident(incident_id)["historical_label"] == label


def test_parse_flow_inputs():
    assert fmt.parse_flow_inputs({"a": " 1,200 ", "b": "2 s", "c": "", "d": "  "}) == {"a": 1200.0, "b": "2 s"}


def test_parse_incident_ids():
    valid, invalid = fmt.parse_incident_ids("inc-000001, INC-000001\nINC-000035; foo INC-12")
    assert valid == ["INC-000001", "INC-000035"] and invalid == ["foo", "INC-12"]


def test_ranking_rows_handle_incidents_flows_and_follow_ups():
    rows = fmt.ranking_rows([
        {"incident_id": "INC-000001", "priority": "P1", "score": 90},
        {"flow_index": 1, "priority": "P4", "score": 1},
        {"flow_index": 0, "priority": None, "follow_up_question": "Need bytes"},
    ])
    assert [r["flow"] for r in rows] == ["INC-000001", "row 2", "row 1"]
    assert rows[2]["priority"] == "needs data" and rows[2]["reason"] == "Need bytes"


def test_trace_lines_and_guardrail_notices():
    assert "blocked" in fmt.trace_line({"type": "guard_input", "action": "blocked", "category": "prompt_injection"})
    assert "Risk & Priority agent" in fmt.trace_line({"type": "route", "route": "risk", "reason": "r",
                                                      "plan": ["risk_priority"], "elapsed_ms": 1200})
    assert fmt.trace_line({"type": "tool_call"}) is None
    assert "passed" in fmt.trace_line({"type": "guard_output", "issues": []})
    assert "4.2/5" in fmt.trace_line({"type": "judge", "verdict": "pass", "overall": 4.2})
    result = {"guardrails": {
        "input": {"action": "redacted", "redactions": ["email"]},
        "output": {"issues": [{"check": "unverified_incident_ids", "ids": ["INC-123456"]},
                              {"check": "confidence_disclaimer", "level": "low"}]},
    }}
    inputs, outputs = fmt.guardrail_notices(result)
    assert "email" in inputs[0] and len(outputs) == 1 and "INC-123456" in outputs[0]


def test_shap_rows_sorted_by_strength():
    rows = fmt.shap_rows([{"text": "a", "shap": 0.1}, {"text": "b", "shap": -0.5}, {"text": "c", "shap": 0.3}])
    assert [r["feature"] for r in rows] == ["a", "c", "b"]


# ============================================================================
# Streamlit pages (API not running: pages must show errors, not crash)
# ============================================================================
@pytest.mark.parametrize("page", ["app.py", "views/assistant.py", "views/flow_analyzer.py",
                                  "views/incident_search.py", "views/triage_queue.py", "views/answer_quality.py"])
def test_pages_render_without_api(page, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")
    import components

    components.get_client.clear()
    components.health.clear()
    components.attack_type_names.clear()
    app = AppTest.from_file(str(FRONTEND / page), default_timeout=30)
    app.run()
    assert not app.exception, app.exception
