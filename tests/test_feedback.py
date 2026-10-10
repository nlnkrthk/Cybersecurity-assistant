"""
Feedback loop tests: store, turn snapshots, recording, analyst feedback in tools, API,
correction export and retraining (synthetic data for the training part).
"""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import feedback as feedback_pkg
from agents.graph import Assistant
from feedback.retraining import FEATURES, collect_corrections, new_classifier, retrain_with_corrections
from feedback.service import FeedbackError, record_feedback
from feedback.snapshot import build_snapshot, has_flow
from feedback.store import FeedbackStore
from test_agents import FakeLLM, route, text, tool_call


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(tmp_path / "feedback.db")
    feedback_pkg.set_feedback_store(s)
    yield s
    feedback_pkg.set_feedback_store(None)
    s.close()


def snap(**kw):
    base = {"question": "q", "answer": "a", "route": "classify", "agents_run": ["classification"],
            "predicted_label": "Infiltration", "prediction_confidence": 0.9, "incident_id": "INC-004393",
            "historical_label": "Infiltration", "flow_args": {"incident_id": "INC-004393"},
            "input_quality": "full", "judge_verdict": None, "judge_overall": None, "llm_model": "test"}
    return {**base, **kw}


# ---------------------------------------------------------------- store
def test_store_add_update_and_list(store):
    row = store.add("t1", "turn1", "up", snap(), comment="  great  ")
    assert row["rating"] == "up" and row["comment"] == "great"
    assert row["agents_run"] == ["classification"] and row["flow_args"] == {"incident_id": "INC-004393"}

    updated = store.add("t1", "turn1", "down", snap(), corrected_label="Bot")
    assert updated["id"] == row["id"] and updated["rating"] == "down" and store.count() == 1

    store.add("t2", "turn2", "up", snap(route="risk", predicted_label="DDoS"))
    assert [r["thread_id"] for r in store.list()] == ["t2", "t1"]
    assert len(store.list(rating="down")) == 1
    assert len(store.list(route="risk")) == 1
    assert len(store.list(label="Bot")) == 1
    assert len(store.list(corrected_only=True)) == 1


def test_store_rejects_invalid_rating(store):
    with pytest.raises(ValueError):
        store.add("t", "x", "meh", snap())


def test_corrections_for_incident(store):
    store.add("a", "1", "down", snap(), corrected_label="Bot")
    store.add("b", "1", "down", snap(), corrected_label="Bot")
    store.add("c", "1", "down", snap(), corrected_label="PortScan")
    store.add("d", "1", "up", snap())
    assert store.corrections_for("INC-004393") == {"Bot": 2, "PortScan": 1}
    assert store.corrections_for("INC-000001") == {}


def test_stats(store):
    store.add("a", "1", "up", snap(judge_verdict="pass"))
    store.add("b", "1", "up", snap(judge_verdict="fail"))
    store.add("c", "1", "down", snap(judge_verdict="needs_review"), comment="wrong type",
              corrected_label="Bot")
    store.add("d", "1", "down", snap(route="risk", judge_verdict="pass"))
    store.add("e", "1", "up", snap(route="risk"))
    stats = store.stats()
    assert stats["total"] == 5 and stats["up"] == 3 and stats["approval_rate"] == 0.6
    assert stats["by_route"]["classify"] == {"n": 3, "up": 2, "approval_rate": 0.667}
    agreement = stats["judge_agreement"]
    assert agreement["n"] == 4
    assert (agreement["pass_up"], agreement["pass_down"], agreement["not_pass_up"], agreement["not_pass_down"]) == (1, 1, 1, 1)
    assert agreement["agreement_rate"] == 0.5
    assert stats["corrections"]["top_pairs"] == [{"predicted": "Infiltration", "corrected": "Bot", "count": 1}]
    assert stats["recent_comments"][0]["comment"] == "wrong type"


def test_empty_stats(store):
    stats = store.stats()
    assert stats["total"] == 0 and stats["approval_rate"] is None and stats["judge_agreement"]["agreement_rate"] is None


# ---------------------------------------------------------------- snapshot
def test_snapshot_from_explain_evidence():
    state = {
        "final_answer": "It is Infiltration.", "route": "classify", "completed": ["classification"],
        "flow_args": {"incident_id": "INC-004393"},
        "judgement": {"verdict": "pass", "overall": 4.8},
        "evidence": [{"tool": "lookup_incident", "result": {"status": "ok", "incident_id": "INC-004393"}},
                     {"tool": "explain_flow", "result": {
                         "status": "ok", "incident": {"incident_id": "INC-004393", "historical_label": "Infiltration"},
                         "attack_type": {"prediction": "Infiltration", "confidence": 0.9},
                         "input": {"input_quality": "full"}}}],
    }
    s = build_snapshot(state, "What is INC-004393?", "nvidia:gpt-oss")
    assert (s["predicted_label"], s["prediction_confidence"], s["incident_id"], s["historical_label"]) == \
        ("Infiltration", 0.9, "INC-004393", "Infiltration")
    assert s["judge_verdict"] == "pass" and s["judge_overall"] == 4.8 and s["llm_model"] == "nvidia:gpt-oss"
    assert has_flow(s)


def _copied_metrics_state(args):
    lookup = {"status": "ok", "incident_id": "INC-000035", "historical_label": "SSH-Patator",
              "metrics": {"Flow duration": 13903754.0, "Forward packets": 21.0, "Backward packets": 32.0}}
    return {"final_answer": "SSH-Patator.", "route": "investigate", "evidence": [
        {"tool": "lookup_incident", "args": {"incident_id": "INC-000035"}, "result": lookup},
        {"tool": "classify_attack_type", "args": args, "result": {
            "status": "ok", "attack_type": {"prediction": "SSH-Patator", "confidence": 1.0},
            "input": {"input_quality": "full"}}},
    ]}


def test_snapshot_links_incident_when_agent_passes_its_metrics():
    # Seen live: the agent looked the incident up, then classified it with the copied metrics.
    s = build_snapshot(_copied_metrics_state(
        {"flow_duration": 13903754.0, "total_fwd_packets": 21.0, "total_bwd_packets": 32.0}), "investigate")
    assert (s["incident_id"], s["historical_label"], s["predicted_label"]) == ("INC-000035", "SSH-Patator", "SSH-Patator")


def test_snapshot_does_not_link_incident_for_different_metrics():
    for args in ({"flow_duration": 5.0, "total_fwd_packets": 21.0}, {"flow_duration": "2 s"}):
        s = build_snapshot(_copied_metrics_state(args), "investigate")
        assert s["incident_id"] is None and s["predicted_label"] == "SSH-Patator"


def test_snapshot_without_flow():
    s = build_snapshot({"final_answer": "DoS vs DDoS ...", "route": "general"}, "difference?")
    assert s["predicted_label"] is None and not has_flow(s)


# ---------------------------------------------------------------- recording via the assistant
def classify_assistant(extra_routes=(), extra_finals=()):
    llm = FakeLLM(
        routes=[route("classify"), *extra_routes],
        agent_replies=[tool_call("classify_attack_type", {"incident_id": "INC-004393"}), text("Infiltration 90%")],
        final_replies=["INC-004393 is Infiltration.", *extra_finals],
    )
    return Assistant(llm=llm)


def test_record_feedback_snapshots_the_real_turn(store):
    assistant = classify_assistant()
    result = assistant.ask("What attack is INC-004393?")
    assert result["turn_id"]
    row = record_feedback(assistant, result["thread_id"], "up", comment="spot on")
    assert row["turn_id"] == result["turn_id"]
    assert row["question"] == "What attack is INC-004393?" and row["answer"] == "INC-004393 is Infiltration."
    assert row["predicted_label"] == "Infiltration" and row["incident_id"] == "INC-004393"
    assert row["route"] == "classify"


def test_corrected_label_is_normalized_and_validated(store):
    assistant = classify_assistant()
    thread = assistant.ask("What attack is INC-004393?")["thread_id"]
    row = record_feedback(assistant, thread, "down", corrected_label="web attack - xss")
    assert row["corrected_label"] == "Web Attack - XSS"
    with pytest.raises(FeedbackError, match="Unknown or ambiguous"):
        record_feedback(assistant, thread, "down", corrected_label="martian attack")


def test_correction_needs_a_flow(store):
    assistant = Assistant(llm=FakeLLM(routes=[route("general")], final_replies=["A DDoS is ..."]))
    thread = assistant.ask("What is a DDoS?")["thread_id"]
    record_feedback(assistant, thread, "up")  # rating alone is fine
    with pytest.raises(FeedbackError, match="no flow or incident"):
        record_feedback(assistant, thread, "down", corrected_label="DDoS")


def test_feedback_for_an_earlier_turn(store):
    assistant = classify_assistant(extra_routes=[route("general")], extra_finals=["hello"])
    thread = assistant.new_thread_id()
    first = assistant.ask("What attack is INC-004393?", thread)
    assistant.ask("thanks", thread)
    row = record_feedback(assistant, thread, "down", turn_id=first["turn_id"], corrected_label="Bot")
    assert row["question"] == "What attack is INC-004393?" and row["corrected_label"] == "Bot"
    with pytest.raises(KeyError):
        record_feedback(assistant, thread, "up", turn_id="no-such-turn")
    with pytest.raises(KeyError):
        record_feedback(assistant, "no-such-thread", "up")


def test_corrections_reach_the_tools(store):
    from agents.tools.ml_tools import classify_attack_type, lookup_incident

    assert "analyst_feedback" not in lookup_incident(incident_id="INC-004393")
    assistant = classify_assistant()
    thread = assistant.ask("What attack is INC-004393?")["thread_id"]
    record_feedback(assistant, thread, "down", corrected_label="Bot")

    looked_up = lookup_incident(incident_id="INC-004393")["analyst_feedback"]
    assert looked_up["corrected_labels"] == {"Bot": 1} and "human feedback" in looked_up["note"]
    classified = classify_attack_type(incident_id="INC-004393")
    assert classified["incident"]["analyst_feedback"]["corrected_labels"] == {"Bot": 1}
    assert classified["attack_type"]["prediction"] == "Infiltration"  # the model prediction is unchanged


def test_no_database_means_no_feedback(tmp_path, monkeypatch):
    feedback_pkg.set_feedback_store(None)
    monkeypatch.setenv("FEEDBACK_DB_PATH", str(tmp_path / "missing" / "feedback.db"))
    assert feedback_pkg.corrections_for_incident("INC-004393") == {}
    assert not (tmp_path / "missing").exists()  # a lookup never creates the database


# ---------------------------------------------------------------- API
def make_client(assistant):
    from api.main import create_app

    return TestClient(create_app(assistant=assistant, load_search_index=False))


def test_api_feedback_flow(store):
    with make_client(classify_assistant()) as c:
        chat = c.post("/chat", json={"message": "What attack is INC-004393?"}).json()
        assert chat["turn_id"]
        r = c.post("/feedback", json={"thread_id": chat["thread_id"], "turn_id": chat["turn_id"],
                                       "rating": "down", "comment": "it's a bot", "corrected_label": "bot"})
        assert r.status_code == 201 and r.json()["corrected_label"] == "Bot"

        assert c.get("/feedback").json()[0]["comment"] == "it's a bot"
        assert c.get("/feedback", params={"rating": "up"}).json() == []
        stats = c.get("/feedback/stats").json()
        assert stats["total"] == 1 and stats["corrections"]["total"] == 1

        export = c.get("/feedback/export.csv")
        assert export.status_code == 200 and export.headers["content-type"].startswith("text/csv")
        lines = export.text.strip().splitlines()
        assert lines[0].startswith("flow_duration") and len(lines) == 2 and ",Bot," in lines[1]


def test_api_feedback_errors(store):
    with make_client(classify_assistant()) as c:
        thread = c.post("/chat", json={"message": "What attack is INC-004393?"}).json()["thread_id"]
        assert c.post("/feedback", json={"thread_id": "unknown", "rating": "up"}).status_code == 404
        assert c.post("/feedback", json={"thread_id": thread, "rating": "meh"}).status_code == 422
        assert c.post("/feedback", json={"thread_id": thread, "rating": "down",
                                          "corrected_label": "martian"}).status_code == 422


def test_health_reports_classifier_version():
    with make_client(classify_assistant()) as c:
        assert c.get("/health").json()["classifier_version"] == "original"


# ---------------------------------------------------------------- corrections export
def test_collect_corrections_majority_ties_and_skips(store):
    store.add("a", "1", "down", snap(incident_id="INC-004393"), corrected_label="Bot")
    store.add("b", "1", "down", snap(incident_id="INC-004393"), corrected_label="Bot")
    store.add("c", "1", "down", snap(incident_id="INC-004393"), corrected_label="DDoS")
    store.add("d", "1", "down", snap(incident_id="INC-000035"), corrected_label="Bot")
    store.add("e", "1", "down", snap(incident_id="INC-000035"), corrected_label="DDoS")      # tie
    store.add("f", "1", "down", snap(incident_id=None, flow_args={"flow_duration": "2 s", "total_fwd_packets": 3,
                                                                  "total_bwd_packets": 2, "total_len_fwd": 100,
                                                                  "total_len_bwd": 50}), corrected_label="Bot")
    frame, skipped = collect_corrections(store)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["label"] == "Bot" and row["source"] == "INC-004393" and row["votes"] == 2 and row["total_votes"] == 3
    assert list(frame.columns[:24]) == FEATURES
    reasons = " ".join(str(s["reason"]) for s in skipped)
    assert "tied corrections" in reasons and "input quality" in reasons


# ---------------------------------------------------------------- retraining (synthetic data)
def synthetic_flows(n_per_class=120, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i, label in enumerate(["BENIGN", "DDoS", "PortScan"]):
        X = rng.normal(loc=i * 3.0, scale=1.0, size=(n_per_class, len(FEATURES)))
        frame = pd.DataFrame(X, columns=FEATURES)
        frame["label"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_retrain_with_corrections_reports_and_guards():
    flows = synthetic_flows()
    current = new_classifier(20).fit(flows[FEATURES], flows["label"])
    corrections = flows.sample(5, random_state=1)[FEATURES].copy()
    corrections["label"] = "PortScan"

    model, report = retrain_with_corrections(flows, corrections, current, n_estimators=20)
    assert report["corrections_used"] == 5
    assert report["training_rows_relabelled"] + report["corrections_from_test_split"] == 5
    assert report["n_test"] == round(len(flows) * 0.2) - report["corrections_from_test_split"]
    assert set(report["current_test_metrics"]) == {"accuracy", "macro_f1", "weighted_f1"}
    assert report["promote"] == (report["new_test_metrics"]["macro_f1"] >= report["current_test_metrics"]["macro_f1"])
    assert report["new_on_corrections"] >= report["current_on_corrections"]
    assert set(model.classes_) == {"BENIGN", "DDoS", "PortScan"}


def test_retrain_blocks_promotion_when_quality_drops():
    flows = synthetic_flows()
    current = new_classifier(20).fit(flows[FEATURES], flows["label"])
    # Many wrong "corrections": relabel most BENIGN rows as DDoS.
    corrections = flows[flows["label"] == "BENIGN"].iloc[:90][FEATURES].copy()
    corrections["label"] = "DDoS"
    _, report = retrain_with_corrections(flows, corrections, current, n_estimators=20)
    assert report["macro_f1_change"] < 0 and report["promote"] is False


def test_active_classifier_pointer(tmp_path, monkeypatch):
    import services.ml_service as ml_service

    pointer = tmp_path / "active.json"
    monkeypatch.setattr(ml_service, "ACTIVE_CLASSIFIER_POINTER", pointer)
    assert ml_service.active_classifier_files()[2] == "original"

    (tmp_path / "m.pkl").write_bytes(b"x")
    (tmp_path / "m_meta.json").write_text("{}")
    pointer.write_text(json.dumps({"version": "v1", "model_file": "m.pkl", "meta_file": "m_meta.json"}))
    model_file, meta_file, version = ml_service.active_classifier_files()
    assert version == "v1" and model_file.name == "m.pkl"

    pointer.write_text(json.dumps({"version": "v2", "model_file": "gone.pkl", "meta_file": "m_meta.json"}))
    assert ml_service.active_classifier_files()[2] == "original"  # missing files fall back safely


def test_activate_and_list_versions(tmp_path, monkeypatch):
    import feedback.retraining as retraining

    monkeypatch.setattr(retraining, "ML_MODEL_VERSIONS_DIR", tmp_path)
    monkeypatch.setattr(retraining, "ACTIVE_CLASSIFIER_POINTER", tmp_path / "active.json")
    (tmp_path / "attack_classifier_v1.pkl").write_bytes(b"x")
    (tmp_path / "attack_classifier_v1_meta.json").write_text(json.dumps({
        "version": "v1", "trained_at": "2026-10-10T12:00:00", "test_metrics": {"macro_f1": 0.84},
        "feedback": {"corrections_used": 3}}))

    assert [v["version"] for v in retraining.list_versions()] == ["original", "v1"]
    retraining.activate("v1")
    assert next(v for v in retraining.list_versions() if v["version"] == "v1")["active"]
    retraining.activate("original")
    assert not (tmp_path / "active.json").exists()
    with pytest.raises(FileNotFoundError):
        retraining.activate("v9")
