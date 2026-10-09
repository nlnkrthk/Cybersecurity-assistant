import time

import pytest

from services.feature_input import FEATURE_NAMES, normalize_flow
from services.ml_service import get_ml_service

PARTIAL = {
    "flow_duration": "46 us",
    "total_fwd_packets": 1,
    "total_bwd_packets": 1,
    "total_len_fwd": 2,
    "total_len_bwd": 6,
}


def _features(row):
    return {name: float(row[name]) for name in FEATURE_NAMES}


def test_service_is_a_singleton(ml_service):
    assert get_ml_service() is ml_service


def test_models_share_feature_order_and_classes(ml_service):
    assert ml_service.features == list(FEATURE_NAMES)
    assert len(ml_service.list_attack_types()) == 15
    assert "BENIGN" in ml_service.list_attack_types()


def test_one_real_flow_per_class_is_classified(ml_service, one_per_class):
    correct = sum(
        ml_service.classify(_features(row))["attack_type"]["prediction"] == row["label"]
        for _, row in one_per_class.iterrows()
    )
    assert correct >= 13, f"only {correct}/15 classes predicted correctly"


def test_predict_combines_the_individual_calls(ml_service, full_flow):
    combined = ml_service.predict(full_flow)
    assert combined["status"] == "ok"
    assert combined["attack_type"] == ml_service.classify(full_flow)["attack_type"]
    assert combined["risk"] == ml_service.predict_risk(full_flow)["risk"]
    assert combined["anomaly"] == ml_service.detect_anomaly(full_flow)["anomaly"]
    assert combined["overall_confidence"] in {"high", "medium", "low"}


def test_single_calls_only_run_their_model(ml_service, full_flow):
    assert set(ml_service.classify(full_flow)) >= {"attack_type"} and "risk" not in ml_service.classify(full_flow)
    assert "attack_type" not in ml_service.predict_risk(full_flow)
    assert "risk" not in ml_service.detect_anomaly(full_flow)


def test_batch_matches_single_predictions(ml_service, flows_df):
    rows = [_features(r) for _, r in flows_df.sample(20, random_state=1).iterrows()]
    batch = ml_service.predict_batch(rows)
    for flow, result in zip(rows, batch):
        assert result == ml_service.predict(flow)


def test_batch_handles_follow_up_items(ml_service, full_flow):
    results = ml_service.predict_batch([full_flow, {"flow_duration": 5}])
    assert results[0]["status"] == "ok"
    assert results[1]["status"] == "needs_follow_up"


def test_risk_and_anomaly_levels_follow_thresholds(ml_service, flows_df):
    for _, row in flows_df.sample(30, random_state=2).iterrows():
        result = ml_service.predict(_features(row))
        risk, anomaly = result["risk"], result["anomaly"]
        assert risk["is_intrusion"] == (risk["probability"] >= ml_service.decision_threshold)
        assert 0 <= anomaly["benign_percentile"] <= 100
        if anomaly["score"] >= ml_service.anomaly_thresholds["high"]:
            assert anomaly["level"] == "High"


def test_needs_follow_up_runs_no_model(ml_service):
    result = ml_service.predict({"flow_duration": 5})
    assert result["status"] == "needs_follow_up"
    assert "attack_type" not in result and "risk" not in result
    assert result["follow_up_question"]
    assert result["missing_key_features"]


def test_partial_input_is_flagged(ml_service):
    result = ml_service.predict(PARTIAL)
    assert result["status"] == "ok"
    assert result["input"]["input_quality"] == "reduced"
    assert result["input"]["filled"]
    assert result["overall_confidence"] != "high"


def test_explain_has_reasons_and_summary(ml_service, full_flow):
    result = ml_service.explain(full_flow)
    explanation = result["explanation"]
    assert explanation["attack_type_reasons"]
    assert explanation["risk_reasons"]
    assert result["attack_type"]["prediction"] in explanation["summary"]
    for reason in explanation["attack_type_reasons"]:
        assert reason["shap"] > 0
        assert reason["feature"] in FEATURE_NAMES


def test_explain_never_uses_filled_features_as_reasons(ml_service):
    result = ml_service.explain(PARTIAL)
    filled = set(result["input"]["filled"])
    explanation = result["explanation"]
    used = {r["feature"] for r in explanation["attack_type_reasons"] + explanation["risk_reasons"]}
    used |= {f["feature"] for f in result["anomaly"]["most_unusual_features"]}
    assert not used & filled
    assert "filled with normal-traffic values" in explanation["summary"]


def test_explain_follow_up_passthrough(ml_service):
    assert ml_service.explain({"flow_duration": 5})["status"] == "needs_follow_up"


def test_attack_profiles(ml_service):
    for attack in ml_service.list_attack_types():
        if attack == "BENIGN":
            continue
        profile = ml_service.attack_profile(attack)
        assert profile["top_features"] and profile["summary"]
    with pytest.raises(KeyError):
        ml_service.attack_profile("Alien Invasion")


def test_describe_never_claims_100_percent(ml_service):
    text = ml_service.describe("bwd_packet_length_mean", 1e9)
    assert "99.9%" in text and "100%" not in text


def test_speed(ml_service, full_flow):
    ml_service.explain(full_flow)  # warm-up (SHAP explainers)
    start = time.perf_counter()
    ml_service.predict(full_flow)
    predict_s = time.perf_counter() - start
    start = time.perf_counter()
    ml_service.explain(full_flow)
    explain_s = time.perf_counter() - start
    assert predict_s < 0.5, predict_s
    assert explain_s < 1.5, explain_s


def test_normalized_flow_input_is_accepted(ml_service, full_flow):
    assert ml_service.predict(normalize_flow(full_flow)) == ml_service.predict(full_flow)
