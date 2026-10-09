import math

import pytest

from services.feature_input import (
    FEATURE_NAMES,
    FlowFeatures,
    FlowInputError,
    benign_medians,
    follow_up_question,
    get_incident_features,
    normalize_flow,
    normalize_incident_id,
    parse_quantity,
)

KEY_ONLY = {
    "flow_duration": 2_000_000,
    "total_fwd_packets": 10,
    "total_bwd_packets": 8,
    "total_len_fwd": 1200,
    "total_len_bwd": 9600,
}


# ---------------------------------------------------------------- full input
def test_full_flow_passes_unchanged(full_flow):
    flow = normalize_flow(full_flow)
    assert flow.features == full_flow
    assert flow.provided == list(FEATURE_NAMES)
    assert flow.derived == [] and flow.filled == []
    assert flow.input_quality == "full"
    assert not flow.needs_follow_up


def test_flowfeatures_model_is_accepted(full_flow):
    flow = normalize_flow(FlowFeatures(**full_flow))
    assert flow.features == full_flow


# ---------------------------------------------------------------- unit parsing
@pytest.mark.parametrize("feature, value, expected", [
    ("flow_duration", "2 s", 2_000_000),
    ("flow_duration", "150 ms", 150_000),
    ("flow_duration", "46 µs", 46),
    ("flow_duration", "46us", 46),
    ("flow_duration", "1.5 minutes", 90_000_000),
    ("idle_mean", "3 sec", 3_000_000),
    ("total_len_fwd", "1.5 KB", 1_500),
    ("total_len_bwd", "2 MB", 2_000_000),
    ("total_len_bwd", "1,024 bytes", 1_024),
    ("flow_packets_s", "43k", 43_000),
    ("flow_packets_s", "1.2M pps", 1_200_000),
    ("fwd_packets_s", "500 packets/s", 500),
    ("total_fwd_packets", "15 packets", 15),
    ("total_fwd_packets", 7, 7),
    ("syn_flag_count", True, 1),
])
def test_parse_quantity(feature, value, expected):
    assert parse_quantity(feature, value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["abc", "5 parsecs", float("nan"), float("inf"), "1e999"])
def test_parse_quantity_rejects_garbage(value):
    with pytest.raises(FlowInputError):
        parse_quantity("flow_duration", value)


def test_empty_values_are_treated_as_missing():
    assert parse_quantity("flow_duration", None) is None
    assert parse_quantity("flow_duration", "  ") is None


# ---------------------------------------------------------------- range guardrails
def test_negative_value_rejected():
    with pytest.raises(FlowInputError, match="must be >= 0"):
        normalize_flow({**KEY_ONLY, "total_len_fwd": -5})


def test_window_minus_one_allowed_but_lower_rejected():
    assert normalize_flow({**KEY_ONLY, "init_win_bytes_forward": -1}).features["init_win_bytes_forward"] == -1
    with pytest.raises(FlowInputError):
        normalize_flow({**KEY_ONLY, "init_win_bytes_forward": -2})


def test_flag_above_one_is_clamped_with_warning():
    flow = normalize_flow({**KEY_ONLY, "syn_flag_count": 3})
    assert flow.features["syn_flag_count"] == 1
    assert any("syn_flag_count" in w for w in flow.warnings)


def test_unknown_field_is_ignored_with_warning():
    flow = normalize_flow({**KEY_ONLY, "favourite_colour": "blue"})
    assert any("favourite_colour" in w for w in flow.warnings)


def test_aliases_are_accepted():
    flow = normalize_flow({"Duration": "2 s", "fwd packets": 10, "bwd-packets": 8, "fwd_bytes": 1200, "bwd_bytes": 9600})
    assert flow.provided == ["flow_duration", "total_fwd_packets", "total_bwd_packets", "total_len_fwd", "total_len_bwd"]


def test_inconsistent_rate_produces_warning():
    flow = normalize_flow({**KEY_ONLY, "flow_packets_s": 5000})  # real rate is 18 / 2 s = 9
    assert any("flow_packets_s" in w for w in flow.warnings)


# ---------------------------------------------------------------- derivation
def test_derived_features_match_real_data(full_flow):
    keys = ["flow_duration", "total_fwd_packets", "total_bwd_packets", "total_len_fwd", "total_len_bwd"]
    flow = normalize_flow({k: full_flow[k] for k in keys})
    for name in ["flow_packets_s", "fwd_packets_s", "bwd_packets_s",
                 "fwd_packet_length_mean", "bwd_packet_length_mean", "down_up_ratio"]:
        assert name in flow.derived
        assert flow.features[name] == pytest.approx(full_flow[name], rel=1e-6), name


def test_derivation_values():
    flow = normalize_flow(KEY_ONLY)
    assert flow.features["flow_packets_s"] == pytest.approx(9.0)   # 18 packets / 2 s
    assert flow.features["fwd_packets_s"] == pytest.approx(5.0)
    assert flow.features["fwd_packet_length_mean"] == pytest.approx(120.0)
    assert flow.features["bwd_packet_length_mean"] == pytest.approx(1200.0)
    assert flow.features["down_up_ratio"] == 0.0                    # floor(8 / 10)
    assert flow.features["packet_length_mean"] == pytest.approx(10800 / 19)


def test_given_values_are_not_overwritten_by_derivation():
    flow = normalize_flow({**KEY_ONLY, "flow_packets_s": 9.5})
    assert flow.features["flow_packets_s"] == 9.5
    assert "flow_packets_s" in flow.provided and "flow_packets_s" not in flow.derived


# ---------------------------------------------------------------- fill + flag
def test_missing_features_are_filled_with_benign_medians_and_flagged():
    flow = normalize_flow(KEY_ONLY)
    medians = benign_medians()
    assert set(flow.provided) | set(flow.derived) | set(flow.filled) == set(FEATURE_NAMES)
    assert not set(flow.filled) & (set(flow.provided) | set(flow.derived))
    for name in flow.filled:
        assert flow.features[name] == medians[name]
    assert all(math.isfinite(v) for v in flow.features.values())


def test_input_quality_full(full_flow):
    assert normalize_flow(full_flow).input_quality == "full"
    first_20 = {n: full_flow[n] for n in FEATURE_NAMES[:20]}
    assert normalize_flow(first_20).input_quality == "full"


def test_input_quality_reduced_with_key_features_only():
    flow = normalize_flow(KEY_ONLY)  # 5 provided + 7 derived = 12
    assert flow.known_count == 12
    assert flow.input_quality == "reduced"


def test_input_quality_low_with_one_direction_only():
    flow = normalize_flow({"flow_duration": 2_000_000, "total_fwd_packets": 10, "total_len_fwd": 1200})
    assert flow.known_count < 12
    assert flow.input_quality == "low"
    assert not flow.needs_follow_up


# ---------------------------------------------------------------- follow-up rule
@pytest.mark.parametrize("drop, group", [
    (["flow_duration"], "flow duration"),
    (["total_fwd_packets", "total_bwd_packets"], "packet counts (forward/backward)"),
    (["total_len_fwd", "total_len_bwd"], "byte counts (forward/backward)"),
])
def test_follow_up_when_key_group_missing(drop, group):
    flow = normalize_flow({k: v for k, v in KEY_ONLY.items() if k not in drop})
    assert flow.needs_follow_up
    assert group in flow.missing_key_features
    assert group in follow_up_question(flow)


def test_one_direction_missing_is_filled_not_followed_up():
    flow = normalize_flow({k: v for k, v in KEY_ONLY.items() if k != "total_bwd_packets"})
    assert not flow.needs_follow_up
    assert "total_bwd_packets" in flow.filled


def test_no_follow_up_question_when_complete():
    assert follow_up_question(normalize_flow(KEY_ONLY)) is None


# ---------------------------------------------------------------- incident lookup
def test_incident_lookup():
    incident = get_incident_features("INC-000001")
    assert incident.incident_id == "INC-000001"
    assert incident.historical_label == "DoS Hulk"
    assert len(incident.flow.features) == len(FEATURE_NAMES)
    assert not incident.flow.needs_follow_up


@pytest.mark.parametrize("raw, expected", [("1", "INC-000001"), ("inc-42", "INC-000042"), (" INC-000123 ", "INC-000123")])
def test_incident_id_normalization(raw, expected):
    assert normalize_incident_id(raw) == expected


def test_incident_bad_id_and_unknown_id():
    with pytest.raises(FlowInputError):
        get_incident_features("not-an-id")
    with pytest.raises(KeyError):
        get_incident_features("INC-999999999")
