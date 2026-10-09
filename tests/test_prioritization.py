import pandas as pd
import pytest

from services.feature_input import FEATURE_NAMES
from services.prioritization import PRIORITY_BANDS, _band_for, prioritize, prioritize_batch, prioritize_flow


def make_prediction(attack="BENIGN", confidence=1.0, risk=0.0, percentile=50.0, level="Low", quality="full"):
    return {
        "status": "ok",
        "input": {"input_quality": quality},
        "attack_type": {"prediction": attack, "confidence": confidence},
        "risk": {"probability": risk},
        "anomaly": {"benign_percentile": percentile, "level": level},
    }


# ---------------------------------------------------------------- scoring and bands
def test_benign_is_p4():
    result = prioritize(make_prediction())
    assert result["priority"] == "P4" and result["score"] == 0


def test_ordering_by_severity():
    scores = [
        prioritize(make_prediction(attack, risk=1.0, percentile=95))["score"]
        for attack in ["BENIGN", "PortScan", "SSH-Patator", "DDoS", "Infiltration"]
    ]
    assert scores == sorted(scores)
    assert len(set(scores)) == len(scores)


def test_score_formula():
    result = prioritize(make_prediction("DDoS", confidence=0.5, risk=0.8, percentile=95))
    # 100 * (0.45*0.8 + 0.35*0.8*0.5 + 0.20*0.5) = 60
    assert result["score"] == pytest.approx(60.0)
    assert result["priority"] == "P2"


@pytest.mark.parametrize("score, priority", [(100, "P1"), (75, "P1"), (74.9, "P2"), (50, "P2"), (49.9, "P3"), (25, "P3"), (24.9, "P4"), (0, "P4")])
def test_band_edges(score, priority):
    assert _band_for(score)[0] == priority


def test_bands_are_ordered():
    assert [b[0] for b in PRIORITY_BANDS] == ["P1", "P2", "P3", "P4"]


# ---------------------------------------------------------------- overrides
def test_unknown_pattern_override():
    result = prioritize(make_prediction("BENIGN", risk=0.1, percentile=99.5, level="High"))
    assert result["priority"] == "P2"
    assert any("Unknown suspicious pattern" in o for o in result["overrides_applied"])


def test_critical_attack_override():
    result = prioritize(make_prediction("Heartbleed", confidence=0.5, risk=0.95))
    assert result["score"] < 75
    assert result["priority"] == "P1"
    assert any("Critical attack type" in o for o in result["overrides_applied"])


def test_reconnaissance_cap():
    capped = prioritize(make_prediction("PortScan", risk=1.0, percentile=95, level="Medium"))
    assert capped["priority"] == "P3"
    assert any("Reconnaissance" in o for o in capped["overrides_applied"])
    uncapped = prioritize(make_prediction("PortScan", risk=1.0, percentile=99.9, level="High"))
    assert uncapped["priority"] in {"P1", "P2"}


def test_low_input_quality_keeps_priority_but_adds_note():
    full = prioritize(make_prediction("DDoS", risk=1.0))
    low = prioritize(make_prediction("DDoS", risk=1.0, quality="low"))
    assert low["priority"] == full["priority"]
    assert low["confidence_note"].startswith("Indicative only")
    assert full["confidence_note"] is None


def test_follow_up_prediction_has_no_priority():
    result = prioritize({"status": "needs_follow_up", "follow_up_question": "Need packets"})
    assert result["priority"] is None
    assert result["reason"] == "Need packets"


def test_missing_model_output_is_an_error():
    prediction = make_prediction()
    del prediction["anomaly"]
    with pytest.raises(ValueError):
        prioritize(prediction)


# ---------------------------------------------------------------- end to end
def test_prioritize_flow_end_to_end(full_flow):
    result = prioritize_flow(full_flow)
    assert result["status"] == "ok"
    assert result["priority"]["priority"] in {"P1", "P2"}  # a real DDoS flow


def test_batch_is_ranked_and_keeps_index(flows_df):
    rows = flows_df.groupby("label").head(2)
    flows = [{n: float(r[n]) for n in FEATURE_NAMES} for _, r in rows.iterrows()] + [{"flow_duration": 5}]
    ranked = prioritize_batch(flows)
    assert sorted(r["index"] for r in ranked) == list(range(len(flows)))
    assert ranked[-1]["status"] == "needs_follow_up"
    ranks = [int(r["priority"]["priority"][1]) for r in ranked if r["priority"]["priority"]]
    assert ranks == sorted(ranks)


def test_priority_distribution_on_real_traffic(flows_df):
    sample = pd.concat(
        [g.sample(n=min(len(g), 200), random_state=0) for _, g in flows_df.groupby("label")]
    ).reset_index(drop=True)
    ranked = prioritize_batch([{n: float(r[n]) for n in FEATURE_NAMES} for _, r in sample.iterrows()])
    priority = pd.Series({r["index"]: r["priority"]["priority"] for r in ranked}).sort_index()
    share = pd.crosstab(sample["label"], priority, normalize="index")

    assert share.loc["BENIGN", "P4"] >= 0.95
    assert share.loc["Heartbleed", "P1"] >= 0.9
    assert share.loc["Infiltration", "P1"] >= 0.75
    assert share.loc["PortScan", "P3"] >= 0.9
    for attack in ["SSH-Patator", "FTP-Patator", "Web Attack - Brute Force", "Web Attack - XSS"]:
        assert share.loc[attack, ["P1", "P2"]].sum() >= 0.9, attack
