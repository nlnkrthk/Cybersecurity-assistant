"""
Incident prioritization: turns the three model outputs into a P1-P4 priority.

    score = 100 x ( 0.45 x risk probability
                  + 0.35 x attack-type severity x classifier confidence
                  + 0.20 x anomaly component )

    anomaly component = clip((benign percentile - 90) / 10, 0, 1)

Overrides after scoring:
  1. classifier says BENIGN but anomaly is High -> at least P2 (possible unknown attack)
  2. critical-severity attack type with risk >= 0.9 -> P1
  3. low-severity reconnaissance (PortScan) capped at P3 unless anomaly is High
  4. low input quality keeps the priority but marks it as indicative

Weights were checked on a stratified sample of CIC-IDS2017 (200 flows per type):
no weight setting put PortScan in P3 without also demoting brute-force and web
attacks, hence override 3 instead of different weights.
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from services.feature_input import FlowInput
from services.ml_service import get_ml_service

logger = logging.getLogger(__name__)

# Attack-type severity: (label, weight).
SEVERITY = {
    "Heartbleed": ("Critical", 1.0),
    "Infiltration": ("Critical", 1.0),
    "Web Attack - Sql Injection": ("Critical", 1.0),
    "DDoS": ("High", 0.8),
    "DoS Hulk": ("High", 0.8),
    "DoS GoldenEye": ("High", 0.8),
    "DoS slowloris": ("High", 0.8),
    "DoS Slowhttptest": ("High", 0.8),
    "Bot": ("High", 0.8),
    "Web Attack - XSS": ("High", 0.8),
    "Web Attack - Brute Force": ("High", 0.8),
    "SSH-Patator": ("Medium", 0.6),
    "FTP-Patator": ("Medium", 0.6),
    "PortScan": ("Low", 0.4),
    "BENIGN": ("None", 0.0),
}
DEFAULT_SEVERITY = ("High", 0.8)  # unknown attack types are treated as serious

WEIGHT_RISK = 0.45
WEIGHT_SEVERITY = 0.35
WEIGHT_ANOMALY = 0.20
ANOMALY_FLOOR_PERCENTILE = 90.0

# (priority, label, minimum score, meaning), highest first.
PRIORITY_BANDS = [
    ("P1", "Critical", 75.0, "Investigate now"),
    ("P2", "High", 50.0, "Investigate today"),
    ("P3", "Medium", 25.0, "Review when possible"),
    ("P4", "Low", 0.0, "Monitor"),
]
_RANK = {band[0]: i for i, band in enumerate(PRIORITY_BANDS)}


def _band_for(score: float):
    return next(band for band in PRIORITY_BANDS if score >= band[2])


def _band_by_priority(priority: str):
    return PRIORITY_BANDS[_RANK[priority]]


def score_components(prediction: Dict[str, Any]) -> Dict[str, float]:
    """The three weighted inputs of the priority score, each 0-1."""
    attack = prediction["attack_type"]
    severity_weight = SEVERITY.get(attack["prediction"], DEFAULT_SEVERITY)[1]
    anomaly_pct = prediction["anomaly"]["benign_percentile"]
    return {
        "risk": float(prediction["risk"]["probability"]),
        "severity": severity_weight * float(attack["confidence"]),
        "anomaly": min(max((anomaly_pct - ANOMALY_FLOOR_PERCENTILE) / (100 - ANOMALY_FLOOR_PERCENTILE), 0.0), 1.0),
    }


def prioritize(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Priority for one flow, from the output of MLService.predict() or explain()."""
    if prediction.get("status") != "ok":
        return {
            "priority": None,
            "score": None,
            "label": "Needs more input",
            "reason": prediction.get("follow_up_question") or "Key flow metrics are missing",
            "overrides_applied": [],
            "confidence_note": None,
        }
    for part in ("attack_type", "risk", "anomaly"):
        if part not in prediction:
            raise ValueError(f"prioritize() needs all three model outputs; missing {part!r} (use predict())")

    components = score_components(prediction)
    score = round(100 * (
        WEIGHT_RISK * components["risk"]
        + WEIGHT_SEVERITY * components["severity"]
        + WEIGHT_ANOMALY * components["anomaly"]
    ), 1)
    priority = _band_for(score)[0]

    attack, risk, anomaly = prediction["attack_type"], prediction["risk"], prediction["anomaly"]
    severity_label = SEVERITY.get(attack["prediction"], DEFAULT_SEVERITY)[0]
    overrides: List[str] = []

    if attack["prediction"] == "BENIGN" and anomaly["level"] == "High" and _RANK[priority] > _RANK["P2"]:
        priority = "P2"
        overrides.append("Unknown suspicious pattern: classified BENIGN but highly unusual, review")
    if severity_label == "Critical" and risk["probability"] >= 0.9 and priority != "P1":
        priority = "P1"
        overrides.append(f"Critical attack type ({attack['prediction']}) with high intrusion risk")
    if severity_label == "Low" and anomaly["level"] != "High" and _RANK[priority] < _RANK["P3"]:
        priority = "P3"
        overrides.append(f"Reconnaissance ({attack['prediction']}) capped at P3 unless highly anomalous")

    confidence_note = None
    if prediction["input"]["input_quality"] == "low":
        confidence_note = "Indicative only: most metrics were missing; verify with full flow data"
    elif prediction["input"]["input_quality"] == "reduced":
        confidence_note = "Some metrics were missing and filled with normal-traffic values"

    band = _band_by_priority(priority)
    return {
        "priority": priority,
        "score": score,
        "label": band[1],
        "action": band[3],
        "reason": (
            f"{attack['prediction']} ({attack['confidence']:.0%}, severity {severity_label}), "
            f"intrusion risk {risk['probability']:.0%}, anomaly {anomaly['level']}"
        ),
        "components": {k: round(v, 4) for k, v in components.items()},
        "overrides_applied": overrides,
        "confidence_note": confidence_note,
    }


def prioritize_flow(flow: FlowInput) -> Dict[str, Any]:
    """Run the models on one flow and prioritize it."""
    prediction = get_ml_service().predict(flow)
    return {**prediction, "priority": prioritize(prediction)}


def prioritize_batch(flows: Iterable[FlowInput]) -> List[Dict[str, Any]]:
    """Run the models on many flows and return them ranked, most urgent first.

    Each item keeps its original position as `index`. Flows that need more input
    are placed at the end.
    """
    predictions = get_ml_service().predict_batch(list(flows))
    ranked = [
        {"index": i, **prediction, "priority": prioritize(prediction)}
        for i, prediction in enumerate(predictions)
    ]
    ranked.sort(key=lambda item: (
        item["priority"]["priority"] is None,
        _RANK.get(item["priority"]["priority"], len(_RANK)),
        -(item["priority"]["score"] or 0),
    ))
    return ranked
