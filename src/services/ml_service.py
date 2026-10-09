"""
ML service: one place that loads the three trained models and answers questions about flows.

    classify(flow)        attack type (Random Forest, 15 classes)
    predict_risk(flow)    probability the flow is malicious (calibrated HistGradientBoosting)
    detect_anomaly(flow)  how unusual the flow is vs normal traffic (Isolation Forest)
    predict(flow)         all three, no SHAP (fast; used for prioritization and batches)
    explain(flow)         all three + SHAP reasons + a plain-language summary
    predict_batch(flows)  predict() for many flows, vectorized

Every function accepts partial input (dict, FlowFeatures or NormalizedFlow); missing
features are derived or filled and flagged by services.feature_input. If key features
are missing, no model runs and the result asks for a follow-up instead.

The service has no dependency on RAG, the LLM or LangGraph.
"""
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANOMALY_DETECTOR_FILE,
    ANOMALY_DETECTOR_META_FILE,
    ATTACK_CLASSIFIER_FILE,
    ATTACK_CLASSIFIER_META_FILE,
    ATTACK_PROFILES_FILE,
    INTRUSION_RISK_FILE,
    INTRUSION_RISK_META_FILE,
)
from services.feature_input import (
    FEATURE_NAMES,
    FlowInput,
    NormalizedFlow,
    follow_up_question,
    load_feature_reference,
    normalize_flow,
)

logger = logging.getLogger(__name__)

TOP_ALTERNATIVES = 3
UNUSUAL_EXTREMENESS = 45  # outside the 5th-95th percentile of normal traffic


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required ML file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required ML model not found: {path}")
    return joblib.load(path)


def _fmt_value(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


class MLService:
    """Loads the models once and exposes prediction and explanation functions."""

    def __init__(self) -> None:
        logger.info("Loading ML models")
        self.classifier = _load_model(ATTACK_CLASSIFIER_FILE)
        self.risk_model = _load_model(INTRUSION_RISK_FILE)
        self.anomaly_model = _load_model(ANOMALY_DETECTOR_FILE)

        self.classifier_meta = _read_json(ATTACK_CLASSIFIER_META_FILE)
        self.risk_meta = _read_json(INTRUSION_RISK_META_FILE)
        self.anomaly_meta = _read_json(ANOMALY_DETECTOR_META_FILE)
        self.attack_profiles = _read_json(ATTACK_PROFILES_FILE)
        self.reference = load_feature_reference()

        self.features = list(FEATURE_NAMES)
        for name, meta in [("classifier", self.classifier_meta), ("risk", self.risk_meta), ("anomaly", self.anomaly_meta)]:
            if meta["features"] != self.features:
                raise ValueError(f"{name} model feature order does not match services.feature_input.FEATURE_NAMES")
            trained_with = meta.get("sklearn_version")
            if trained_with and trained_with != sklearn.__version__:
                logger.warning(
                    f"{name} model was trained with scikit-learn {trained_with}, running {sklearn.__version__}"
                )

        self.classes = list(self.classifier.classes_)
        self.risk_bands = sorted(self.risk_meta["risk_bands"].items(), key=lambda kv: kv[1], reverse=True)
        self.decision_threshold = float(self.risk_meta["decision_threshold"])
        self.anomaly_thresholds = self.anomaly_meta["thresholds"]
        anomaly_ref = self.anomaly_meta["benign_percentile_reference"]
        self._anomaly_ref_scores = np.asarray(anomaly_ref["scores"])
        self._anomaly_ref_levels = np.asarray(anomaly_ref["levels"])
        self._feature_quantiles = {
            f: np.asarray(self.reference["features"][f]["benign_quantiles"]) for f in self.features
        }

        # SHAP explainers are built lazily: shap is slow to import and only explain() needs it.
        self._clf_explainer = None
        self._risk_explainer = None
        self._explainer_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _frame(self, flows: List[NormalizedFlow]) -> pd.DataFrame:
        return pd.DataFrame([f.features for f in flows], columns=self.features)

    def benign_percentile(self, feature: str, value: float) -> float:
        """Mid-rank percentile of `value` among normal (BENIGN) traffic, 0-100."""
        q = self._feature_quantiles[feature]
        lo = np.searchsorted(q, value, side="left")
        hi = np.searchsorted(q, value, side="right")
        return float((lo + hi) / 2 / (len(q) - 1) * 100)

    def describe(self, feature: str, value: float) -> str:
        """e.g. 'Flow duration = 46 µs (lower than 94% of normal traffic)'."""
        info = self.reference["features"][feature]
        pct = self.benign_percentile(feature, value)
        share = min(pct if pct >= 50 else 100 - pct, 99.9)
        share_txt = f"{share:.1f}%" if share >= 99 else f"{share:.0f}%"
        where = "higher" if pct >= 50 else "lower"
        unit = f" {info['unit']}" if info["unit"] else ""
        return f"{info['display_name']} = {_fmt_value(value)}{unit} ({where} than {share_txt} of normal traffic)"

    def _risk_level(self, probability: float) -> str:
        return next(name for name, lower in self.risk_bands if probability >= lower)

    def _anomaly_level(self, score: float) -> str:
        if score >= self.anomaly_thresholds["high"]:
            return "High"
        if score >= self.anomaly_thresholds["medium"]:
            return "Medium"
        return "Low"

    @staticmethod
    def _overall_confidence(flow: NormalizedFlow, model_confidence: float) -> str:
        if flow.input_quality == "low" or model_confidence < 0.5:
            return "low"
        if flow.input_quality == "reduced" or model_confidence < 0.8:
            return "medium"
        return "high"

    @staticmethod
    def _input_summary(flow: NormalizedFlow) -> Dict[str, Any]:
        return {
            "input_quality": flow.input_quality,
            "known_count": flow.known_count,
            "provided": flow.provided,
            "derived": flow.derived,
            "filled": flow.filled,
            "warnings": flow.warnings,
        }

    def _follow_up_result(self, flow: NormalizedFlow) -> Dict[str, Any]:
        return {
            "status": "needs_follow_up",
            "input": self._input_summary(flow),
            "missing_key_features": flow.missing_key_features,
            "follow_up_question": follow_up_question(flow),
        }

    # ------------------------------------------------------------------
    # Vectorized model runs
    # ------------------------------------------------------------------
    def _classify_frame(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        proba = self.classifier.predict_proba(X)
        results = []
        for row in proba:
            order = np.argsort(row)[::-1]
            results.append({
                "prediction": self.classes[order[0]],
                "confidence": round(float(row[order[0]]), 4),
                "alternatives": [
                    {"attack_type": self.classes[i], "probability": round(float(row[i]), 4)}
                    for i in order[1:1 + TOP_ALTERNATIVES] if row[i] >= 0.01
                ],
            })
        return results

    def _risk_frame(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        proba = self.risk_model.predict_proba(X)[:, 1]
        return [
            {
                "probability": round(float(p), 4),
                "is_intrusion": bool(p >= self.decision_threshold),
                "level": self._risk_level(float(p)),
            }
            for p in proba
        ]

    def _anomaly_frame(self, X: pd.DataFrame, flows: List[NormalizedFlow]) -> List[Dict[str, Any]]:
        scores = -self.anomaly_model.score_samples(X)
        percentiles = np.interp(scores, self._anomaly_ref_scores, self._anomaly_ref_levels)
        results = []
        for i, (score, pct) in enumerate(zip(scores, percentiles)):
            filled = set(flows[i].filled)
            values = X.iloc[i]
            feature_pcts = {f: self.benign_percentile(f, values[f]) for f in self.features if f not in filled}
            unusual = sorted(feature_pcts, key=lambda f: abs(feature_pcts[f] - 50), reverse=True)[:3]
            results.append({
                "score": round(float(score), 4),
                "benign_percentile": round(float(pct), 1),
                "level": self._anomaly_level(float(score)),
                "most_unusual_features": [
                    {
                        "feature": f,
                        "value": float(values[f]),
                        "benign_percentile": round(feature_pcts[f], 1),
                        "text": self.describe(f, values[f]),
                    }
                    for f in unusual if abs(feature_pcts[f] - 50) >= UNUSUAL_EXTREMENESS
                ],
            })
        return results

    def _run(self, flows: List[NormalizedFlow], parts: Iterable[str]) -> List[Dict[str, Any]]:
        """Run the requested models on normalized flows; follow-up results skip the models."""
        parts = set(parts)
        results: List[Optional[Dict[str, Any]]] = [None] * len(flows)
        ready = [i for i, f in enumerate(flows) if not f.needs_follow_up]
        for i, f in enumerate(flows):
            if f.needs_follow_up:
                results[i] = self._follow_up_result(f)

        if ready:
            ready_flows = [flows[i] for i in ready]
            X = self._frame(ready_flows)
            attack = self._classify_frame(X) if "attack_type" in parts else None
            risk = self._risk_frame(X) if "risk" in parts else None
            anomaly = self._anomaly_frame(X, ready_flows) if "anomaly" in parts else None

            for j, i in enumerate(ready):
                result: Dict[str, Any] = {"status": "ok", "input": self._input_summary(flows[i])}
                if attack is not None:
                    result["attack_type"] = attack[j]
                if risk is not None:
                    result["risk"] = risk[j]
                if anomaly is not None:
                    result["anomaly"] = anomaly[j]
                model_conf = attack[j]["confidence"] if attack is not None else (
                    max(risk[j]["probability"], 1 - risk[j]["probability"]) if risk is not None else 1.0
                )
                result["overall_confidence"] = self._overall_confidence(flows[i], model_conf)
                results[i] = result
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, flow: FlowInput) -> Dict[str, Any]:
        """Attack type only."""
        return self._run([normalize_flow(flow)], ["attack_type"])[0]

    def predict_risk(self, flow: FlowInput) -> Dict[str, Any]:
        """Intrusion probability and risk level only."""
        return self._run([normalize_flow(flow)], ["risk"])[0]

    def detect_anomaly(self, flow: FlowInput) -> Dict[str, Any]:
        """Anomaly score and level only."""
        return self._run([normalize_flow(flow)], ["anomaly"])[0]

    def predict(self, flow: FlowInput) -> Dict[str, Any]:
        """All three models, no SHAP."""
        return self._run([normalize_flow(flow)], ["attack_type", "risk", "anomaly"])[0]

    def predict_batch(self, flows: Iterable[FlowInput]) -> List[Dict[str, Any]]:
        """predict() for many flows in one vectorized pass."""
        normalized = [normalize_flow(f) for f in flows]
        return self._run(normalized, ["attack_type", "risk", "anomaly"])

    def explain(self, flow: FlowInput, top_k: int = 3) -> Dict[str, Any]:
        """All three models plus SHAP reasons and a plain-language summary."""
        normalized = normalize_flow(flow)
        result = self._run([normalized], ["attack_type", "risk", "anomaly"])[0]
        if result["status"] != "ok":
            return result

        self._ensure_explainers()
        X = self._frame([normalized])
        values = X.iloc[0]
        filled = set(normalized.filled)

        k = self.classes.index(result["attack_type"]["prediction"])
        clf_shap = pd.Series(self._clf_explainer(X).values[0, :, k], index=self.features).drop(list(filled))
        type_reasons = [
            self._reason(f, values[f], s, "toward " + result["attack_type"]["prediction"])
            for f, s in clf_shap.sort_values(ascending=False).head(top_k).items() if s > 0
        ]

        risk_shap = pd.Series(self._risk_explainer(X).values[0], index=self.features).drop(list(filled))
        risk_reasons = [
            self._reason(f, values[f], s, "raises risk" if s > 0 else "lowers risk")
            for f, s in risk_shap.reindex(risk_shap.abs().sort_values(ascending=False).index).head(top_k).items()
        ]

        result["explanation"] = {
            "attack_type_reasons": type_reasons,
            "risk_reasons": risk_reasons,
            "summary": self._summary(result, type_reasons, normalized),
        }
        return result

    def attack_profile(self, attack_type: str) -> Dict[str, Any]:
        """Typical traffic characteristics of an attack type (from SHAP analysis)."""
        if attack_type not in self.attack_profiles:
            raise KeyError(f"No profile for {attack_type!r}; known types: {sorted(self.attack_profiles)}")
        return {"attack_type": attack_type, **self.attack_profiles[attack_type]}

    def list_attack_types(self) -> List[str]:
        return list(self.classes)

    # ------------------------------------------------------------------
    # Explanation internals
    # ------------------------------------------------------------------
    def _ensure_explainers(self) -> None:
        if self._clf_explainer is not None:
            return
        with self._explainer_lock:
            if self._clf_explainer is None:
                import shap  # imported here: slow import, only needed for explanations

                risk_base = self.risk_model.calibrated_classifiers_[0].estimator
                risk_base = getattr(risk_base, "estimator", risk_base)  # unwrap FrozenEstimator
                self._risk_explainer = shap.TreeExplainer(risk_base)
                self._clf_explainer = shap.TreeExplainer(self.classifier)

    def _reason(self, feature: str, value: float, shap_value: float, effect: str) -> Dict[str, Any]:
        return {
            "feature": feature,
            "value": float(value),
            "shap": round(float(shap_value), 4),
            "effect": effect,
            "benign_percentile": round(self.benign_percentile(feature, value), 1),
            "text": self.describe(feature, value),
        }

    @staticmethod
    def _summary(result: Dict[str, Any], type_reasons: List[Dict[str, Any]], flow: NormalizedFlow) -> str:
        attack, risk, anomaly = result["attack_type"], result["risk"], result["anomaly"]
        reasons = "; ".join(r["text"] for r in type_reasons) or "no single dominant feature"
        summary = (
            f"Classified as {attack['prediction']} ({attack['confidence']:.0%} confidence). "
            f"Intrusion risk {risk['probability']:.0%} ({risk['level']}). Anomaly level {anomaly['level']}. "
            f"Main reasons: {reasons}."
        )
        if attack["prediction"] == "BENIGN" and anomaly["level"] == "High":
            summary += " Note: looks benign to the classifier but is highly unusual - possible unknown attack, review."
        if flow.filled:
            summary += (
                f" Based on {flow.known_count} of {len(FEATURE_NAMES)} metrics; "
                f"{len(flow.filled)} were filled with normal-traffic values"
            )
            if flow.input_quality == "low":
                summary += ", so treat this as indicative only."
            elif flow.input_quality == "reduced":
                summary += ", so treat this as a reduced-confidence result."
            else:
                summary += "."
        return summary


_service: Optional[MLService] = None
_service_lock = threading.Lock()


def get_ml_service() -> MLService:
    """Shared MLService instance, created on first use."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = MLService()
    return _service
