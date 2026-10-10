"""
Feedback loop, step 3: turn analyst label corrections into training data and retrain the
attack-type classifier.

    collect_corrections(store)   one row per corrected flow: 24 features + majority label
    retrain_with_corrections()   same split and model settings as ml/attack_classifier.ipynb;
                                 corrections go into the TRAINING split only; old and new
                                 model are compared on the same test rows
    retrain_and_maybe_promote()  saves a new version and makes it active only if test macro F1
                                 did not drop (ml/models/versions/, active.json)

Safeguards: a correction is used only for an incident or a flow with full input quality;
conflicting corrections for the same flow are resolved by majority vote and skipped on a tie.
"""
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config import (
    ACTIVE_CLASSIFIER_POINTER,
    ATTACK_CLASSIFIER_META_FILE,
    CLEANED_FLOWS_FILE,
    ML_MODEL_VERSIONS_DIR,
)
from services.feature_input import FEATURE_NAMES, FlowInputError, get_incident_features, normalize_flow

logger = logging.getLogger(__name__)

FEATURES = list(FEATURE_NAMES)
LABEL = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_JOBS = 2
KEY_DECIMALS = 4  # feature vectors are matched after rounding (Parquet vs CSV float noise)


# ============================================================================
# Corrections → training rows
# ============================================================================
def _flow_features(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """24 features for a corrected turn, or (None, reason it can't be used)."""
    if row.get("incident_id"):
        try:
            return dict(get_incident_features(row["incident_id"]).flow.features), None
        except (FlowInputError, KeyError, ValueError) as exc:
            return None, f"incident lookup failed: {exc}"
    args = row.get("flow_args") or {}
    metrics = {k: v for k, v in args.items() if k in FEATURE_NAMES}
    if not metrics:
        return None, "no flow metrics"
    try:
        flow = normalize_flow(metrics)
    except FlowInputError as exc:
        return None, f"invalid metrics: {exc}"
    if flow.input_quality != "full":
        return None, f"input quality {flow.input_quality} (needs full)"
    return dict(flow.features), None


def collect_corrections(store) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Corrected flows (24 features, label, source, votes) and the corrections that were skipped."""
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"labels": Counter(), "features": None, "source": None})
    skipped: List[Dict[str, Any]] = []
    for row in store.list(corrected_only=True, limit=1_000_000):
        features, reason = _flow_features(row)
        if features is None:
            skipped.append({"feedback_id": row["id"], "reason": reason})
            continue
        key = row.get("incident_id") or json.dumps([round(features[f], KEY_DECIMALS) for f in FEATURES])
        group = groups[key]
        group["labels"][row["corrected_label"]] += 1
        group["features"], group["source"] = features, row.get("incident_id") or "flow"

    records = []
    for key, group in groups.items():
        (label, votes), *rest = group["labels"].most_common()
        if rest and rest[0][1] == votes:
            skipped.append({"key": key, "reason": f"tied corrections {dict(group['labels'])}"})
            continue
        records.append({**group["features"], LABEL: label, "source": group["source"], "votes": votes,
                        "total_votes": sum(group["labels"].values())})
    columns = [*FEATURES, LABEL, "source", "votes", "total_votes"]
    return pd.DataFrame(records, columns=columns), skipped


# ============================================================================
# Retraining
# ============================================================================
def _keys(frame: pd.DataFrame) -> pd.Series:
    return frame[FEATURES].round(KEY_DECIMALS).apply(tuple, axis=1)


def new_classifier(n_estimators: int = 100) -> RandomForestClassifier:
    """Same settings as ml/attack_classifier.ipynb."""
    return RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS
    )


def _metrics(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    pred = model.predict(X)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 4),
    }


def retrain_with_corrections(
    flows: pd.DataFrame,
    corrections: pd.DataFrame,
    current_model,
    n_estimators: int = 100,
) -> Tuple[Any, Dict[str, Any]]:
    """Train a new classifier with the corrections and compare it with the current one.

    flows: the original training table (24 features + label). The split is the notebook's
    (dedupe on features, 80/20 stratified, seed 42). Test rows whose features were corrected
    are removed from the test set and their corrected version is added to training, so both
    models are compared on identical, untouched test rows.
    """
    data = flows.drop_duplicates(subset=FEATURES).reset_index(drop=True)
    train, test = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=data[LABEL])

    corr = corrections[[*FEATURES, LABEL]].copy()
    corr_keys = set(_keys(corr)) if len(corr) else set()
    test_keys = _keys(test)
    in_test = test_keys.isin(corr_keys)
    test_eval = test[~in_test]

    # Corrections replace matching training rows (keep="last") or are added.
    combined = pd.concat([train[[*FEATURES, LABEL]], corr], ignore_index=True)
    combined["_key"] = _keys(combined)
    combined = combined.drop_duplicates(subset="_key", keep="last").drop(columns="_key")
    relabelled = int(_keys(train).isin(corr_keys).sum())

    model = new_classifier(n_estimators)
    model.fit(combined[FEATURES], combined[LABEL])

    current = _metrics(current_model, test_eval[FEATURES], test_eval[LABEL])
    new = _metrics(model, test_eval[FEATURES], test_eval[LABEL])
    report = {
        "corrections_used": len(corr),
        "training_rows_relabelled": relabelled,
        "corrections_from_test_split": int(in_test.sum()),
        "n_train": len(combined),
        "n_test": len(test_eval),
        "current_test_metrics": current,
        "new_test_metrics": new,
        "macro_f1_change": round(new["macro_f1"] - current["macro_f1"], 4),
        "promote": new["macro_f1"] >= current["macro_f1"],
    }
    if len(corr):
        report["current_on_corrections"] = round(float(accuracy_score(corr[LABEL], current_model.predict(corr[FEATURES]))), 4)
        report["new_on_corrections"] = round(float(accuracy_score(corr[LABEL], model.predict(corr[FEATURES]))), 4)
    return model, report


# ============================================================================
# Versions
# ============================================================================
def _read_pointer() -> Optional[Dict[str, Any]]:
    if not ACTIVE_CLASSIFIER_POINTER.exists():
        return None
    return json.loads(ACTIVE_CLASSIFIER_POINTER.read_text(encoding="utf-8"))


def list_versions() -> List[Dict[str, Any]]:
    """Saved retrained versions (newest first) and which one is active."""
    active = (_read_pointer() or {}).get("version", "original")
    versions = [{"version": "original", "active": active == "original", "macro_f1": None, "created_at": None}]
    for meta_file in sorted(ML_MODEL_VERSIONS_DIR.glob("attack_classifier_*_meta.json"), reverse=True):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        versions.append({
            "version": meta["version"],
            "active": meta["version"] == active,
            "macro_f1": meta["test_metrics"]["macro_f1"],
            "created_at": meta["trained_at"],
            "corrections_used": meta["feedback"]["corrections_used"],
        })
    return versions


def activate(version: str) -> Dict[str, Any]:
    """Make a saved version (or "original") the classifier the ML service loads."""
    if version == "original":
        if ACTIVE_CLASSIFIER_POINTER.exists():
            ACTIVE_CLASSIFIER_POINTER.unlink()
        return {"version": "original"}
    model_file = ML_MODEL_VERSIONS_DIR / f"attack_classifier_{version}.pkl"
    meta_file = ML_MODEL_VERSIONS_DIR / f"attack_classifier_{version}_meta.json"
    if not model_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Version {version} not found in {ML_MODEL_VERSIONS_DIR}")
    pointer = {"version": version, "model_file": model_file.name, "meta_file": meta_file.name,
               "activated_at": datetime.now().isoformat(timespec="seconds")}
    ACTIVE_CLASSIFIER_POINTER.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    return pointer


def retrain_and_maybe_promote(store, dry_run: bool = False, n_estimators: int = 100) -> Dict[str, Any]:
    """Collect corrections, retrain, save a new version and promote it if it is not worse."""
    from services.ml_service import active_classifier_files

    corrections, skipped = collect_corrections(store)
    result: Dict[str, Any] = {"corrections_available": len(corrections), "skipped": skipped}
    if corrections.empty:
        result["status"] = "nothing_to_do"
        result["message"] = "No usable label corrections yet (need incidents or flows with full input)."
        return result

    current_file, _, current_version = active_classifier_files()
    current_model = joblib.load(current_file)
    flows = pd.read_csv(CLEANED_FLOWS_FILE)
    model, report = retrain_with_corrections(flows, corrections, current_model, n_estimators)
    result.update(report)
    result["current_version"] = current_version

    if dry_run:
        result["status"] = "dry_run"
        return result

    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    ML_MODEL_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    base_meta = json.loads(ATTACK_CLASSIFIER_META_FILE.read_text(encoding="utf-8"))
    meta = {
        **base_meta,
        "version": version,
        "base_version": current_version,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "classes": [str(c) for c in model.classes_],
        "n_train": report["n_train"],
        "n_test": report["n_test"],
        "test_metrics": report["new_test_metrics"],
        "feedback": {k: report[k] for k in ("corrections_used", "training_rows_relabelled",
                                            "corrections_from_test_split", "current_test_metrics",
                                            "macro_f1_change")},
    }
    joblib.dump(model, ML_MODEL_VERSIONS_DIR / f"attack_classifier_{version}.pkl")
    (ML_MODEL_VERSIONS_DIR / f"attack_classifier_{version}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    result["version"] = version
    if report["promote"]:
        activate(version)
        result["status"] = "promoted"
        result["message"] = (f"Version {version} is now active (macro F1 {report['current_test_metrics']['macro_f1']}"
                             f" -> {report['new_test_metrics']['macro_f1']}). Restart running services to load it.")
    else:
        result["status"] = "kept_current"
        result["message"] = (f"Version {version} saved but NOT activated: test macro F1 dropped by "
                             f"{-report['macro_f1_change']}. Activate it manually with --activate {version} if wanted.")
    return result


def export_corrections_csv(store, path) -> Dict[str, Any]:
    corrections, skipped = collect_corrections(store)
    corrections.to_csv(path, index=False)
    return {"rows": len(corrections), "skipped": skipped, "path": str(path)}


__all__ = [
    "collect_corrections", "retrain_with_corrections", "retrain_and_maybe_promote", "export_corrections_csv",
    "list_versions", "activate", "new_classifier",
]
