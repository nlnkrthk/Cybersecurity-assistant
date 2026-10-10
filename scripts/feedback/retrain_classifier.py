#!/usr/bin/env python3
"""Retrain the attack-type classifier with analyst label corrections (feedback loop).

    python scripts/feedback/retrain_classifier.py --dry-run     # train and compare, save nothing
    python scripts/feedback/retrain_classifier.py               # save a new version; activate if not worse
    python scripts/feedback/retrain_classifier.py --list        # saved versions
    python scripts/feedback/retrain_classifier.py --activate original      # roll back
    python scripts/feedback/retrain_classifier.py --activate v20261010_120000

The training reproduces ml/attack_classifier.ipynb (same features, split and model settings).
Corrections are added to the training split only, and the new model becomes active only if
its test macro F1 is not lower than the current model's on the same test rows. Restart the
API / chat after a change so the ML service loads the active version.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from feedback import feedback_db_path, get_feedback_store  # noqa: E402
from feedback.retraining import activate, list_versions, retrain_and_maybe_promote  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain the classifier with analyst corrections.")
    parser.add_argument("--dry-run", action="store_true", help="Train and compare, but save nothing.")
    parser.add_argument("--list", action="store_true", help="List saved classifier versions.")
    parser.add_argument("--activate", metavar="VERSION", help="Activate a saved version, or 'original'.")
    parser.add_argument("--trees", type=int, default=100, help="Number of trees (default 100, as in the notebook).")
    args = parser.parse_args()

    if args.list:
        for item in list_versions():
            mark = "*" if item["active"] else " "
            extra = f"  macro F1 {item['macro_f1']}  corrections {item['corrections_used']}" if item["macro_f1"] else ""
            print(f" {mark} {item['version']:<20}{extra}")
        return 0
    if args.activate:
        try:
            pointer = activate(args.activate)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[OK] Active classifier: {pointer['version']}. Restart the API / chat to load it.")
        return 0

    if not feedback_db_path().exists():
        print("No feedback yet; nothing to retrain with.")
        return 0
    print("Collecting corrections and retraining (about a minute)...")
    result = retrain_and_maybe_promote(get_feedback_store(), dry_run=args.dry_run, n_estimators=args.trees)

    print(f"\nStatus: {result['status']}")
    if result.get("message"):
        print(result["message"])
    if "new_test_metrics" in result:
        print(f"\nCorrections used: {result['corrections_used']} "
              f"(relabelled training rows: {result['training_rows_relabelled']}, "
              f"from the test split: {result['corrections_from_test_split']})")
        print(f"Test rows compared: {result['n_test']}")
        print(f"  current ({result['current_version']}): {json.dumps(result['current_test_metrics'])}")
        print(f"  new:                {json.dumps(result['new_test_metrics'])}")
        print(f"  macro F1 change:    {result['macro_f1_change']:+.4f}")
        if "new_on_corrections" in result:
            print(f"  accuracy on the corrected flows: current {result['current_on_corrections']:.0%}, "
                  f"new {result['new_on_corrections']:.0%}")
    for item in result.get("skipped", []):
        print(f"  skipped: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
