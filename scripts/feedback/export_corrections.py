#!/usr/bin/env python3
"""Export analyst label corrections as a training CSV (24 flow features + corrected label).

    python scripts/feedback/export_corrections.py                       # data/feedback/corrections.csv
    python scripts/feedback/export_corrections.py --output my_file.csv

Only incidents and flows with full input quality are exported; conflicting corrections for
the same flow use the majority label (ties are skipped and listed).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback import feedback_db_path, get_feedback_store  # noqa: E402
from feedback.retraining import export_corrections_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export analyst label corrections for retraining.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "feedback" / "corrections.csv")
    args = parser.parse_args()

    if not feedback_db_path().exists():
        print("No feedback yet; nothing to export.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = export_corrections_csv(get_feedback_store(), args.output)
    print(f"Exported {result['rows']} corrected flow(s) to {result['path']}")
    for item in result["skipped"]:
        print(f"  skipped: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
