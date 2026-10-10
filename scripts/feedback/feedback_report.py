#!/usr/bin/env python3
"""Summary of analyst feedback: approval rates, judge-vs-analyst agreement, common corrections.

    python scripts/feedback/feedback_report.py           # summary
    python scripts/feedback/feedback_report.py --list 20 # also the 20 newest feedback rows
    python scripts/feedback/feedback_report.py --json    # raw statistics as JSON
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from feedback import feedback_db_path, get_feedback_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyst feedback report.")
    parser.add_argument("--list", type=int, default=0, metavar="N", help="Also show the N newest feedback rows.")
    parser.add_argument("--json", action="store_true", help="Print the statistics as JSON.")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if not feedback_db_path().exists():
        print(f"No feedback yet ({feedback_db_path()} does not exist).")
        return 0
    store = get_feedback_store()
    stats = store.stats()
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0

    print(f"Feedback database: {store.path}")
    if not stats["total"]:
        print("No feedback yet.")
        return 0
    print(f"\nRatings: {stats['total']}  (up {stats['up']}, down {stats['down']}, approval {stats['approval_rate']:.0%})")
    print("\nBy route:")
    for route, entry in stats["by_route"].items():
        print(f"  {route:<12} {entry['n']:>4} rated   approval {entry['approval_rate']:.0%}")

    agreement = stats["judge_agreement"]
    print("\nLLM judge vs analysts:")
    if agreement["n"]:
        print(f"  {agreement['n']} judged answers, agreement {agreement['agreement_rate']:.0%}")
        print(f"                   analyst up   analyst down")
        print(f"  judge pass       {agreement['pass_up']:>10}   {agreement['pass_down']:>12}")
        print(f"  judge not pass   {agreement['not_pass_up']:>10}   {agreement['not_pass_down']:>12}")
    else:
        print("  no judged answers have been rated yet (use --judge in the chat or judge=true in the API)")

    corrections = stats["corrections"]
    print(f"\nLabel corrections: {corrections['total']}")
    for pair in corrections["top_pairs"]:
        print(f"  {pair['predicted']:<28} -> {pair['corrected']:<28} x{pair['count']}")

    if stats["recent_comments"]:
        print("\nRecent comments:")
        for item in stats["recent_comments"]:
            print(f"  [{item['rating']}] ({item['route']}) {item['comment']}")

    if args.list:
        print(f"\nNewest {args.list} feedback rows:")
        for row in store.list(limit=args.list):
            label = f" -> {row['corrected_label']}" if row["corrected_label"] else ""
            print(f"  #{row['id']} {row['created_at']} [{row['rating']}] {row['route']}: "
                  f"{(row['question'] or '')[:60]}  (pred {row['predicted_label']}{label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
