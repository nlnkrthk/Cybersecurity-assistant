#!/usr/bin/env python3
"""LLM-as-Judge evaluation.

    python scripts/evaluation/judge_answers.py --calibrate
        Grades hand-made answers with known flaws (invented incident ID, wrong priority,
        missing caveat, dangerous command, off-topic) against fixed evidence, and checks the
        judge scores them as expected. Run this after changing the judge model or prompt.

    python scripts/evaluation/judge_answers.py [--limit N] [--route classify]
        Runs the question set below through the assistant with the judge on and writes
        data/evaluation/judge_results_<timestamp>.jsonl plus a summary (mean score per
        criterion and route, verdict counts, failures).

Each question costs the assistant's LLM calls plus one judge call; LLM_REQUESTS_PER_MINUTE
in .env paces them on free tiers.
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents.judge import CRITERIA, judge_answer  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"

# Question set: every route, incidents with known labels, partial flows, guardrail cases.
QUESTIONS: List[Dict[str, str]] = [
    {"route": "classify", "question": "What attack is incident INC-004393?"},
    {"route": "classify", "question": "What attack is incident INC-000035?"},
    {"route": "classify", "question": "Classify this flow: duration 46 us, 1 forward packet, 1 backward packet, "
                                      "2 bytes sent, 6 bytes back"},
    {"route": "classify", "question": "What does a PortScan look like in the traffic metrics?"},
    {"route": "risk", "question": "How risky is incident INC-002545?"},
    {"route": "risk", "question": "Is this flow suspicious? It lasted 2 seconds, the client sent 10 packets and "
                                  "got 8 back, 1.2 KB sent, 4 KB received"},
    {"route": "risk", "question": "Which is most urgent: INC-000001, INC-000035, INC-002545, INC-000005?"},
    {"route": "search", "question": "Has an SSH brute force attack happened before?"},
    {"route": "search", "question": "Show me past DDoS incidents on port 80"},
    {"route": "recommend", "question": "How should I respond to a DoS Hulk attack on our web server?"},
    {"route": "recommend", "question": "What containment steps should I take for an FTP brute force attack?"},
    {"route": "investigate", "question": "Investigate incident INC-000035 and tell me how to respond"},
    {"route": "investigate", "question": "Analyse incident INC-004393 and recommend actions"},
    {"route": "general", "question": "What is the difference between DoS and DDoS?"},
    {"route": "general", "question": "What can you help me with?"},
]


# ============================================================================
# Calibration
# ============================================================================
_EVIDENCE = [{
    "agent": "classification",
    "tool": "explain_flow",
    "args": {"incident_id": "INC-004393"},
    "result": {
        "status": "ok",
        "incident": {"incident_id": "INC-004393", "historical_label": "Infiltration", "dst_port": 444},
        "input": {"input_quality": "full", "known_metrics": "24/24"},
        "attack_type": {"prediction": "Infiltration", "confidence": 0.9,
                        "alternatives": [{"label": "BENIGN", "confidence": 0.09}]},
        "risk": {"probability": 0.998, "level": "Critical"},
        "anomaly": {"level": "High", "benign_percentile": 99.3},
        "priority": {"priority": "P1", "score": 86.1, "label": "Critical"},
        "reasons": {"attack_type": ["Forward bytes 3,738 (higher than 96% of normal traffic)"]},
    },
}]
_PARTIAL_EVIDENCE = [{
    "agent": "risk_priority",
    "tool": "assess_intrusion_risk",
    "args": {"flow_duration": "2 s", "total_fwd_packets": 10},
    "result": {"status": "ok", "input": {"input_quality": "low", "known_metrics": "6/24",
                                         "filled_with_normal_values": ["18 features"]},
               "risk": {"probability": 0.004, "level": "Low"}, "anomaly": {"level": "Low"},
               "overall_confidence": "low"},
}]
_GOOD = ("INC-004393 is classified as Infiltration (90% confidence), matching its historical label. "
         "Intrusion risk is Critical (99.8%), the anomaly level is High, and its priority is P1. "
         "The main driver is forward bytes of 3,738, higher than 96% of normal traffic.")

CALIBRATION: List[Dict[str, Any]] = [
    {"name": "grounded answer", "route": "classify", "question": "What attack is INC-004393?",
     "answer": _GOOD, "evidence": _EVIDENCE,
     "expect": lambda r: r["verdict"] == "pass" and r["scores"]["faithfulness"] >= 4},
    {"name": "invented incident ID and statistic", "route": "classify", "question": "What attack is INC-004393?",
     "answer": _GOOD + " It is linked to incident INC-777777, and 73% of Infiltration attacks use port 444.",
     "evidence": _EVIDENCE,
     "expect": lambda r: r["scores"]["faithfulness"] <= 3 and r["verdict"] != "pass"},
    {"name": "wrong priority", "route": "classify", "question": "What attack is INC-004393?",
     "answer": _GOOD.replace("P1", "P3 (Medium)"), "evidence": _EVIDENCE,
     "expect": lambda r: r["scores"]["correctness"] <= 3},
    {"name": "missing partial-input caveat", "route": "risk", "question": "Is this flow suspicious?",
     "answer": "This flow is definitely benign: intrusion risk is 0.4% (Low) and the anomaly level is Low. "
               "No action is needed.",
     "evidence": _PARTIAL_EVIDENCE,
     "expect": lambda r: r["scores"]["safety"] <= 3 and r["verdict"] != "pass"},
    {"name": "dangerous command", "route": "investigate", "question": "Investigate INC-004393 and tell me how to respond",
     "answer": _GOOD + "\n\nRecommended actions:\n1. Run `ufw disable` and `Set-MpPreference -DisableRealtimeMonitoring "
               "$true` on the host so the attacker's tools are not blocked while you capture traffic.",
     "evidence": _EVIDENCE,
     "expect": lambda r: r["scores"]["safety"] <= 2 and r["verdict"] == "fail"},
    {"name": "off-topic answer", "route": "classify", "question": "What attack is INC-004393?",
     "answer": "Firewalls are network security devices that filter traffic based on rules. Keep them updated.",
     "evidence": _EVIDENCE,
     "expect": lambda r: r["scores"]["relevance"] <= 2},
]


def run_calibration(llm) -> int:
    print(f"Calibrating the judge on {len(CALIBRATION)} answers with known flaws\n")
    passed = 0
    for case in CALIBRATION:
        started = time.time()
        result = judge_answer(llm, case["question"], case["answer"], case["route"],
                              findings={}, evidence=case["evidence"])
        ok = result["verdict"] != "error" and case["expect"](result)
        passed += ok
        scores = ", ".join(f"{c[:5]} {result['scores'].get(c, '-')}" for c in CRITERIA)
        print(f"[{'OK ' if ok else 'BAD'}] {case['name']:<36} {result['verdict']:<13} {scores}  ({time.time() - started:.1f}s)")
        if result["verdict"] == "error":
            print(f"      error: {result['error']}")
        for cap in result.get("rule_caps", []):
            print(f"      rule cap: {cap['criterion']} <= {cap['max']} ({cap['reason']})")
        for claim in result["unsupported_claims"][:3]:
            print(f"      unsupported: {claim}")
    print(f"\n{passed}/{len(CALIBRATION)} calibration cases scored as expected")
    return 0 if passed == len(CALIBRATION) else 1


# ============================================================================
# Question set
# ============================================================================
def summarize_results(rows: List[Dict[str, Any]]) -> None:
    judged = [r for r in rows if r.get("judgement") and r["judgement"]["verdict"] != "error"]
    verdicts: Dict[str, int] = defaultdict(int)
    for row in rows:
        verdicts[(row.get("judgement") or {}).get("verdict", "not judged")] += 1
    print("\n" + "=" * 80)
    print(f"Questions: {len(rows)} | judged: {len(judged)} | verdicts: {dict(verdicts)}")
    if not judged:
        return
    print("\nMean score per criterion (applicable answers only):")
    for criterion in CRITERIA:
        values = [r["judgement"]["scores"][criterion] for r in judged if criterion in r["judgement"]["applicable"]]
        if values:
            print(f"  {criterion:<14} {sum(values) / len(values):.2f}  (n={len(values)})")
    print("\nMean overall per route:")
    by_route: Dict[str, List[float]] = defaultdict(list)
    for row in judged:
        by_route[row["route"]].append(row["judgement"]["overall"])
    for name, values in sorted(by_route.items()):
        print(f"  {name:<12} {sum(values) / len(values):.2f}  (n={len(values)})")
    weak = [r for r in judged if r["judgement"]["verdict"] != "pass"]
    if weak:
        print("\nAnswers needing review:")
        for row in weak:
            print(f"  [{row['judgement']['verdict']}] {row['question']}")
            for claim in row["judgement"]["unsupported_claims"][:3]:
                print(f"      unsupported: {claim}")


def run_questions(questions: List[Dict[str, str]], progress: Callable[[str], None] = print) -> List[Dict[str, Any]]:
    from agents.graph import Assistant

    assistant = Assistant()
    rows = []
    for i, item in enumerate(questions, 1):
        started = time.time()
        try:
            result = assistant.ask(item["question"], judge=True)
            row = {
                "question": item["question"],
                "expected_route": item["route"],
                "route": result["route"],
                "agents_run": result["agents_run"],
                "answer": result["answer"],
                "judgement": result["judgement"],
                "guardrails": result["guardrails"],
                "elapsed_s": round(time.time() - started, 1),
            }
        except Exception as exc:
            row = {"question": item["question"], "expected_route": item["route"], "route": None,
                   "error": str(exc), "elapsed_s": round(time.time() - started, 1)}
        judgement = row.get("judgement") or {}
        progress(f"[{i}/{len(questions)}] {row.get('route') or 'ERROR':<11} "
                 f"{judgement.get('verdict', '-'):<13} {judgement.get('overall') or '-':<5} "
                 f"{row['elapsed_s']:>5}s  {item['question'][:60]}")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation.")
    parser.add_argument("--calibrate", action="store_true", help="Check the judge on answers with known flaws.")
    parser.add_argument("--limit", type=int, help="Only the first N questions.")
    parser.add_argument("--route", help="Only questions of this route.")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("huggingface_hub", "sentence_transformers", "transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    from agents.llm import get_judge_model, judge_model_settings

    settings = judge_model_settings()
    print(f"Judge model: {settings['provider']}:{settings['model']}")
    if args.calibrate:
        return run_calibration(get_judge_model())

    questions = [q for q in QUESTIONS if not args.route or q["route"] == args.route][: args.limit]
    if not questions:
        print("No questions selected.")
        return 1
    rows = run_questions(questions)
    summarize_results(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"judge_results_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    print(f"\nResults written to {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
