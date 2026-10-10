#!/usr/bin/env python3
"""Interactive multi-agent incident-response assistant (LangGraph supervisor + 4 agents).

Ask anything in plain language, for example:
    > What attack is incident INC-004393?
    > Is this flow suspicious? It lasted 2 seconds, 10 packets out, 8 back
    > Has an SSH brute force attack happened before?
    > Investigate incident INC-000035 and tell me how to respond

While a question is answered, the agent trace is printed live: the supervisor's
route, each agent that runs, every tool it calls with a one-line result, and timings.

The conversation is remembered, so you can answer follow-up questions and refer
back to earlier flows. With --judge (or 'judge on') every answer is also graded by
LLM-as-Judge.

Rate the last answer with 'feedback up [comment]', 'feedback down [comment]' or
'feedback correct <attack type>' (e.g. feedback correct Web Attack - XSS);
'feedback stats' shows the summary. Corrections appear in later answers about the same
incident and are used when the classifier is retrained (scripts/feedback/).

Commands: new, trace on|off, judge on|off, feedback ..., quit.
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")  # hide embedding progress bars
warnings.filterwarnings("ignore")

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from agents.graph import Assistant  # noqa: E402
from agents.trace import format_event  # noqa: E402

LINE = "=" * 80
EXAMPLES = [
    "What attack is incident INC-004393?",
    "Is this flow suspicious? It lasted 2 seconds, the client sent 10 packets and got 8 back",
    "Has an SSH brute force attack happened before?",
    "Which is most urgent: INC-000001, INC-000035, INC-002545, INC-000005?",
    "Investigate incident INC-000035 and tell me how to respond",
    "What does a PortScan look like?",
]


TRACE_HEADER = "-" * 34 + " agent trace " + "-" * 33


def handle_feedback(assistant, state, args: str) -> None:
    """feedback up|down [comment] / feedback correct <attack type> / feedback stats"""
    from feedback import get_feedback_store
    from feedback.service import FeedbackError, record_feedback

    command, _, rest = args.partition(" ")
    command = command.lower()
    if command == "stats":
        print_feedback_stats(get_feedback_store().stats())
        return
    if command not in {"up", "down", "correct"}:
        print("[INFO] Use: feedback up [comment] | feedback down [comment] | feedback correct <attack type> | feedback stats")
        return
    if not state["turn"]:
        print("[INFO] Ask a question first; feedback applies to the last answer.")
        return
    if command == "correct" and not rest.strip():
        print("[INFO] Give the correct attack type, e.g. feedback correct Web Attack - XSS")
        return
    try:
        row = record_feedback(
            assistant,
            thread_id=state["thread"],
            turn_id=state["turn"],
            rating="down" if command == "correct" else command,
            comment=None if command == "correct" else rest,
            corrected_label=rest.strip() if command == "correct" else None,
        )
    except (FeedbackError, KeyError) as exc:
        print(f"[ERROR] {exc}")
        return
    label = f", corrected label: {row['corrected_label']}" if row["corrected_label"] else ""
    print(f"[OK] Feedback saved (rating {row['rating']}{label}).")


def print_feedback_stats(stats) -> None:
    if not stats["total"]:
        print("[INFO] No feedback yet.")
        return
    print(f"[FEEDBACK] {stats['total']} rating(s): {stats['up']} up, {stats['down']} down "
          f"(approval {stats['approval_rate']:.0%})")
    for route, entry in stats["by_route"].items():
        print(f"  {route:<12} {entry['n']:>3} rated, approval {entry['approval_rate']:.0%}")
    agreement = stats["judge_agreement"]
    if agreement["n"]:
        print(f"  judge vs analyst agreement: {agreement['agreement_rate']:.0%} of {agreement['n']} judged answers")
    for pair in stats["corrections"]["top_pairs"][:5]:
        print(f"  correction: {pair['predicted']} -> {pair['corrected']} (x{pair['count']})")


def print_judgement(judgement) -> None:
    """LLM-as-Judge scores under the answer."""
    if judgement["verdict"] == "error":
        print(f"[JUDGE] not scored: {judgement.get('error')}")
        return
    print(f"[JUDGE] {judgement['overall']:.1f}/5  verdict: {judgement['verdict'].upper()}")
    for name in judgement["applicable"]:
        print(f"  {name:<14} {judgement['scores'][name]}/5  {judgement['reasons'][name]}")
    for claim in judgement["unsupported_claims"]:
        print(f"  ! unsupported: {claim}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(level=logging.WARNING)
    # Model-download notices from the embedding model would interrupt the live trace.
    for noisy in ("huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Interactive multi-agent incident-response assistant.")
    parser.add_argument("query", nargs="?", help="Single question to answer without entering interactive mode.")
    parser.add_argument("--no-trace", action="store_true", help="Hide the live agent trace.")
    parser.add_argument("--judge", action="store_true", help="Grade every answer with LLM-as-Judge (one extra LLM call).")
    args = parser.parse_args()

    print("Loading assistant (models load on first use)...")
    try:
        assistant = Assistant()
    except Exception as exc:
        print(f"[ERROR] Could not start the assistant: {exc}")
        return 1

    state = {"trace": not args.no_trace, "judge": args.judge, "thread": Assistant.new_thread_id(), "turn": None}

    def answer(question: str) -> None:
        start = time.time()
        result = None
        if state["trace"]:
            print(f"\n{TRACE_HEADER}")
        try:
            for kind, payload in assistant.stream(question, state["thread"], judge=state["judge"]):
                if kind == "result":
                    result = payload
                elif state["trace"]:
                    line = format_event(payload)
                    if line:
                        print(line, flush=True)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            return
        if result is None:
            print("[ERROR] No answer was produced.")
            return
        print(f"\n{LINE}")
        print(result["answer"])
        print(LINE)
        if result.get("judgement"):
            print_judgement(result["judgement"])
        state["turn"] = result.get("turn_id")
        suffix = "  (answer the question above to continue)" if result["needs_follow_up"] else ""
        print(f"[{time.time() - start:.1f}s total]{suffix}")

    if args.query:
        answer(args.query)
        return 0

    print("\n" + LINE)
    print("INCIDENT RESPONSE ASSISTANT  (supervisor + Classification, Risk & Priority, Incident Search, Response)")
    print(LINE)
    print("Examples:")
    for example in EXAMPLES:
        print(f"  - {example}")
    print("\nCommands: new (start a new conversation), trace on|off, judge on|off, quit")
    print("Feedback on the last answer: feedback up|down [comment], feedback correct <attack type>, feedback stats")

    while True:
        try:
            question = input("\n[YOU] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[EXIT] Goodbye!")
            return 0
        lower = question.lower()
        if lower in {"quit", "exit", "q"}:
            print("\n[EXIT] Goodbye!")
            return 0
        if not question:
            continue
        if lower == "new":
            state["thread"] = Assistant.new_thread_id()
            print("[OK] New conversation started.")
            continue
        if lower.startswith("trace "):
            state["trace"] = lower.endswith("on")
            print(f"[OK] Agent trace {'on' if state['trace'] else 'off'}")
            continue
        if lower.startswith("judge "):
            state["judge"] = lower.endswith("on")
            print(f"[OK] LLM-as-Judge {'on' if state['judge'] else 'off'}")
            continue
        if lower == "feedback" or lower.startswith("feedback "):
            handle_feedback(assistant, state, question[len("feedback"):].strip())
            continue
        answer(question)


if __name__ == "__main__":
    raise SystemExit(main())
