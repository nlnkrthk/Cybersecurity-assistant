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
back to earlier flows. Commands: new, trace on|off, quit.
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
    args = parser.parse_args()

    print("Loading assistant (models load on first use)...")
    try:
        assistant = Assistant()
    except Exception as exc:
        print(f"[ERROR] Could not start the assistant: {exc}")
        return 1

    state = {"trace": not args.no_trace, "thread": Assistant.new_thread_id()}

    def answer(question: str) -> None:
        start = time.time()
        result = None
        if state["trace"]:
            print(f"\n{TRACE_HEADER}")
        try:
            for kind, payload in assistant.stream(question, state["thread"]):
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
    print("\nCommands: new (start a new conversation), trace on|off, quit")

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
        answer(question)


if __name__ == "__main__":
    raise SystemExit(main())
