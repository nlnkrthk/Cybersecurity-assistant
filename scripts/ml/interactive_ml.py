#!/usr/bin/env python3
"""Interactive tester for the ML service (attack type, intrusion risk, anomaly, priority).

Type a question with whatever flow metrics you know, for example:
    > A flow lasted 46 microseconds with 1 forward and 1 backward packet, 2 bytes sent and 6 bytes back
    > flow_duration=2s total_fwd_packets=10 total_bwd_packets=8 total_len_fwd=1.2KB total_len_bwd=9.6KB

The metrics are extracted (by the LLM, or by a rule-based parser with --no-llm),
shown to you, normalized (derive + fill + flag), and run through the models.
If key metrics are missing the tool asks a follow-up question; your next answer
is merged with the metrics already given.

Commands: incident <ID>, example <attack type>, profile <attack type>, types,
          json on|off, mode explain|predict, reset, help, quit
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from services.feature_input import (  # noqa: E402
    FEATURE_NAMES,
    FlowFeatures,
    FlowInputError,
    get_incident_features,
)
from services.ml_service import get_ml_service  # noqa: E402
from services.prioritization import prioritize  # noqa: E402

CLEANED_CSV = project_root / "data" / "cleaned_network_data" / "cleaned_network_data.csv"
LINE = "=" * 80

EXAMPLES = [
    "A flow lasted 46 microseconds with 1 forward and 1 backward packet, 2 bytes sent and 6 bytes received",
    "flow_duration=2s total_fwd_packets=10 total_bwd_packets=8 total_len_fwd=1.2KB total_len_bwd=9.6KB",
    "Connection of 5.3 seconds, 4 packets out and 0 back, 24 bytes forward, nothing returned, init window 229",
    "incident INC-000042",
    "example DDoS",
]


# ============================================================================
# Metric extraction
# ============================================================================
# The worked example matters for smaller models: without it, gpt-oss-20b often answered {}
# even after reasoning out the right values.
EXTRACTION_SYSTEM_PROMPT = """You extract network-flow metrics from a security analyst's message.
Reply with ONLY a JSON object that maps schema field names to the values stated in the message.
Include every metric the message states explicitly; never guess or invent values. Values may be numbers
in the schema's unit or strings with a unit (e.g. "2 s", "1.5 KB", "43k"). Use an empty object only
when the message states no metric at all.

Example: "it lasted 3 ms, 4 packets out and 2 back, 300 bytes sent" ->
{"flow_duration": "3 ms", "total_fwd_packets": 4, "total_bwd_packets": 2, "total_len_fwd": "300 bytes"}

Schema (field: description):
"""


def _schema_text() -> str:
    fields = FlowFeatures.model_json_schema()["properties"]
    return "\n".join(f"- {name}: {spec.get('description', '')}" for name, spec in fields.items())


def extract_with_llm(llm, query: str) -> Dict[str, Any]:
    text = llm.generate(
        f"Message:\n{query}",
        system_prompt=EXTRACTION_SYSTEM_PROMPT + _schema_text(),
        max_tokens=1000,
    )
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    data = json.loads(match.group(0))
    return {k: v for k, v in data.items() if v is not None}


_NUM = r"(\d[\d,]*(?:\.\d+)?|\.\d+)"
_RULE_PATTERNS = [
    # explicit key=value / key: value
    (re.compile(r"\b([a-z_]+)\s*[=:]\s*([^\s,;]+(?:\s*(?:us|µs|ms|s|sec|seconds|min|minutes|kb|mb|gb|bytes|b|k|pps)(?!\w))?)", re.I), None),
    (re.compile(rf"(?:duration|lasted|lasting|lasts)\D{{0,12}}{_NUM}\s*(microseconds?|us|µs|milliseconds?|ms|seconds?|secs?|s|minutes?|mins?)\b", re.I), "flow_duration"),
    # any other "<number> <time unit>" is taken as the flow duration
    (re.compile(rf"{_NUM}\s*(microseconds?|µs|milliseconds?|ms|seconds?|secs?|minutes?|mins?)\b", re.I), "flow_duration"),
    # "N forward and M backward packets"
    (re.compile(rf"{_NUM}\s*(?:forward|fwd|outgoing|out)\s+and\s+{_NUM}\s*(?:backward|bwd|back|incoming|in)\s*(?:packets?|pkts?)", re.I), ("total_fwd_packets", "total_bwd_packets")),
    (re.compile(rf"{_NUM}\s*(?:forward|fwd|outgoing|out)\s*(?:packets?|pkts?)", re.I), "total_fwd_packets"),
    (re.compile(rf"{_NUM}\s*(?:packets?|pkts?)\s*(?:forward|fwd|out|outgoing|sent)", re.I), "total_fwd_packets"),
    (re.compile(rf"{_NUM}\s*(?:backward|bwd|incoming|reply|response|back)\s*(?:packets?|pkts?)", re.I), "total_bwd_packets"),
    (re.compile(rf"{_NUM}\s*(?:packets?|pkts?)\s*(?:backward|bwd|back|in|incoming|received)", re.I), "total_bwd_packets"),
    (re.compile(rf"{_NUM}\s*(kb|mb|gb|bytes|b)\s*(?:forward|fwd|sent|out|outgoing)", re.I), "total_len_fwd"),
    (re.compile(rf"{_NUM}\s*(kb|mb|gb|bytes|b)\s*(?:backward|bwd|back|received|returned|in|incoming)", re.I), "total_len_bwd"),
    (re.compile(rf"{_NUM}\s*([km]?)\s*(?:packets/s|pps|packets per second)", re.I), "flow_packets_s"),
    (re.compile(rf"init(?:ial)?\s*(?:tcp\s*)?window\D{{0,12}}{_NUM}", re.I), "init_win_bytes_forward"),
]
_FLAG_RE = re.compile(r"\b(syn|ack|rst|psh|fin)\s+flags?\b", re.I)


def extract_with_rules(query: str) -> Dict[str, Any]:
    """Fallback parser: key=value pairs plus a few common phrasings."""
    metrics: Dict[str, Any] = {}
    for pattern, feature in _RULE_PATTERNS:
        for match in pattern.finditer(query):
            if feature is None:
                key, value = match.group(1), match.group(2)
                metrics.setdefault(key, value)
            elif isinstance(feature, tuple):
                for name, value in zip(feature, match.groups()):
                    metrics.setdefault(name, value)
            else:
                value = match.group(1) + (" " + match.group(2) if match.lastindex and match.lastindex >= 2 else "")
                metrics.setdefault(feature, value.strip())
    for match in _FLAG_RE.finditer(query):
        metrics.setdefault(f"{match.group(1).lower()}_flag_count", 1)
    if re.search(r"nothing (?:came )?(?:back|returned)|no (?:reply|response|backward packets)", query, re.I):
        metrics.setdefault("total_bwd_packets", 0)
        metrics.setdefault("total_len_bwd", 0)
    return metrics


# ============================================================================
# Display
# ============================================================================
def show_metrics(metrics: Dict[str, Any], source: str) -> None:
    print(f"\n[EXTRACTED METRICS] ({source})")
    if not metrics:
        print("  (none)")
    for key, value in metrics.items():
        print(f"  {key:26} {value}")


def show_result(result: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, default=str))
        return

    inp = result["input"]
    print(f"\n[INPUT] quality={inp['input_quality']}  known={inp['known_count']}/{len(FEATURE_NAMES)}  "
          f"provided={len(inp['provided'])}  derived={len(inp['derived'])}  filled={len(inp['filled'])}")
    if inp["derived"]:
        print(f"  derived: {', '.join(inp['derived'])}")
    for warning in inp["warnings"]:
        print(f"  ! {warning}")

    if result["status"] == "needs_follow_up":
        print("\n[FOLLOW-UP NEEDED]")
        print(f"  {result['follow_up_question']}")
        print("  (Your next message will be added to the metrics above. Type 'reset' to start over.)")
        return

    attack, risk, anomaly = result["attack_type"], result["risk"], result["anomaly"]
    print(f"\n[ATTACK TYPE] {attack['prediction']}  (confidence {attack['confidence']:.0%})")
    for alt in attack["alternatives"]:
        print(f"  alternative: {alt['attack_type']} ({alt['probability']:.0%})")
    print(f"[RISK]        {risk['probability']:.0%}  level={risk['level']}  intrusion={'YES' if risk['is_intrusion'] else 'no'}")
    print(f"[ANOMALY]     level={anomaly['level']}  score={anomaly['score']}  "
          f"(more unusual than {anomaly['benign_percentile']:.1f}% of normal traffic)")
    for feature in anomaly["most_unusual_features"]:
        print(f"  unusual: {feature['text']}")
    print(f"[CONFIDENCE]  {result['overall_confidence']}")

    priority = prioritize(result)
    print(f"\n[PRIORITY]    {priority['priority']} {priority['label']} (score {priority['score']}) - {priority['action']}")
    print(f"  {priority['reason']}")
    for override in priority["overrides_applied"]:
        print(f"  override: {override}")
    if priority["confidence_note"]:
        print(f"  note: {priority['confidence_note']}")

    explanation = result.get("explanation")
    if explanation:
        print("\n[WHY: ATTACK TYPE]")
        for reason in explanation["attack_type_reasons"] or [{"text": "no single dominant feature"}]:
            print(f"  + {reason['text']}")
        print("[WHY: RISK]")
        for reason in explanation["risk_reasons"]:
            print(f"  {'+' if reason['shap'] > 0 else '-'} {reason['text']}  ({reason['effect']})")
        print("\n[SUMMARY]")
        print(f"  {explanation['summary']}")


def show_profile(service, attack_type: str) -> None:
    profile = service.attack_profile(attack_type)
    print(f"\n[PROFILE] {attack_type}")
    print(f"  {profile['summary']}")
    for f in profile["top_features"]:
        direction = "similar to normal" if f["direction_vs_normal"] == "similar" else f"{f['direction_vs_normal']} than normal"
        print(f"  - {f['display_name']}: {direction} "
              f"(median {f['attack_median']:,.2f} vs {f['benign_median']:,.2f}), SHAP share {f['shap_share']:.0%}")


def print_help(service) -> None:
    print("\nType a question with the flow metrics you know, or a command:")
    print("  incident <ID>           analyse a historical incident, e.g. incident INC-000042")
    print("  example <attack type>   analyse a real flow of that type, e.g. example PortScan")
    print("  profile <attack type>   typical characteristics of an attack type")
    print("  types                   list attack types")
    print("  mode explain|predict    with or without SHAP explanations (default explain)")
    print("  json on|off             show raw JSON output")
    print("  reset                   forget metrics from a pending follow-up")
    print("  quit                    exit")
    print("\nExamples:")
    for example in EXAMPLES:
        print(f"  • {example}")
    print(f"\nAttack types: {', '.join(service.list_attack_types())}")


def _match_attack_type(service, text: str) -> Optional[str]:
    wanted = text.strip().lower()
    for name in service.list_attack_types():
        if name.lower() == wanted:
            return name
    matches = [n for n in service.list_attack_types() if wanted in n.lower()]
    return matches[0] if len(matches) == 1 else None


# ============================================================================
# Main loop
# ============================================================================
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Interactive tester for the ML service.")
    parser.add_argument("query", nargs="?", help="Single query to run without entering interactive mode.")
    parser.add_argument("--no-llm", action="store_true", help="Use the rule-based metric parser instead of the LLM.")
    parser.add_argument("--mode", choices=["explain", "predict"], default="explain",
                        help="explain = with SHAP reasons (default); predict = models only.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON results.")
    args = parser.parse_args()

    llm = None
    if not args.no_llm:
        try:
            from llm.client import LLMClient

            llm = LLMClient()
            if not llm.api_key:
                print("[WARN] No LLM API key configured; using the rule-based parser.")
                llm = None
        except Exception as exc:
            print(f"[WARN] LLM unavailable ({exc}); using the rule-based parser.")
            llm = None

    print("Loading ML models...")
    try:
        service = get_ml_service()
    except Exception as exc:
        print(f"[ERROR] Could not load the ML service: {exc}")
        return 1

    state = {"mode": args.mode, "json": args.json, "pending": {}}
    flows_df = None

    def analyse(flow, source_label: str) -> None:
        runner = service.explain if state["mode"] == "explain" else service.predict
        result = runner(flow)
        print(f"\n{LINE}\n{source_label}")
        show_result(result, state["json"])
        print(LINE)

    def handle(query: str) -> None:
        nonlocal flows_df
        lower = query.lower()

        if lower in {"help", "?"}:
            print_help(service)
            return
        if lower == "types":
            print("\n" + "\n".join(f"  {t}" for t in service.list_attack_types()))
            return
        if lower == "reset":
            state["pending"] = {}
            print("[OK] Pending metrics cleared.")
            return
        if lower.startswith("mode "):
            mode = lower.split(maxsplit=1)[1]
            if mode in {"explain", "predict"}:
                state["mode"] = mode
                print(f"[OK] Mode: {mode}")
            else:
                print("[WARN] Use: mode explain | mode predict")
            return
        if lower.startswith("json "):
            state["json"] = lower.endswith("on")
            print(f"[OK] JSON output {'on' if state['json'] else 'off'}")
            return
        if lower.startswith("profile "):
            attack = _match_attack_type(service, query.split(maxsplit=1)[1])
            if not attack or attack == "BENIGN":
                print("[WARN] Unknown attack type. Type 'types' to list them.")
            else:
                show_profile(service, attack)
            return
        if lower.startswith("incident") or re.fullmatch(r"inc-?\d+", lower):
            incident_id = query.split(maxsplit=1)[1] if " " in query else query
            incident = get_incident_features(incident_id)
            port = f", destination port {incident.dst_port}" if incident.dst_port is not None else ""
            analyse(incident.flow, f"INCIDENT {incident.incident_id} (historical label: {incident.historical_label}{port})")
            return
        if lower.startswith("example "):
            attack = _match_attack_type(service, query.split(maxsplit=1)[1])
            if not attack:
                print("[WARN] Unknown attack type. Type 'types' to list them.")
                return
            if flows_df is None:
                import pandas as pd

                flows_df = pd.read_csv(CLEANED_CSV)
            row = flows_df[flows_df["label"] == attack].sample(1).iloc[0]
            analyse({n: float(row[n]) for n in FEATURE_NAMES}, f"EXAMPLE: a real {attack} flow (all 24 metrics)")
            return

        # Natural-language query with metrics
        if llm is not None:
            try:
                metrics, source = extract_with_llm(llm, query), "LLM"
            except Exception as exc:
                print(f"[WARN] LLM extraction failed ({exc}); using the rule-based parser.")
                metrics, source = extract_with_rules(query), "rules"
            if not metrics:
                # Smaller models occasionally return an empty object; the rules may still find metrics.
                metrics, source = extract_with_rules(query), "rules (LLM found none)"
        else:
            metrics, source = extract_with_rules(query), "rules"

        if state["pending"]:
            metrics = {**state["pending"], **metrics}
            source += " + previous message"
        show_metrics(metrics, source)
        if not metrics:
            print("[WARN] No flow metrics found. Type 'help' for examples.")
            return

        runner = service.explain if state["mode"] == "explain" else service.predict
        result = runner(metrics)
        state["pending"] = metrics if result["status"] == "needs_follow_up" else {}
        print(f"\n{LINE}")
        show_result(result, state["json"])
        print(LINE)

    def safe_handle(query: str) -> None:
        try:
            handle(query)
        except (FlowInputError, KeyError, ValueError) as exc:
            print(f"[ERROR] {exc}")

    if args.query:
        safe_handle(args.query)
        return 0

    print("\n" + LINE)
    print("INTERACTIVE ML SERVICE")
    print(LINE)
    print(f"Metric extraction: {'LLM (' + llm.model + ')' if llm else 'rule-based parser'}   Mode: {state['mode']}")
    print_help(service)
    print()

    while True:
        try:
            query = input("\n[QUERY] Enter flow details, a command, or 'quit':\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[EXIT] Goodbye!")
            return 0
        if query.lower() in {"quit", "exit", "q"}:
            print("\n[EXIT] Goodbye!")
            return 0
        if not query:
            print("[WARN] Please enter something (or 'help').")
            continue
        safe_handle(query)


if __name__ == "__main__":
    raise SystemExit(main())
