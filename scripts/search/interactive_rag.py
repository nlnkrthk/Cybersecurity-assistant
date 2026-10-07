#!/usr/bin/env python3
"""Interactive RAG workflow using the existing hybrid retrieval layer."""

import argparse
import sys
from pathlib import Path

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from llm.client import LLMClient
from rag.rag_pipeline import rag_pipeline


def parse_filter_args(filter_args):
    filters = {}
    for entry in filter_args or []:
        field, separator, value = entry.partition("=")
        if not separator or not field.strip() or not value.strip():
            raise ValueError(f"Invalid filter {entry!r}; expected FIELD=VALUE")
        filters[field.strip()] = value.strip()
    return filters


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the APCIRA document-grounded RAG pipeline.")
    parser.add_argument("query", nargs="?", help="Single query to answer without entering interactive mode.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of historical incidents to include as evidence.")
    parser.add_argument("--filter", action="append", default=None, metavar="FIELD=VALUE", help="Metadata filter passed to hybrid search.")
    parser.add_argument("--debug", action="store_true", help="Show hard-filter thresholds, candidate counts, and final incident rates.")
    args = parser.parse_args()

    try:
        filters = parse_filter_args(args.filter)
    except ValueError as exc:
        parser.error(str(exc))

    if args.top_k < 1:
        parser.error("--top-k must be a positive integer")

    llm = LLMClient()

    def display_result(result):
        if args.debug:
            print("\n[HARD FILTER DEBUG]")
            print(f"Metadata filters: {result.get('filters') or 'none'}")
            for condition in result.get("numeric_conditions", []):
                print(
                    f"Numeric: {condition['feature']} {condition['operator']} "
                    f"{condition['value']} {condition['unit']} "
                    f"(source={condition.get('threshold_source', condition.get('source'))}, "
                    f"sample_count={condition.get('sample_count', 'N/A')})"
                )
            print(f"Candidate counts: {result.get('candidate_counts')}")
            for record in result.get("results", []):
                metadata = record.get("metadata", {})
                print(
                    f"{record['incident_id']}: "
                    f"flow_packets_s={metadata.get('flow_packets_s', 'N/A')}"
                )
            print()
        print(result["answer"])

    if args.query:
        result = rag_pipeline(args.query, filters=filters, top_k=args.top_k, llm_client=llm)
        display_result(result)
        return 0

    print("\n" + "=" * 80)
    print("INTERACTIVE RAG")
    print("=" * 80)
    print("Examples:")
    print("  • What historical incidents are similar to DDoS traffic with high packet volume?")
    print("  • How do PortScan incidents differ from benign traffic?")
    print("  • Find evidence for Web Attack patterns and response actions\n")

    while True:
        try:
            query = input("[QUERY] Enter your question or 'quit' to exit:\n> ").strip()
        except KeyboardInterrupt:
            print("\n[EXIT] Goodbye!")
            return 0
        if query.lower() in {"quit", "exit", "q"}:
            print("\n[EXIT] Goodbye!")
            return 0
        if not query:
            print("[WARN] Please enter a question.")
            continue
        result = rag_pipeline(query, filters=filters, top_k=args.top_k, llm_client=llm)
        print("\n" + "=" * 80)
        display_result(result)
        print("=" * 80)


if __name__ == "__main__":
    raise SystemExit(main())
