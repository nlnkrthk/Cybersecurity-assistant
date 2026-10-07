#!/usr/bin/env python3
"""
Interactive hybrid search - test hybrid search with your own queries.

Supports both interactive terminal usage and non-interactive single-query runs.
For example:
  python scripts/search/interactive_hybrid_search.py
  python scripts/search/interactive_hybrid_search.py "DDoS attack high packet volume"
"""

import argparse
import sys
from pathlib import Path

# Configure truststore BEFORE importing any HTTPS libraries
print("Initializing SSL with truststore...")
try:
    import truststore
    truststore.inject_into_ssl()
    print("[OK] Truststore SSL context injected")
except ImportError:
    print("[WARN] truststore not installed")
except Exception as e:
    print(f"[WARN] Truststore injection failed: {e}")

import pandas as pd
import logging

logging.basicConfig(
    level=logging.WARNING,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Add src/ to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from config import CLEANED_INCIDENTS_FILE
from retrieval.embedder import IncidentEmbedder
from retrieval.bm25_index import BM25Index
from retrieval.chroma_store import ChromaVectorStore
from retrieval.hybrid import HybridSearcher
from retrieval.metadata import build_incident_metadata, validate_incident_frame


def format_value(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def print_intro():
    print("\n" + "=" * 80)
    print("INTERACTIVE HYBRID SEARCH")
    print("=" * 80)
    print("\nSearch combines:")
    print("  * BM25: Exact keyword matching")
    print("  * Vector: Semantic similarity (384D embeddings)")
    print("  * Hybrid: RRF fusion of both (best of both)")
    print("\nExamples:")
    print("  * 'DDoS attack high packet volume'")
    print("  * 'PortScan SYN multiple destinations'")
    print("  * 'Infiltration data exfiltration'")
    print("  * 'Web Attack SQL Injection'")
    print("  * 'Bot attack traffic patterns'\n")
    print("Optional metadata filters use --filter FIELD=VALUE, for example:")
    print("  python scripts/search/interactive_hybrid_search.py --filter label=PortScan")
    print("  python scripts/search/interactive_hybrid_search.py --filter dst_port=80 --debug")


def display_results(
    query: str,
    df: pd.DataFrame,
    results: dict,
    debug: bool = False,
) -> None:
    records = df.set_index("incident_id", drop=False)

    def print_record(rank: int, incident_id: str, score_label: str, score: float, text: str):
        if incident_id not in records.index:
            raise ValueError(
                f"Retrieved incident ID {incident_id!r} is missing from the validated Parquet data"
            )
        row = records.loc[incident_id]
        port = format_value(row["dst_port"])
        print(f"\n  {rank}. {incident_id} ({score_label}: {score:.4f})")
        print(f"     Type: {row['label']}")
        print(f"     Destination port: {port}")
        print(f"     Text: {text[:120]}...")

    print(f"\nSearching for: '{query}'")
    print("-" * 80)

    print("\n[BM25] KEYWORD MATCH")
    if results['bm25']:
        for i, (incident_id, score) in enumerate(results['bm25'], 1):
            text = str(records.loc[incident_id, "incident_text"])
            print_record(i, incident_id, "score", score, text)
    else:
        print("  No keyword matches found")

    print("\n[VECTOR] SEMANTIC SIMILARITY (lower distance is better)")
    if results['vector']:
        for i, (incident_id, distance, text) in enumerate(results['vector'], 1):
            print_record(i, incident_id, "distance", distance, text)
    else:
        print("  No semantic matches found")

    print("\n[HYBRID] RRF FUSION")
    if results['fused']:
        for i, (incident_id, rrf_score) in enumerate(results['fused'], 1):
            text = str(records.loc[incident_id, "incident_text"])
            print_record(i, incident_id, "RRF score", rrf_score, text)
    else:
        print("  No hybrid matches found")

    if debug:
        trace = results["debug"]
        filter_summary = trace.get("metadata_filter")
        if filter_summary:
            print("\n[DEBUG] METADATA FILTER")
            print(f"  Filters: {filter_summary['filters']}")
            print(f"  Total incidents: {filter_summary['total_incidents']}")
            print(
                "  Eligible after metadata filtering: "
                f"{filter_summary['eligible_incidents']}"
            )
            print(f"  BM25 results: {filter_summary['bm25_results']}")
            print(f"  Vector results: {filter_summary['vector_results']}")
            print(f"  RRF results: {filter_summary['rrf_results']}")
        print("\n[DEBUG] QUERY NORMALIZATION")
        print(f"  Original: {trace['original_query']}")
        print(f"  Normalized: {trace['normalized_query']}")
        print(f"  BM25 tokens: {', '.join(trace['bm25_tokens'])}")
        print("\n[DEBUG] RRF CONTRIBUTIONS")
        for incident_id, details in trace["rrf_contributions"].items():
            print(
                f"  {incident_id}: BM25 rank={details['bm25_rank']}, "
                f"contribution={details['bm25_contribution']:.6f}; "
                f"Vector rank={details['vector_rank']}, "
                f"contribution={details['vector_contribution']:.6f}; "
                f"total={details['rrf_score']:.6f}"
            )

    print("\n" + "=" * 80)


def run_query(
    hybrid_searcher,
    df: pd.DataFrame,
    query: str,
    top_k: int = 5,
    debug: bool = False,
    filters: dict | None = None,
) -> None:
    results = hybrid_searcher.search(
        query, top_k=top_k, filters=filters, debug=debug
    )
    display_results(query, df, results, debug=debug)


def parse_filter_args(filter_args: list[str] | None) -> dict[str, str]:
    """Parse repeated FIELD=VALUE arguments into the canonical filter mapping."""
    filters = {}
    for filter_arg in filter_args or []:
        field, separator, value = filter_arg.partition("=")
        if not separator or not field.strip() or not value.strip():
            raise ValueError(
                f"Invalid filter {filter_arg!r}; expected FIELD=VALUE"
            )
        field = field.strip()
        if field in filters:
            raise ValueError(f"Metadata filter field {field!r} was specified twice")
        filters[field] = value.strip()
    return filters


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Search cybersecurity incidents using BM25 + vector + RRF hybrid retrieval.")
    parser.add_argument("query", nargs="?", help="Optional query to run once without an interactive prompt.")
    parser.add_argument("--top-k", type=positive_int, default=5, help="Number of results to show per search.")
    parser.add_argument("--debug", action="store_true", help="Show normalized tokens and per-retriever RRF contributions.")
    parser.add_argument(
        "--filter",
        action="append",
        metavar="FIELD=VALUE",
        help="Restrict retrieval by metadata; repeat to combine filters with AND.",
    )
    args = parser.parse_args()
    try:
        filters = parse_filter_args(args.filter)
    except ValueError as exc:
        parser.error(str(exc))

    print("\n" + "=" * 80)
    print("INTERACTIVE HYBRID SEARCH")
    print("=" * 80)

    print("\nLoading incidents...")
    df = pd.read_parquet(CLEANED_INCIDENTS_FILE)
    try:
        validate_incident_frame(df)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1
    print(f"[OK] Loaded {len(df)} incidents")

    print("Checking for cached indexes...")
    chroma_path = project_root / "data" / "processed" / "chroma"
    bm25_path = project_root / "data" / "processed" / "bm25_index.pkl"

    cache_exists = chroma_path.exists() and bm25_path.exists()

    if not cache_exists:
        print("[WARN] Indexes not found in cache")
        print("  Please run: python scripts/indexing/build_search_indexes.py")
        return 1

    print("[OK] Cached indexes found, loading...\n")

    print("Loading incident text...")
    incident_texts = df.set_index('incident_id')['incident_text']
    metadata_list = build_incident_metadata(df)
    print(f"[OK] Loaded {len(incident_texts)} text descriptions from parquet")
    filterable_fields = sorted({
        field
        for record in metadata_list
        for field, value in record.items()
        if field != "incident_id" and value is not None
    })
    print(f"[INFO] Available metadata filter fields: {', '.join(filterable_fields)}")

    print("Loading BM25 index...")
    print(f"  Loading from cache: {bm25_path}")
    try:
        bm25_index = BM25Index.load(bm25_path)
        if (
            bm25_index.incident_ids != incident_texts.index.astype(str).tolist()
            or bm25_index.texts != incident_texts.astype(str).tolist()
            or bm25_index.metadata != metadata_list
        ):
            print("[ERROR] BM25 index is stale or misaligned with the Parquet data.")
            print("        Run: python scripts/indexing/build_search_indexes.py --rebuild")
            return 1
        print("[OK] BM25 index loaded from cache")
    except Exception as e:
        print(f"[ERROR] Failed to load BM25 cache: {e}")
        return 1

    print("Loading Chroma vector store...")
    print(f"  Loading from cache: {chroma_path}")
    try:
        vector_store = ChromaVectorStore(persist_dir=chroma_path)
        vector_store.get_collection()
        vector_store.validate_alignment(
            incident_texts.index.astype(str).tolist(),
            incident_texts.astype(str).tolist(),
            metadata_list=metadata_list,
        )
        print("[OK] Chroma vector store loaded from cache")
    except Exception as e:
        print(f"[ERROR] Failed to load Chroma cache: {e}")
        print("        Run: python scripts/indexing/build_search_indexes.py --rebuild")
        return 1

    print("Loading embedding model (for query encoding)...")
    try:
        embedder = IncidentEmbedder()
        print("[OK] Embedding model loaded")
    except Exception as e:
        print("\n[ERROR] Could not download embedding model")
        print(f"   Reason: {str(e)[:100]}...")
        return 1

    hybrid_searcher = HybridSearcher(bm25_index, vector_store, embedder)
    print_intro()

    if args.query:
        run_query(
            hybrid_searcher,
            df,
            args.query.strip(),
            top_k=args.top_k,
            debug=args.debug,
            filters=filters,
        )
        return 0

    if sys.stdin is not None and not sys.stdin.isatty():
        default_query = "DDoS attack high packet volume"
        print(f"[INFO] Non-interactive session detected; using default query: '{default_query}'")
        run_query(
            hybrid_searcher,
            df,
            default_query,
            top_k=args.top_k,
            debug=args.debug,
            filters=filters,
        )
        return 0

    while True:
        try:
            query = input("\n[SEARCH] Enter search query (or 'quit' to exit):\n> ").strip()
        except EOFError:
            print("\n[INFO] No input received; exiting gracefully.")
            return 0

        if query.lower() in ['quit', 'exit', 'q']:
            print("\n[EXIT] Goodbye!")
            return 0

        if not query:
            print("[WARN] Please enter a query")
            continue

        run_query(
            hybrid_searcher,
            df,
            query,
            top_k=args.top_k,
            debug=args.debug,
            filters=filters,
        )


if __name__ == "__main__":
    raise SystemExit(main())
