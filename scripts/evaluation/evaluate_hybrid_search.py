#!/usr/bin/env python3
"""Evaluate hybrid retrieval against a small, explicit attack-label query set."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from config import CLEANED_INCIDENTS_FILE, DATA_PROCESSED_DIR
from retrieval.bm25_index import BM25Index
from retrieval.chroma_store import ChromaVectorStore
from retrieval.embedder import IncidentEmbedder
from retrieval.hybrid import HybridSearcher
from retrieval.metadata import validate_incident_frame


EVALUATION_QUERIES: List[Tuple[str, Set[str]]] = [
    ("PortScan SYN multiple destinations", {"PortScan"}),
    ("port scanning TCP SYN", {"PortScan"}),
    ("Web Attack SQL Injection", {"Web Attack - Sql Injection"}),
    ("SQL injection traffic", {"Web Attack - Sql Injection"}),
    ("web cross site scripting", {"Web Attack - XSS"}),
    ("FTP brute force", {"FTP-Patator"}),
    ("SSH brute force", {"SSH-Patator"}),
    ("DDoS attack", {"DDoS"}),
    ("slow HTTP denial of service", {"DoS Slowhttptest"}),
    ("Hulk denial of service", {"DoS Hulk"}),
    ("suspicious TCP SYN traffic", {"PortScan"}),
    (
        "abnormal high packet rate",
        {"DDoS", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"},
    ),
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def evaluate_ranking(
    ranking: Iterable[Tuple[str, float]],
    labels: Dict[str, str],
    expected: Set[str],
    top_k: int,
) -> Tuple[int, float, float]:
    ranked_labels = [labels[incident_id] for incident_id, _ in ranking]
    relevant_ranks = [
        rank for rank, label in enumerate(ranked_labels[:top_k], 1)
        if label in expected
    ]
    hit_at_k = int(bool(relevant_ranks))
    precision_at_k = len(relevant_ranks) / top_k
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0
    return hit_at_k, precision_at_k, reciprocal_rank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=positive_int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional UTF-8 report path for full per-query results and metrics.",
    )
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_parquet(CLEANED_INCIDENTS_FILE)
    validate_incident_frame(frame)
    incident_texts = frame.set_index("incident_id")["incident_text"]
    expected_ids = incident_texts.index.astype(str).tolist()

    bm25 = BM25Index.load(DATA_PROCESSED_DIR / "bm25_index.pkl")
    if bm25.incident_ids != expected_ids:
        raise ValueError(
            "BM25 index is stale or misaligned; run "
            "scripts/indexing/build_search_indexes.py --rebuild"
        )
    expected_texts = incident_texts.astype(str).tolist()
    if bm25.texts != expected_texts:
        raise ValueError(
            "BM25 text differs from Parquet; run "
            "scripts/indexing/build_search_indexes.py --rebuild"
        )

    vector_store = ChromaVectorStore(DATA_PROCESSED_DIR / "chroma")
    vector_store.get_collection()
    vector_store.validate_alignment(expected_ids, expected_texts)

    embedder = IncidentEmbedder()
    searcher = HybridSearcher(bm25, vector_store, embedder)
    records = frame.set_index("incident_id", drop=False)
    labels = {str(row.incident_id): str(row.label) for row in frame.itertuples()}
    lines: List[str] = []
    lines.append("APCIRA Hybrid Retrieval Evaluation")
    lines.append("==================================")
    lines.append(
        "Expected labels are relevance targets for this small, hand-labeled query set; "
        "the two broad behavior queries accept multiple attack labels."
    )
    lines.append(
        f"Documents: {len(frame)}; unique incident texts: {incident_texts.nunique()}; "
        f"vector metric: cosine; embedding dimensions: "
        f"{(vector_store.collection.metadata or {}).get('embedding_dimension', 'unknown')}."
    )
    metric_values = {
        method: []
        for method in ("BM25", "Vector", "RRF")
    }

    for query, expected in EVALUATION_QUERIES:
        results = searcher.search(query, top_k=args.top_k)
        lines.extend(
            [
                "",
                "-" * 78,
                f"QUERY: {query}",
                f"EXPECTED LABELS: {', '.join(sorted(expected))}",
            ]
        )

        method_rankings = {
            "BM25": results["bm25"],
            "Vector": [(incident_id, distance) for incident_id, distance, _ in results["vector"]],
            "RRF": results["fused"],
        }
        for method, ranking in method_rankings.items():
            metrics = evaluate_ranking(ranking, labels, expected, args.top_k)
            metric_values[method].append(metrics)
            lines.append(f"\n{method}:")
            for rank, item in enumerate(ranking[:args.top_k], 1):
                incident_id = item[0]
                row = records.loc[incident_id]
                if method == "Vector":
                    score_label = "distance"
                    score = item[1]
                    text = results["vector"][rank - 1][2]
                else:
                    score_label = "score" if method == "BM25" else "RRF score"
                    score = item[1]
                    text = str(row["incident_text"])
                protocol = row["protocol"] if "protocol" in row.index else "N/A"
                destination_port = row["dst_port"]
                if pd.isna(destination_port):
                    destination_port = "N/A"
                elif isinstance(destination_port, float) and destination_port.is_integer():
                    destination_port = int(destination_port)
                lines.append(
                    f"{rank}. {incident_id} | label={row['label']} | "
                    f"{score_label}={score:.6f} | protocol={protocol} | "
                    f"dst_port={destination_port}"
                )
                lines.append(f"   {text[:160]}")

    lines.extend(["", "Aggregate metrics", "-----------------"])
    lines.append(f"Evaluated queries: {len(EVALUATION_QUERIES)}")
    for method, values in metric_values.items():
        count = len(values)
        hit_at_k = sum(value[0] for value in values) / count
        precision_at_k = sum(value[1] for value in values) / count
        mean_reciprocal_rank = sum(value[2] for value in values) / count
        lines.append(
            f"{method}: Hit@{args.top_k}={hit_at_k:.3f}, "
            f"Precision@{args.top_k}={precision_at_k:.3f}, "
            f"MRR={mean_reciprocal_rank:.3f}"
        )
    lines.append(
        "\nMetrics count any returned result whose label is in the expected-label set; "
        "Precision@K divides by K even when fewer than K results are returned."
    )

    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
