#!/usr/bin/env python3
"""Build or rebuild the BM25 and Chroma indexes from prepared incidents."""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from config import CLEANED_INCIDENTS_FILE, DATA_PROCESSED_DIR
from retrieval.bm25_index import BM25Index
from retrieval.chroma_store import ChromaVectorStore
from retrieval.embedder import IncidentEmbedder
from retrieval.metadata import build_incident_metadata, validate_incident_frame


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or rebuild BM25 and Chroma search indexes."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Replace existing indexes using the current Parquet data.",
    )
    args = parser.parse_args()

    log_dir = project_root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"build_indexes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("BUILD SEARCH INDEXES (Embeddings + BM25 + Chroma)")
    logger.info("=" * 80)

    bm25_path = DATA_PROCESSED_DIR / "bm25_index.pkl"
    chroma_path = DATA_PROCESSED_DIR / "chroma"
    temporary_bm25_path = bm25_path.with_name(bm25_path.name + ".tmp")

    logger.info("Loading incidents from %s", CLEANED_INCIDENTS_FILE)
    try:
        import pandas as pd

        frame = pd.read_parquet(CLEANED_INCIDENTS_FILE)
        validate_incident_frame(frame)
    except Exception as exc:
        logger.error("Could not load or validate incident data: %s", exc)
        return 1

    incident_texts = frame.set_index("incident_id")["incident_text"]
    metadata_list = build_incident_metadata(frame)
    logger.info("Loaded %s incidents", f"{len(frame):,}")
    logger.info(
        "Exact duplicate texts: %s (%s unique)",
        f"{int(incident_texts.duplicated().sum()):,}",
        f"{incident_texts.nunique():,}",
    )

    indexes_exist = bm25_path.exists() and chroma_path.exists()
    if indexes_exist and not args.rebuild:
        logger.info("Indexes already exist; no files were changed.")
        logger.info(
            "To rebuild, run: python scripts/indexing/build_search_indexes.py --rebuild"
        )
        return 0

    logger.info("Encoding %s documents", f"{len(frame):,}")
    try:
        embedder = IncidentEmbedder()
        embeddings = embedder.embed_texts(incident_texts.tolist(), batch_size=128)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(frame):
            raise ValueError(f"Unexpected document embedding shape: {embeddings.shape}")
        logger.info(
            "Embedding shape: %s; dimension: %s",
            embeddings.shape,
            embeddings.shape[1],
        )
    except Exception as exc:
        logger.error("Failed to encode incident texts: %s", exc)
        return 1

    try:
        bm25_index = BM25Index(incident_texts, metadata=metadata_list)
        bm25_index.save(temporary_bm25_path)
        logger.info("Staged BM25 index at %s", temporary_bm25_path)
    except Exception as exc:
        logger.error("Failed to build BM25 index: %s", exc)
        temporary_bm25_path.unlink(missing_ok=True)
        return 1

    try:
        vector_store = ChromaVectorStore(persist_dir=chroma_path)
        vector_store.build_collection(
            embeddings=embeddings.tolist(),
            incident_ids=frame["incident_id"].astype(str).tolist(),
            texts=incident_texts.tolist(),
            metadata_list=metadata_list,
            batch_size=5000,
        )
    except Exception as exc:
        logger.error("Failed to build Chroma index: %s", exc)
        temporary_bm25_path.unlink(missing_ok=True)
        return 1

    logger.info("Verifying rebuilt indexes")
    try:
        bm25_loaded = BM25Index.load(temporary_bm25_path)
        expected_ids = frame["incident_id"].astype(str).tolist()
        if bm25_loaded.incident_ids != expected_ids:
            raise ValueError("BM25 IDs are not aligned with the Parquet row order")

        vector_store_check = ChromaVectorStore(persist_dir=chroma_path)
        collection = vector_store_check.get_collection()
        if collection.count() != len(frame):
            raise ValueError(
                f"Chroma contains {collection.count()} documents; expected {len(frame)}"
            )
        if (collection.metadata or {}).get("embedding_dimension") != embeddings.shape[1]:
            raise ValueError("Chroma embedding dimension differs from the generated vectors")
        vector_store_check.validate_alignment(
            expected_ids,
            incident_texts.astype(str).tolist(),
            metadata_list=metadata_list,
        )

        temporary_bm25_path.replace(bm25_path)
    except Exception as exc:
        logger.error("Index verification failed: %s", exc)
        temporary_bm25_path.unlink(missing_ok=True)
        return 1

    logger.info("BM25 index: %s", bm25_path)
    logger.info("Chroma database: %s", chroma_path)
    logger.info("Build log: %s", log_file)
    logger.info("Indexes rebuilt and verified successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
