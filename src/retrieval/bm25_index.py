"""
Build and query a BM25 index over incident texts.
"""
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pickle
from pathlib import Path
import re
import pandas as pd
from rank_bm25 import BM25Okapi
from retrieval.filters import filter_field_names, matches_metadata

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """
    Normalize punctuation, separators, camel case, and common inflections.

    Args:
        text: Text to tokenize.

    Returns:
        A normalized token sequence shared by documents and queries.
    """
    def split_camel_case(match: re.Match) -> str:
        word = match.group(0)
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", word)
        if len(parts) > 1 and all(len(part) >= 3 for part in parts):
            return " ".join(parts)
        return word

    text = re.sub(r"[A-Za-z0-9]+", split_camel_case, text)
    text = re.sub(r"[\W_]+", " ", text.casefold())
    tokens = []

    for token in text.split():
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
            if len(token) > 3 and token[-1] == token[-2]:
                token = token[:-1]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
            if len(token) > 3 and token[-1] == token[-2]:
                token = token[:-1]
        elif len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        tokens.append(token)

    return tokens


class BM25Index:
    """BM25 keyword search index over incident texts."""

    def __init__(
        self,
        texts: pd.Series,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Build a BM25 index from incident texts.

        Args:
            texts: Series of incident texts (index=incident_id, values=text).
        """
        logger.info(f"Building BM25 index for {len(texts)} incidents")

        self.incident_ids = [str(incident_id) for incident_id in texts.index.tolist()]
        self.texts = [str(text) for text in texts.values.tolist()]
        if len(set(self.incident_ids)) != len(self.incident_ids):
            raise ValueError("BM25 incident IDs must be unique")
        if metadata is None:
            self.metadata = [{} for _ in self.incident_ids]
        else:
            if len(metadata) != len(self.incident_ids):
                raise ValueError("BM25 metadata count must match the document count")
            self.metadata = metadata
            for incident_id, item in zip(self.incident_ids, self.metadata):
                metadata_id = item.get("incident_id")
                if metadata_id is not None and str(metadata_id) != incident_id:
                    raise ValueError(
                        f"BM25 metadata ID {metadata_id!r} does not match document ID {incident_id!r}"
                    )

        # Tokenize texts: lowercase, remove punctuation, split on whitespace
        tokenized_texts = [tokenize(text) for text in self.texts]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_texts)
        for token, inverse_document_frequency in self.bm25.idf.items():
            if inverse_document_frequency < 0:
                self.bm25.idf[token] = 0.0
        logger.info("BM25 index built")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        deduplicate_exact_text: bool = True,
        candidate_ids: Optional[Iterable[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for incidents using BM25 ranking.

        Args:
            query: Search query string.
            top_k: Number of top results to return.
            filters: Optional metadata filters applied before ranking.
            deduplicate_exact_text: Return at most one incident per exact document.
            candidate_ids: Optional incident IDs eligible for ranking.

        Returns:
            List of (incident_id, score) tuples, sorted by score (descending).
        """
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        if filters and not any(self.metadata):
            raise ValueError(
                "BM25 metadata is unavailable; rebuild the indexes before using metadata filters"
            )
        unavailable_fields = filter_field_names(filters) - {
            field for item in self.metadata for field in item
        }
        if unavailable_fields:
            raise ValueError(
                "BM25 metadata fields are unavailable in this index: "
                f"{', '.join(sorted(unavailable_fields))}"
            )

        allowed_ids = (
            None if candidate_ids is None else {str(item) for item in candidate_ids}
        )
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            (
                idx for idx, score in enumerate(scores)
                if score != 0
                and matches_metadata(self.metadata[idx], filters)
                and (
                    allowed_ids is None
                    or self.incident_ids[idx] in allowed_ids
                )
            ),
            key=lambda idx: (-float(scores[idx]), self.incident_ids[idx]),
        )

        results: List[Tuple[str, float]] = []
        seen_texts = set()
        for idx in ranked_indices:
            text = self.texts[idx]
            if deduplicate_exact_text and text in seen_texts:
                continue
            seen_texts.add(text)
            results.append((self.incident_ids[idx], float(scores[idx])))
            if len(results) == top_k:
                break
        return results

    def normalized_query(self, query: str) -> str:
        """Return the exact token-normalized form used by BM25."""
        return " ".join(tokenize(query))

    def save(self, path: Path):
        """Save BM25 index to disk using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "incident_ids": self.incident_ids,
                "texts": self.texts,
                "metadata": self.metadata,
            }, f)
        logger.info(f"BM25 index saved to {path}")

    @classmethod
    def load(cls, path: Path):
        """Load BM25 index from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.bm25 = data["bm25"]
        instance.incident_ids = data["incident_ids"]
        instance.texts = data["texts"]
        instance.metadata = data.get("metadata", [{} for _ in instance.incident_ids])
        if len(instance.metadata) != len(instance.incident_ids):
            raise ValueError("Saved BM25 metadata is not aligned with incident IDs")
        if len(set(instance.incident_ids)) != len(instance.incident_ids):
            raise ValueError("Saved BM25 incident IDs are not unique")
        logger.info(f"BM25 index loaded from {path}")
        return instance
