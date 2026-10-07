"""
Hybrid search combining BM25 and vector similarity via Reciprocal Rank Fusion (RRF).
"""
import logging
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BM25_WEIGHT, RRF_K, VECTOR_WEIGHT
from retrieval.metadata_filter import MetadataFilter

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Fuses BM25 and vector search results using Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25_index,
        vector_store,
        embedder,
        bm25_weight: float = BM25_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
        rrf_k: int = RRF_K,
        deduplicate_exact_text: bool = True,
    ):
        """
        Initialize hybrid searcher with both indexes.

        Args:
            bm25_index: BM25Index instance.
            vector_store: ChromaVectorStore instance.
            embedder: IncidentEmbedder instance.
        """
        self.bm25 = bm25_index
        self.vector_store = vector_store
        self.embedder = embedder
        if bm25_weight < 0 or vector_weight < 0:
            raise ValueError("RRF weights must be non-negative")
        if rrf_k <= 0:
            raise ValueError("RRF k must be a positive integer")
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.deduplicate_exact_text = deduplicate_exact_text

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: List[List[Tuple[str, float]]],
        k: int = RRF_K,
        top_k: int = 10,
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple ranked result lists using Reciprocal Rank Fusion.

        The RRF score for a document is: sum(1 / (k + rank)) across all result lists.

        Args:
            result_lists: List of ranked result lists. Each result list contains
                         tuples of (incident_id, score).
            k: Constant in the RRF formula (default 60). Higher k = more emphasis
              on multiple lists having the document.
            top_k: Number of top fused results to return.

        Returns:
            List of (incident_id, rrf_score) tuples, sorted by score (descending).
        """
        if k <= 0:
            raise ValueError("RRF k must be a positive integer")
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if weights is None:
            weights = [1.0] * len(result_lists)
        if len(weights) != len(result_lists):
            raise ValueError("There must be one RRF weight per result list")
        if any(weight < 0 for weight in weights):
            raise ValueError("RRF weights must be non-negative")

        rrf_scores = defaultdict(float)

        for weight, result_list in zip(weights, result_lists):
            seen_ids = set()
            for rank, (incident_id, _) in enumerate(result_list):
                if incident_id in seen_ids:
                    continue
                seen_ids.add(incident_id)
                rrf_scores[incident_id] += weight / (k + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]

        return fused

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        deduplicate_exact_text: Optional[bool] = None,
        candidate_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining BM25 and vector similarity.

        Args:
            query: Search query string.
            top_k: Number of top results to return.
            filters: Metadata filters to apply to vector search.

        Returns:
            Dict with keys:
              - 'fused': List of (incident_id, rrf_score) from hybrid fusion
              - 'bm25': List of (incident_id, bm25_score) from BM25 only
              - 'vector': List of (incident_id, distance, text) from vector only
        """
        logger.info(f"Hybrid search for: '{query}' (top_k={top_k})")

        # BM25 search
        if not query.strip():
            raise ValueError("query must not be empty")
        deduplicate = (
            self.deduplicate_exact_text
            if deduplicate_exact_text is None
            else deduplicate_exact_text
        )
        metadata_filter = None
        eligible_ids = None
        normalized_filters = filters
        if filters:
            metadata_filter = MetadataFilter(self.bm25.metadata)
            normalized_filters, eligible_ids = metadata_filter.eligible_ids(filters)

        if candidate_ids is not None:
            restricted_ids = {str(item) for item in candidate_ids}
            if eligible_ids is not None:
                restricted_ids.intersection_update(eligible_ids)
            eligible_ids = sorted(restricted_ids)

        bm25_results = self.bm25.search(
            query,
            top_k=top_k,
            filters=normalized_filters,
            deduplicate_exact_text=deduplicate,
            candidate_ids=eligible_ids,
        )
        logger.debug(f"BM25 found {len(bm25_results)} results")

        query_embeddings = self.embedder.embed_texts([query])
        if query_embeddings.ndim != 2 or query_embeddings.shape[0] != 1:
            raise ValueError(
                f"Expected one two-dimensional query embedding, got {query_embeddings.shape}"
            )
        query_embedding = query_embeddings[0].tolist()
        vector_search_args = {
            "top_k": top_k,
            "filters": normalized_filters,
            "deduplicate_exact_text": deduplicate,
        }
        if candidate_ids is not None:
            vector_search_args["candidate_ids"] = eligible_ids or []
        vector_results = self.vector_store.search(query_embedding, **vector_search_args)
        logger.debug(f"Vector search found {len(vector_results)} results")

        bm25_for_rrf = [(incident_id, score) for incident_id, score in bm25_results]
        vector_for_rrf = [
            (incident_id, 0.0) for incident_id, _, _ in vector_results
        ]
        fused = self.reciprocal_rank_fusion(
            [bm25_for_rrf, vector_for_rrf],
            k=self.rrf_k,
            top_k=top_k,
            weights=[self.bm25_weight, self.vector_weight],
        )
        result_ids = {
            incident_id
            for incident_id, _ in bm25_results
        } | {
            incident_id
            for incident_id, _, _ in vector_results
        } | {
            incident_id
            for incident_id, _ in fused
        }

        response = {
            'fused': fused,
            'bm25': bm25_results,
            'vector': vector_results,
            'metadata': {
                record["incident_id"]: record
                for record in self.bm25.metadata
                if record.get("incident_id") in result_ids
            },
        }
        if metadata_filter is not None:
            response["metadata_filter"] = {
                "filters": normalized_filters,
                "total_incidents": len(self.bm25.incident_ids),
                "eligible_incidents": len(eligible_ids),
            }
        if candidate_ids is not None:
            response["candidate_filter"] = {
                "candidate_count": len(eligible_ids or []),
                "bm25_result_count": len(bm25_results),
                "vector_result_count": len(vector_results),
                "rrf_result_count": len(fused),
            }
        if debug:
            bm25_ranks = {
                incident_id: rank
                for rank, (incident_id, _) in enumerate(bm25_results, 1)
            }
            vector_ranks = {
                incident_id: rank
                for rank, (incident_id, _, _) in enumerate(vector_results, 1)
            }
            details = {}
            for incident_id, score in fused:
                bm25_rank = bm25_ranks.get(incident_id)
                vector_rank = vector_ranks.get(incident_id)
                details[incident_id] = {
                    "bm25_rank": bm25_rank,
                    "vector_rank": vector_rank,
                    "bm25_contribution": (
                        self.bm25_weight / (self.rrf_k + bm25_rank)
                        if bm25_rank is not None else 0.0
                    ),
                    "vector_contribution": (
                        self.vector_weight / (self.rrf_k + vector_rank)
                        if vector_rank is not None else 0.0
                    ),
                    "rrf_score": score,
                }
            response["debug"] = {
                "original_query": query,
                "normalized_query": self.bm25.normalized_query(query),
                "bm25_tokens": self.bm25.normalized_query(query).split(),
                "rrf_contributions": details,
            }
            if metadata_filter is not None:
                response["debug"]["metadata_filter"] = {
                    **response["metadata_filter"],
                    "bm25_results": len(bm25_results),
                    "vector_results": len(vector_results),
                    "rrf_results": len(fused),
                }
        return response
