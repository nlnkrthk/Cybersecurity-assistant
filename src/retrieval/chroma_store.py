"""
Build and query a Chroma vector store for incident embeddings with metadata filtering.
"""
import logging
import json
from typing import List, Tuple, Optional, Dict, Any
import chromadb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_COLLECTION_NAME, DATA_PROCESSED_DIR
from retrieval.filters import filter_field_names, to_chroma_where

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Vector store for incident embeddings using Chroma."""

    def __init__(self, persist_dir: Optional[Path] = None):
        """
        Initialize Chroma client with persistence.

        Args:
            persist_dir: Directory to persist Chroma data. Defaults to DATA_PROCESSED_DIR/chroma.
        """
        if persist_dir is None:
            persist_dir = DATA_PROCESSED_DIR / "chroma"

        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )

        logger.info(f"Chroma client initialized. Persist dir: {self.persist_dir}")

    def build_collection(
        self,
        embeddings: List[List[float]],
        incident_ids: List[str],
        texts: List[str],
        metadata_list: List[Dict[str, Any]],
        collection_name: str = CHROMA_COLLECTION_NAME,
        batch_size: int = 5000
    ):
        """
        Create or replace a collection with embeddings and metadata.

        Args:
            embeddings: List of embedding vectors.
            incident_ids: List of incident IDs (document IDs).
            texts: List of incident text descriptions.
            metadata_list: List of metadata dicts for each incident.
            collection_name: Name of the collection.
            batch_size: Number of documents to add per batch (default 5000).
        """
        if not embeddings:
            raise ValueError("Cannot build a Chroma collection without embeddings")
        if not (len(embeddings) == len(incident_ids) == len(texts) == len(metadata_list)):
            raise ValueError(
                "Embeddings, incident IDs, texts, and metadata must have equal lengths"
            )
        if len(set(incident_ids)) != len(incident_ids):
            raise ValueError("Chroma incident IDs must be unique")

        embedding_dimension = len(embeddings[0])
        if embedding_dimension == 0 or any(
            len(embedding) != embedding_dimension for embedding in embeddings
        ):
            raise ValueError("All document embeddings must have the same non-zero dimension")
        for incident_id, metadata in zip(incident_ids, metadata_list):
            metadata_id = metadata.get("incident_id")
            if metadata_id is None or str(metadata_id) != str(incident_id):
                raise ValueError(
                    f"Chroma metadata ID {metadata_id!r} does not match document ID {incident_id!r}"
                )

        logger.info(
            "Building collection '%s' with %s documents (%sD embeddings)",
            collection_name,
            len(embeddings),
            embedding_dimension,
        )

        # Delete existing collection if it exists
        existing = self.client.list_collections()
        collection_names = {
            item if isinstance(item, str) else item.name for item in existing
        }
        if collection_name in collection_names:
            self.client.delete_collection(name=collection_name)

        # Create new collection
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_dimension": embedding_dimension,
                "retrieval:metadata_fields": json.dumps(
                    sorted({key for item in metadata_list for key in item})
                ),
            },
        )

        # Add documents in batches (Chroma has batch size limits)
        total = len(embeddings)
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            batch_ids = incident_ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_texts = texts[i:end_idx]
            batch_metadata = metadata_list[i:end_idx]

            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadata
            )

            logger.info(f"Added batch {i//batch_size + 1}: {end_idx}/{total} documents")

        logger.info(f"Collection '{collection_name}' created with {len(embeddings)} documents")
        self.collection = collection

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        deduplicate_exact_text: bool = True,
        candidate_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Query the collection using vector similarity.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of top results to return.
            filters: Chroma metadata filter dict (optional).

        Returns:
            List of (incident_id, distance, text) tuples, sorted by distance (ascending).
        """
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")
        expected_dimension = (self.collection.metadata or {}).get("embedding_dimension")
        if expected_dimension is not None and len(query_embedding) != expected_dimension:
            raise ValueError(
                f"Query embedding dimension {len(query_embedding)} does not match "
                f"indexed dimension {expected_dimension}"
            )

        where = to_chroma_where(filters)
        if candidate_ids is not None:
            normalized_candidate_ids = list(dict.fromkeys(map(str, candidate_ids)))
            if not normalized_candidate_ids:
                return []
            candidate_where = {"incident_id": {"$in": normalized_candidate_ids}}
            where = (
                {"$and": [where, candidate_where]}
                if where
                else candidate_where
            )
        stored_fields = (self.collection.metadata or {}).get("retrieval:metadata_fields")
        if filters and stored_fields is None:
            raise ValueError(
                "Chroma index has no metadata field schema; rebuild the search indexes"
            )
        available_fields = set(json.loads(stored_fields)) if stored_fields else set()
        unavailable_fields = filter_field_names(filters) - available_fields
        if unavailable_fields:
            raise ValueError(
                "Chroma metadata fields are unavailable in this index: "
                f"{', '.join(sorted(unavailable_fields))}"
            )
        collection_count = self.collection.count()
        requested = min(collection_count, max(top_k, top_k * 5))
        if requested == 0:
            return []

        candidates_by_id: Dict[str, Tuple[str, float, str]] = {}
        while True:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=requested,
                where=where,
                include=["distances", "documents"],
            )
            ids = results["ids"][0]
            distances = results["distances"][0]
            documents = results["documents"][0]
            candidates_by_id.update({
                str(ids[i]): (str(ids[i]), float(distances[i]), documents[i])
                for i in range(len(ids))
                if documents[i] is not None
            })
            candidates = list(candidates_by_id.values())
            candidates.sort(key=lambda item: (item[1], item[0]))

            ranked: List[Tuple[str, float, str]] = []
            seen_texts = set()
            for candidate in candidates:
                if deduplicate_exact_text and candidate[2] in seen_texts:
                    continue
                seen_texts.add(candidate[2])
                ranked.append(candidate)
                if len(ranked) == top_k:
                    break
            if (
                len(ranked) == top_k
                or requested == collection_count
                or len(ids) < requested
            ):
                return ranked
            requested = min(collection_count, requested * 2)

    def get_collection(self, collection_name: str = CHROMA_COLLECTION_NAME):
        """
        Get an existing collection by name.

        Args:
            collection_name: Name of the collection.

        Returns:
            Chroma collection object.
        """
        self.collection = self.client.get_collection(name=collection_name)
        logger.info(f"Loaded collection '{collection_name}'")
        return self.collection

    def validate_alignment(
        self,
        incident_ids: List[str],
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Raise if indexed documents or metadata do not match the source rows."""
        if len(incident_ids) != len(texts):
            raise ValueError("Expected incident IDs and texts to have equal lengths")
        if metadata_list is not None and len(metadata_list) != len(incident_ids):
            raise ValueError("Expected one metadata record per incident ID")

        expected_texts = dict(zip(map(str, incident_ids), texts))
        if len(expected_texts) != len(incident_ids):
            raise ValueError("Expected source incident IDs to be unique")
        records = self.collection.get(include=["documents", "metadatas"])
        indexed_ids = records["ids"]
        if set(indexed_ids) != set(expected_texts):
            raise ValueError("Chroma document IDs do not match source incident IDs")

        source_metadata = (
            {str(item["incident_id"]): item for item in metadata_list}
            if metadata_list is not None
            else None
        )
        for incident_id, document, metadata in zip(
            indexed_ids, records["documents"], records["metadatas"]
        ):
            if document != expected_texts[incident_id]:
                raise ValueError(
                    f"Chroma document text does not match source for {incident_id}"
                )
            if metadata.get("incident_id") != incident_id:
                raise ValueError(f"Chroma metadata ID is misaligned for {incident_id}")
            if source_metadata is not None:
                expected = source_metadata[incident_id]
                for key, expected_value in expected.items():
                    if metadata.get(key) != expected_value:
                        raise ValueError(
                            f"Chroma metadata field {key!r} is misaligned for {incident_id}"
                        )
