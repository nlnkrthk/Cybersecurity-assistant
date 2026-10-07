"""End-to-end retrieval-augmented generation pipeline for APCIRA."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from config import CLEANED_INCIDENTS_FILE
from llm.client import LLMClient
from rag.context_builder import build_context
from rag.prompt_builder import build_prompt
from retrieval.bm25_index import BM25Index
from retrieval.chroma_store import ChromaVectorStore
from retrieval.embedder import IncidentEmbedder
from retrieval.hybrid import HybridSearcher
from retrieval.metadata import build_incident_metadata, validate_incident_frame
from retrieval.metadata_filter import FIELD_ALIASES
from retrieval.metadata_filter import MetadataFilter
from retrieval.filters import matches_metadata
from retrieval.numeric_filter import apply_numeric_filters, resolve_numeric_conditions

logger = logging.getLogger(__name__)

def _default_hybrid_searcher():
    """Load the persisted retrieval indexes and construct a HybridSearcher."""
    df = pd.read_parquet(CLEANED_INCIDENTS_FILE)
    validate_incident_frame(df)

    processed_root = Path(CLEANED_INCIDENTS_FILE).resolve().parent
    bm25_path = processed_root / "bm25_index.pkl"
    chroma_path = processed_root / "chroma"

    if not bm25_path.exists() or not chroma_path.exists():
        raise FileNotFoundError(
            "Retrieval indexes are missing. Run: python scripts/indexing/build_search_indexes.py"
        )

    incident_texts = df.set_index("incident_id")["incident_text"]
    metadata_list = build_incident_metadata(df)

    bm25_index = BM25Index.load(bm25_path)
    if (
        bm25_index.incident_ids != incident_texts.index.astype(str).tolist()
        or bm25_index.texts != incident_texts.astype(str).tolist()
        or bm25_index.metadata != metadata_list
    ):
        raise ValueError("Cached BM25 index does not match the current Parquet data")

    vector_store = ChromaVectorStore(persist_dir=chroma_path)
    vector_store.get_collection()
    vector_store.validate_alignment(
        incident_texts.index.astype(str).tolist(),
        incident_texts.astype(str).tolist(),
        metadata_list=metadata_list,
    )

    embedder = IncidentEmbedder()
    return HybridSearcher(bm25_index, vector_store, embedder), df


def _coerce_row_for_record(incident_id: Any, row: Optional[pd.Series] = None, metadata: Optional[Mapping[str, Any]] = None):
    metadata = dict(metadata or {})
    if row is not None:
        metadata = {
            **{
                str(field): value
                for field, value in row.items()
                if value is not None and not pd.isna(value)
            },
            **metadata,
        }
    label = metadata.get("label")
    if row is not None and "label" in row:
        label = row["label"]
    incident_text = metadata.get("incident_text")
    if row is not None and "incident_text" in row:
        incident_text = row["incident_text"]

    record = {
        "incident_id": str(incident_id),
        "label": str(label) if label is not None else "N/A",
        "incident_text": str(incident_text) if incident_text is not None else "N/A",
        "metadata": metadata,
    }
    return record


def _result_records(results: Mapping[str, Any], df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    if not results:
        return []

    records_by_id: Dict[str, Dict[str, Any]] = {}
    if df is not None and not df.empty:
        df_by_id = df.set_index("incident_id")
        for incident_id in df_by_id.index:
            row = df_by_id.loc[incident_id]
            records_by_id[str(incident_id)] = _coerce_row_for_record(incident_id, row=row)

    metadata_records = results.get("metadata", []) or []
    if isinstance(metadata_records, Mapping):
        metadata_records = metadata_records.values()

    for metadata_record in metadata_records:
        if not isinstance(metadata_record, Mapping):
            raise TypeError("Hybrid search metadata entries must be mappings")
        incident_id = metadata_record.get("incident_id")
        if incident_id is None:
            continue
        incident_key = str(incident_id)
        if incident_key in records_by_id:
            records_by_id[incident_key]["metadata"].update(metadata_record)
        else:
            records_by_id[incident_key] = _coerce_row_for_record(
                incident_id,
                row=None,
                metadata=metadata_record,
            )

    candidate_ids: List[str] = []
    seen = set()
    result_keys = ("fused",) if results.get("fused") is not None else ("bm25", "vector")
    for key in result_keys:
        for item in results.get(key, []) or []:
            incident_id = item[0] if len(item) >= 2 or (key == "vector" and len(item) >= 3) else None
            if incident_id is None:
                continue
            incident_id = str(incident_id)
            if incident_id not in seen:
                seen.add(incident_id)
                candidate_ids.append(incident_id)

    ranked_records: List[Dict[str, Any]] = []
    for incident_id in candidate_ids:
        score = None
        for key in ("fused", "bm25", "vector"):
            for item in results.get(key, []) or []:
                if key == "vector":
                    item_id = item[0] if len(item) >= 3 else None
                    if item_id is not None and str(item_id) == incident_id:
                        score = item[1]
                        break
                else:
                    item_id = item[0] if len(item) >= 2 else None
                    if item_id is not None and str(item_id) == incident_id:
                        score = item[1]
                        break
            if score is not None:
                break
        base = records_by_id.get(incident_id, {"incident_id": incident_id, "label": "N/A", "incident_text": "N/A", "metadata": {}})
        record = dict(base)
        record.setdefault("incident_id", incident_id)
        record.setdefault("label", "N/A")
        record.setdefault("incident_text", "N/A")
        record.setdefault("metadata", {})
        if score is not None:
            record["score"] = score
        ranked_records.append(record)

    return ranked_records


def validate_query(query: Any) -> str:
    if query is None or not str(query).strip():
        raise ValueError("Query must be a non-empty string")
    return str(query).strip()


def _extract_query_filters(
    query: str, metadata: List[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Extract unambiguous metadata constraints stated in a natural-language query."""
    values_by_field: Dict[str, set] = {}
    for record in metadata:
        for field, value in record.items():
            if field != "incident_id" and value is not None:
                values_by_field.setdefault(field, set()).add(value)

    query_lower = query.casefold()
    inferred: Dict[str, Any] = {}

    def matches_phrase(value: str) -> bool:
        phrase = re.escape(value.casefold())
        return re.search(rf"(?<!\w){phrase}(?!\w)", query_lower) is not None

    def add_values(field: str, found: List[Any]) -> None:
        if field not in values_by_field or not found:
            return
        unique = list(dict.fromkeys(found))
        inferred[field] = unique[0] if len(unique) == 1 else {"$in": unique}

    labels = sorted(
        (value for value in values_by_field.get("label", set()) if isinstance(value, str)),
        key=lambda value: (-len(value), value.casefold()),
    )
    add_values("label", [label for label in labels if matches_phrase(label)])

    protocol_names = {1: "icmp", 6: "tcp", 17: "udp"}
    stored_protocols = values_by_field.get("protocol", set())
    protocol_matches: List[Any] = []
    for stored_value in stored_protocols:
        if isinstance(stored_value, (int, float)) and not isinstance(stored_value, bool):
            protocol_name = protocol_names.get(int(stored_value))
            if protocol_name and matches_phrase(protocol_name):
                protocol_matches.append(stored_value)
        elif isinstance(stored_value, str) and matches_phrase(stored_value):
            protocol_matches.append(stored_value)
    protocol_match = re.search(r"\bprotocol\s*(?:is|=|:)?\s*(\d+)\b", query_lower)
    if protocol_match:
        number = int(protocol_match.group(1))
        protocol_matches.extend(
            value
            for value in stored_protocols
            if str(value) == str(number)
            or (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value == number
            )
        )
    add_values("protocol", protocol_matches)

    port_pattern = re.compile(
        r"\b(?:(source|src|destination|dest|dst)\s+)?port"
        r"\s*(?:number\s*)?(?:is\s*|=|:)?\s*(\d{1,5})\b",
        re.IGNORECASE,
    )
    port_matches: Dict[str, List[int]] = {}
    for match in port_pattern.finditer(query):
        direction = (match.group(1) or "destination").casefold()
        field = "src_port" if direction in {"source", "src"} else "dst_port"
        if field not in values_by_field:
            continue
        port_number = int(match.group(2))
        if any(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == port_number
            for value in values_by_field[field]
        ):
            port_matches.setdefault(field, []).append(port_number)
    for field, ports in port_matches.items():
        add_values(field, ports)

    ip_pattern = re.compile(
        r"\b(source|src|destination|dest|dst)\s+"
        r"(?:ip(?:\s+address)?|address)\s*(?:is\s*|=|:)?\s*"
        r"((?:\d{1,3}\.){3}\d{1,3})\b",
        re.IGNORECASE,
    )
    for match in ip_pattern.finditer(query):
        direction = match.group(1).casefold()
        field = "src_ip" if direction in {"source", "src"} else "dst_ip"
        ip_value = match.group(2)
        if ip_value in values_by_field.get(field, set()):
            inferred[field] = ip_value

    timestamp_match = re.search(
        r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?\b",
        query,
    )
    if timestamp_match and "timestamp" in values_by_field:
        timestamp = timestamp_match.group(0)
        if timestamp in values_by_field["timestamp"]:
            inferred["timestamp"] = timestamp

    return inferred


def _merge_query_filters(
    inferred_filters: Mapping[str, Any],
    explicit_filters: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Use explicit CLI/API filters in preference to inferred constraints on the same field."""
    merged = dict(inferred_filters)
    for field, value in (explicit_filters or {}).items():
        canonical_field = FIELD_ALIASES.get(field, field)
        merged.pop(canonical_field, None)
        merged[field] = value
    return merged


def rag_pipeline(
    query: Any,
    filters: Optional[Mapping[str, Any]] = None,
    top_k: Optional[int] = None,
    searcher: Optional[HybridSearcher] = None,
    df: Optional[pd.DataFrame] = None,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Run retrieval, build a context block, generate a grounded answer, and return structured output."""
    query_text = validate_query(query)
    if top_k is None:
        top_k = int(os.getenv("RAG_CONTEXT_TOP_K", "5"))
    if top_k < 1:
        raise ValueError("top_k must be a positive integer")

    if searcher is None:
        searcher, df = _default_hybrid_searcher()

    metadata = getattr(getattr(searcher, "bm25", None), "metadata", [])
    inferred_filters = _extract_query_filters(query_text, metadata)
    effective_filters = _merge_query_filters(inferred_filters, filters)

    incident_ids = [
        str(incident_id)
        for incident_id in getattr(getattr(searcher, "bm25", None), "incident_ids", [])
    ]
    if not incident_ids and df is not None and "incident_id" in df.columns:
        incident_ids = df["incident_id"].astype(str).tolist()
    before_metadata_count = len(incident_ids)

    normalized_filters = effective_filters
    if effective_filters:
        if not metadata:
            raise ValueError(
                "Cannot apply metadata filters because the hybrid index has no metadata"
            )
        metadata_filter = MetadataFilter(metadata)
        normalized_filters = metadata_filter.validate(effective_filters)
        normalized_filters, metadata_candidate_ids = metadata_filter.eligible_ids(
            normalized_filters
        )
    else:
        metadata_candidate_ids = incident_ids
    after_metadata_count = len(metadata_candidate_ids)

    if not metadata_candidate_ids:
        candidate_counts = {
            "before_metadata_filter": before_metadata_count,
            "after_metadata_filter": 0,
            "after_numeric_filter": 0,
        }
        return {
            "query": query_text,
            "filters": normalized_filters or None,
            "inferred_filters": inferred_filters,
            "numeric_conditions": [],
            "candidate_counts": candidate_counts,
            "results": [],
            "context": "No matching historical incidents were found.",
            "answer": "No matching historical incidents were found.",
            "evidence_available": False,
        }

    if df is None:
        numeric_conditions: List[Dict[str, Any]] = []
        if re.search(
            r"\bhigh[- ](?:packet|packets|traffic)\s+volume\b|"
            r"\b(?:at\s+least|no\s+less\s+than|greater\s+than|more\s+than|"
            r"above|over|at\s+most|no\s+more\s+than|less\s+than|below|"
            r"under|>=|<=|>|<)\s+\d[\d,]*(?:\.\d+)?\s*"
            r"(?:packets?\s+per\s+second|packets?/s|pps|packets?)\b|"
            r"\b(?:packet\s+rate|packet\s+volume|packets?\s+per\s+second|"
            r"packets?/s|pps|packets)\s+(?:at\s+least|no\s+less\s+than|"
            r"greater\s+than|more\s+than|above|over|at\s+most|no\s+more\s+than|"
            r"less\s+than|below|under|>=|<=|>|<)\s*\d[\d,]*(?:\.\d+)?|"
            r"\bduration\s+(?:at\s+least|no\s+less\s+than|greater\s+than|"
            r"more\s+than|above|over|at\s+most|no\s+more\s+than|less\s+than|"
            r"below|under|>=|<=|>|<)\s*\d[\d,]*(?:\.\d+)?\s*"
            r"(?:microseconds?|milliseconds?|seconds?|minutes?|hours?|us|ms|s)\b",
            query_text,
            re.IGNORECASE,
        ):
            raise ValueError(
                "Numeric traffic conditions require incident feature data for hard filtering"
            )
    else:
        numeric_conditions = resolve_numeric_conditions(
            query_text,
            df,
            metadata_candidate_ids,
        )

    if numeric_conditions:
        if df is None:
            raise ValueError("Numeric traffic conditions require incident feature data")
        numeric_filter_result = apply_numeric_filters(
            df,
            numeric_conditions,
            candidate_ids=metadata_candidate_ids,
        )
        retrieval_candidate_ids: Optional[List[str]] = numeric_filter_result["candidate_ids"]
        after_numeric_count = numeric_filter_result["matched_count"]
    else:
        numeric_filter_result = {
            "candidate_ids": metadata_candidate_ids,
            "matched_count": after_metadata_count,
            "conditions": [],
        }
        after_numeric_count = after_metadata_count
        retrieval_candidate_ids = None

    candidate_counts = {
        "before_metadata_filter": before_metadata_count,
        "after_metadata_filter": after_metadata_count,
        "after_numeric_filter": after_numeric_count,
        "bm25_candidates": after_numeric_count,
        "vector_candidates": after_numeric_count,
        "bm25_results": 0,
        "vector_results": 0,
        "rrf_results": 0,
    }
    logger.info(
        "RAG hard-filter candidates: before_metadata=%d after_metadata=%d "
        "after_numeric=%d conditions=%s",
        before_metadata_count,
        after_metadata_count,
        after_numeric_count,
        numeric_conditions,
    )

    if retrieval_candidate_ids == []:
        return {
            "query": query_text,
            "filters": normalized_filters or None,
            "inferred_filters": inferred_filters,
            "numeric_conditions": numeric_conditions,
            "candidate_counts": candidate_counts,
            "results": [],
            "context": "No matching historical incidents were found.",
            "answer": "No matching historical incidents were found.",
            "evidence_available": False,
        }

    search_arguments = {
        "top_k": top_k,
        "filters": normalized_filters or None,
    }
    if retrieval_candidate_ids is not None:
        search_arguments["candidate_ids"] = retrieval_candidate_ids
    results = searcher.search(query_text, **search_arguments)
    candidate_counts.update(
        {
            "bm25_results": len(results.get("bm25", []) or []),
            "vector_results": len(results.get("vector", []) or []),
            "rrf_results": len(results.get("fused", []) or []),
        }
    )
    records = _result_records(results, df)

    allowed_ids = (
        set(retrieval_candidate_ids)
        if retrieval_candidate_ids is not None
        else None
    )
    violations: List[str] = []
    valid_records: List[Dict[str, Any]] = []
    for record in records:
        incident_id = str(record["incident_id"])
        valid = allowed_ids is None or incident_id in allowed_ids
        valid = valid and (
            not normalized_filters
            or matches_metadata(record.get("metadata", {}), normalized_filters)
        )
        if valid:
            valid_records.append(record)
        else:
            violations.append(incident_id)

    if numeric_conditions and valid_records:
        numeric_validation = apply_numeric_filters(
            df,
            numeric_conditions,
            candidate_ids=[record["incident_id"] for record in valid_records],
        )
        numeric_valid_ids = set(numeric_validation["candidate_ids"])
        for record in valid_records:
            if str(record["incident_id"]) not in numeric_valid_ids:
                violations.append(str(record["incident_id"]))
        valid_records = [
            record
            for record in valid_records
            if str(record["incident_id"]) in numeric_valid_ids
        ]

    if violations:
        logger.error(
            "Hybrid retrieval returned incidents violating hard filters: %s",
            sorted(set(violations)),
        )
    records = valid_records

    if not records:
        no_result = {
            "query": query_text,
            "filters": normalized_filters or None,
            "inferred_filters": inferred_filters,
            "numeric_conditions": numeric_conditions,
            "candidate_counts": candidate_counts,
            "results": [],
            "context": "No matching historical incidents were found.",
            "answer": "No matching historical incidents were found.",
            "evidence_available": False,
        }
        return no_result

    context = build_context(records, max_results=min(top_k, len(records)))
    if numeric_conditions:
        condition_lines = [
            f"{condition['feature']} {condition['operator']} {condition['value']} "
            f"{condition['unit']} ({condition.get('threshold_source', condition['source'])})"
            for condition in numeric_conditions
        ]
        context = (
            "Applied hard conditions before retrieval:\n"
            + "\n".join(condition_lines)
            + f"\nEligible candidates: {after_numeric_count}\n\n"
            + context
        )
    prompt = build_prompt(query_text, context)

    if llm_client is None:
        llm_client = LLMClient()

    try:
        answer = llm_client.generate(prompt)
    except Exception as exc:  # pragma: no cover - provider failures depend on external API state.
        answer = f"LLM generation failed: {exc}"

    return {
        "query": query_text,
        "filters": normalized_filters or None,
        "inferred_filters": inferred_filters,
        "numeric_conditions": numeric_conditions,
        "candidate_counts": candidate_counts,
        "results": records,
        "context": context,
        "prompt": prompt,
        "answer": answer,
        "evidence_available": True,
        "search_results": results,
    }


class RAGPipeline:
    """Simple wrapper around rag_pipeline with a reusable searcher and client."""

    def __init__(
        self,
        searcher: Optional[HybridSearcher] = None,
        df: Optional[pd.DataFrame] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.searcher = searcher
        self.df = df
        self.llm_client = llm_client

    def run(self, query: Any, filters: Optional[Mapping[str, Any]] = None, top_k: Optional[int] = None) -> Dict[str, Any]:
        return rag_pipeline(
            query=query,
            filters=filters,
            top_k=top_k,
            searcher=self.searcher,
            df=self.df,
            llm_client=self.llm_client,
        )


__all__ = ["RAGPipeline", "rag_pipeline", "validate_query"]
