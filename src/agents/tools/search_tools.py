"""
LangGraph tool around the existing hybrid RAG retrieval (BM25 + vectors + metadata
and numeric hard filters).

The tool returns the retrieved incidents only. Answer generation is left to the
agent, so rag_pipeline() is called with a client that skips the LLM call.
"""
import logging
import re
import threading
from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agents.tools.common import error_result, rounded, safe_tool
from agents.tools.ml_tools import match_attack_type
from rag.rag_pipeline import _default_hybrid_searcher, rag_pipeline

logger = logging.getLogger(__name__)

# Sentences of incident_text kept in the tool output (label, port, duration, packets, bytes, rates).
SUMMARY_SENTENCES = 10

_searcher = None
_frame = None
_searcher_lock = threading.Lock()


class _RetrievalOnlyClient:
    """Stands in for LLMClient so rag_pipeline() returns retrieval results without generating."""

    def generate(self, *args, **kwargs) -> str:
        return ""


def _silence_progress_bar(searcher) -> None:
    """Query embedding inside a tool call should not print a tqdm bar (embedder.py forces it on)."""
    model = getattr(getattr(searcher, "embedder", None), "model", None)
    if model is None or getattr(model, "_quiet_encode", False):
        return
    encode = model.encode

    def quiet_encode(*args, **kwargs):
        kwargs["show_progress_bar"] = False
        return encode(*args, **kwargs)

    model.encode = quiet_encode
    model._quiet_encode = True


def _get_searcher():
    """Load the BM25 index, Chroma store and embedder once (slow first call)."""
    global _searcher, _frame
    if _searcher is None:
        with _searcher_lock:
            if _searcher is None:
                logger.info("Loading hybrid search indexes")
                searcher, frame = _default_hybrid_searcher()
                _silence_progress_bar(searcher)
                _searcher, _frame = searcher, frame
    return _searcher, _frame


def _short_text(text: str) -> str:
    sentences = re.split(r"(?<=\.)\s+", str(text).strip())
    return " ".join(sentences[:SUMMARY_SENTENCES])


class SearchArgs(BaseModel):
    query: str = Field(
        description=(
            "Natural-language description of the traffic or incident to find similar historical incidents for. "
            "May include numeric conditions, e.g. 'more than 10000 packets per second' or 'duration under 1 ms'."
        )
    )
    attack_type: Optional[str] = Field(
        default=None, description="Only return incidents of this attack type, e.g. 'DDoS' or 'PortScan'."
    )
    dst_port: Optional[int] = Field(
        default=None, ge=0, le=65535, description="Only return incidents with this destination port, e.g. 22 or 80."
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of incidents to return.")


@safe_tool
def search_similar_incidents(
    query: str, attack_type: Optional[str] = None, dst_port: Optional[int] = None, top_k: int = 5
) -> Dict[str, Any]:
    if not query or not query.strip():
        return error_result("query must be a non-empty description of the traffic to search for.")

    filters: Dict[str, Any] = {}
    if attack_type:
        filters["label"] = match_attack_type(attack_type)
    if dst_port is not None:
        filters["dst_port"] = int(dst_port)

    searcher, frame = _get_searcher()
    result = rag_pipeline(
        query,
        filters=filters or None,
        top_k=top_k,
        searcher=searcher,
        df=frame,
        llm_client=_RetrievalOnlyClient(),
    )

    incidents = [
        {
            "incident_id": record["incident_id"],
            "attack_type": record["label"],
            "dst_port": record.get("metadata", {}).get("dst_port"),
            "relevance_score": record.get("score"),
            "summary": _short_text(record.get("incident_text", "")),
        }
        for record in result["results"]
    ]
    return rounded({
        "status": "ok",
        "query": result["query"],
        "filters_applied": result.get("filters"),
        "filters_inferred_from_query": result.get("inferred_filters"),
        "numeric_conditions": [
            {k: c.get(k) for k in ("feature", "operator", "value", "unit")}
            for c in result.get("numeric_conditions", [])
        ],
        "candidates_after_filters": result.get("candidate_counts", {}).get("after_numeric_filter"),
        "evidence_available": result.get("evidence_available", False),
        "incidents": incidents,
    })


search_similar_incidents_tool = StructuredTool.from_function(
    func=search_similar_incidents,
    name="search_similar_incidents",
    description=(
        "Find similar historical network-security incidents in the CIC-IDS2017 incident base using hybrid "
        "keyword + semantic search, with optional attack-type and destination-port filters and numeric "
        "conditions stated in the query. Returns incident IDs, attack types, relevance scores and metric "
        "summaries. Use for 'has this happened before?' and to ground recommendations in past incidents."
    ),
    args_schema=SearchArgs,
)

SEARCH_TOOLS = [search_similar_incidents_tool]
