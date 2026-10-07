"""Format historical retrieval results into a compact, deterministic LLM context."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List


def _as_text(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def _normalize_incidents(results: Any) -> List[Dict[str, Any]]:
    """Accept either a list of incident dicts or the existing HybridSearcher result dict."""
    if results is None:
        return []
    if isinstance(results, list):
        return [dict(item) for item in results]

    if not isinstance(results, dict):
        raise TypeError("Expected results to be a mapping or a list of incident dicts")

    ordered_ids: List[str] = []
    seen = set()
    for key in ("fused", "bm25", "vector"):
        for item in results.get(key, []) or []:
            if key == "vector":
                incident_id = item[0] if len(item) >= 3 else None
                score = item[1] if len(item) >= 2 else None
            else:
                incident_id = item[0] if len(item) >= 2 else None
                score = item[1] if len(item) >= 2 else None
            if incident_id is None:
                continue
            incident_id = str(incident_id)
            if incident_id not in seen:
                seen.add(incident_id)
                ordered_ids.append(incident_id)

    metadata_map: Dict[str, Any] = dict(results.get("metadata") or {})
    texts_map: Dict[str, Any] = dict(results.get("incident_texts") or {})
    incidents: List[Dict[str, Any]] = []
    for incident_id in ordered_ids:
        source = metadata_map.get(incident_id, {}) if isinstance(metadata_map, dict) else {}
        score = None
        for key in ("fused", "bm25", "vector"):
            for item in results.get(key, []) or []:
                if key == "vector":
                    candidate_id = item[0] if len(item) >= 3 else None
                    candidate_score = item[1] if len(item) >= 2 else None
                else:
                    candidate_id = item[0] if len(item) >= 2 else None
                    candidate_score = item[1] if len(item) >= 2 else None
                if str(candidate_id) == incident_id:
                    score = candidate_score
                    break
            if score is not None:
                break

        incident = {
            "incident_id": incident_id,
            "label": source.get("label", "N/A"),
            "incident_text": texts_map.get(incident_id, source.get("incident_text", "N/A")),
            "metadata": dict(source)
        }
        if score is not None:
            incident["score"] = score
        incidents.append(incident)
    return incidents


def build_context(results: Any, max_results: int | None = None) -> str:
    """Convert retrieved incidents into a clean, deterministic context block."""
    if max_results is None:
        max_results = int(os.getenv("RAG_CONTEXT_TOP_K", "5"))
    if max_results < 1:
        raise ValueError("max_results must be a positive integer")

    incidents = _normalize_incidents(results)
    if not incidents:
        return "No relevant historical incidents were found."

    selected = incidents[:max_results]
    blocks: List[str] = []
    for index, incident in enumerate(selected, start=1):
        metadata = incident.get("metadata") or {}
        incident_id = _as_text(incident.get("incident_id"))
        label = _as_text(incident.get("label", metadata.get("label", "N/A")))
        metadata_port = metadata.get("dst_port", incident.get("dst_port"))
        destination_port = _as_text(metadata_port) if metadata_port is not None else "N/A"
        retrieval_score = incident.get("score")
        if retrieval_score is None:
            retrieval_score = "N/A"
        else:
            retrieval_score = float(retrieval_score)
            if math.isnan(retrieval_score):
                retrieval_score = "N/A"
            elif retrieval_score.is_integer():
                retrieval_score = int(retrieval_score)
            else:
                retrieval_score = retrieval_score
        incident_text = incident.get("incident_text") or incident.get("text") or "N/A"
        incident_text = _as_text(incident_text)

        additional_fields = [
            (field, value)
            for field, value in sorted(metadata.items())
            if field not in {"incident_id", "label", "dst_port", "incident_text"}
            and value is not None
            and not (isinstance(value, float) and math.isnan(value))
        ]
        additional_context = ""
        if additional_fields:
            additional_context = "\n\nAdditional Incident Fields:\n" + "\n".join(
                f"{field}: {_as_text(value)}"
                for field, value in additional_fields
            )

        block = (
            f"=== HISTORICAL INCIDENT {index} ===\n"
            f"Incident ID: {incident_id}\n"
            f"Attack Type: {label}\n"
            f"Destination Port: {destination_port}\n"
            f"Retrieval Score: {retrieval_score}\n\n"
            "Traffic:\n"
            f"{incident_text}"
            f"{additional_context}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


__all__ = ["build_context"]
