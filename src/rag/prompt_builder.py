"""Prompt construction for the APCIRA RAG pipeline."""

from __future__ import annotations

from typing import Any


DEFAULT_SYSTEM_PROMPT = """You are a cybersecurity incident response assistant.

Your task is to analyze a user's cybersecurity query using the provided historical incident data.

The historical incidents are evidence retrieved from a cybersecurity incident database. They are not the user's current incident.

Use the retrieved incidents to:
1. identify similar historical attack patterns,
2. explain relevant traffic characteristics,
3. identify the likely attack type when supported by the evidence,
4. explain why the historical incidents are relevant,
5. provide practical investigation and response recommendations.

IMPORTANT RULES:
- Use only the provided historical incident context for incident-specific claims.
- Do not invent evidence.
- Do not fabricate IP addresses, ports, timestamps, attack types, or traffic values.
- If the retrieved evidence is insufficient, explicitly say that the evidence is insufficient.
- Do not claim that a historical incident is the user's current incident.
- Clearly distinguish historical evidence from your analysis.
- Do not make unsupported claims.
- If multiple historical incidents disagree, explain the uncertainty.
- Prefer evidence from multiple retrieved incidents when available.
- Reference incident IDs when discussing historical evidence.

Return a clear and concise cybersecurity analysis.
"""


def build_prompt(query: Any, context: Any) -> str:
    """Build the final prompt that combines the user question with historical evidence."""
    query_text = "" if query is None else str(query).strip()
    if not query_text:
        raise ValueError("Query must be a non-empty string")

    context_text = "" if context is None else str(context)
    if not context_text:
        context_text = "No relevant historical incidents were found."

    return (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        "USER QUERY:\n"
        f"{query_text}\n\n"
        "HISTORICAL INCIDENT CONTEXT:\n"
        f"{context_text}\n\n"
        "Provide a structured answer using this format:\n"
        "1. Assessment\n"
        "2. Relevant Historical Evidence\n"
        "3. Why It Is Relevant\n"
        "4. Recommended Investigation\n"
        "5. Recommended Response\n"
        "6. Confidence / Limitations\n\n"
        "Be clear, concise, and evidence-based."
    )


__all__ = ["build_prompt", "DEFAULT_SYSTEM_PROMPT"]
