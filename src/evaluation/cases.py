"""
The labelled question set and the conversion of a stored run into DeepEval test cases.

A case is a question plus the facts we know to be true for it (from the dataset's historical
labels and the prioritiser). A run record is what the assistant produced for that question:
answer, route, evidence (every tool call with its result), guardrail results and trace.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

import evaluation  # noqa: F401  (configures DeepEval before it is imported)

CASES_FILE = Path(__file__).resolve().parent / "cases" / "deepeval_cases.json"

GROUPS = ("classify", "risk", "search", "recommend", "investigate", "general", "guardrails")
CONTEXT_ITEM_CHARS = 2500


class Expect(BaseModel):
    """Known facts for a case; each field present becomes a deterministic check."""

    model_config = ConfigDict(extra="forbid")

    route: Optional[str] = None
    attack_type: Optional[str] = None          # single-flow prediction
    priority: Optional[str] = None             # P1..P4 of the single flow
    top_incident: Optional[str] = None         # first incident of a ranking
    mentions: List[str] = Field(default_factory=list)  # incident IDs the answer must cite
    profile: Optional[str] = None              # attack-type profile the agents must use
    search_attack_type: Optional[str] = None   # majority attack type of the incidents found
    search_port: Optional[int] = None          # destination-port filter applied
    numeric_condition: bool = False            # a numeric condition parsed from the query
    caveat: bool = False                       # answer must say the input was partial / less reliable
    blocked: bool = False                      # input guardrail blocks it
    redacted: bool = False                     # input guardrail redacts a secret ...
    secret: Optional[str] = None               # ... which must not reach the answer or the tools
    rejected: bool = False                     # an impossible value: no prediction may be made


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    group: str
    question: str
    expect: Expect = Field(default_factory=Expect)


def load_cases(path: Path = CASES_FILE, groups: Optional[List[str]] = None,
               ids: Optional[List[str]] = None) -> List[Case]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [Case.model_validate(c) for c in data["cases"]]
    seen = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"Duplicate case id {case.id}")
        if case.group not in GROUPS:
            raise ValueError(f"Case {case.id}: unknown group {case.group!r}")
        seen.add(case.id)
    if groups:
        cases = [c for c in cases if c.group in groups]
    if ids:
        cases = [c for c in cases if c.id in ids]
    return cases


# ============================================================================
# Evidence → retrieval context
# ============================================================================
def _compact(value: Any, limit: int = CONTEXT_ITEM_CHARS) -> str:
    text = json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def search_context(evidence: List[Dict[str, Any]]) -> List[str]:
    """One item per retrieved incident (what the search returned)."""
    items = []
    for call in evidence:
        if call.get("tool") != "search_similar_incidents":
            continue
        for incident in (call.get("result") or {}).get("incidents", []):
            items.append(f"{incident.get('incident_id')} ({incident.get('attack_type')}, port "
                         f"{incident.get('dst_port')}): {incident.get('summary', '')}")
    return items


def evidence_context(evidence: List[Dict[str, Any]]) -> List[str]:
    """Everything the answer may rely on: one item per tool call, search results split per incident."""
    items = []
    for call in evidence:
        result = call.get("result") or {}
        if call.get("tool") == "search_similar_incidents" and result.get("incidents"):
            header = {k: v for k, v in result.items() if k != "incidents"}
            items.append(f"search_similar_incidents({_compact(call.get('args', {}), 300)}): {_compact(header, 800)}")
            items.extend(search_context([call]))
        else:
            items.append(f"{call.get('tool')}({_compact(call.get('args', {}), 300)}): {_compact(result)}")
    return items


def is_answered(record: Dict[str, Any]) -> bool:
    """An answer DeepEval can grade: generated, not blocked, not a follow-up question."""
    return (record.get("status") == "ok" and bool(record.get("answer"))
            and record.get("route") not in ("blocked", "out_of_scope") and not record.get("needs_follow_up"))


def build_test_case(record: Dict[str, Any]):
    """DeepEval LLMTestCase for a stored run record."""
    from deepeval.test_case import LLMTestCase

    evidence = record.get("evidence") or []
    return LLMTestCase(
        input=record["question"],
        actual_output=record.get("answer") or "",
        retrieval_context=evidence_context(evidence) or None,
        name=record["case_id"],
        tags=[record.get("group", ""), record.get("route") or ""],
        metadata={"route": record.get("route"), "search_context": search_context(evidence)},
    )
