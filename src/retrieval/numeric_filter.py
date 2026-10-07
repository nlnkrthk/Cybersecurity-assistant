"""Resolve numeric traffic conditions and select eligible incident IDs."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


_NUMBER = r"\d[\d,]*(?:\.\d+)?"
_OPERATOR = (
    r"at\s+least|no\s+less\s+than|greater\s+than|more\s+than|above|over|"
    r"at\s+most|no\s+more\s+than|less\s+than|below|under|>=|<=|>|<"
)
_OPERATOR_MAP = {
    "at least": ">=",
    "no less than": ">=",
    "greater than": ">",
    "more than": ">",
    "above": ">",
    "over": ">",
    "at most": "<=",
    "no more than": "<=",
    "less than": "<",
    "below": "<",
    "under": "<",
    ">=": ">=",
    "<=": "<=",
    ">": ">",
    "<": "<",
}
_RATE_UNITS = ("packets per second", "packet per second", "packets/sec", "packet/sec", "packets/s", "pps")


def _parse_number(value: str) -> float:
    return float(value.replace(",", ""))


def _operator(value: str) -> str:
    return _OPERATOR_MAP[re.sub(r"\s+", " ", value.casefold()).strip()]


def _make_condition(feature: str, operator: str, value: float, unit: str, source: str) -> Dict[str, Any]:
    if feature == "flow_duration":
        normalized_value = value * 1_000_000 if unit in {"second", "seconds", "s"} else value
        normalized_unit = "microseconds"
    else:
        normalized_value = value
        normalized_unit = unit

    return {
        "feature": feature,
        "operator": operator,
        "value": normalized_value,
        "unit": normalized_unit,
        "source": source,
    }


def resolve_numeric_conditions(
    query: str,
    incidents: pd.DataFrame,
    candidate_ids: Iterable[str],
) -> List[Dict[str, Any]]:
    """Resolve explicit numeric phrases and P75 qualitative packet-volume requests."""
    conditions: List[Dict[str, Any]] = []
    ids = {str(incident_id) for incident_id in candidate_ids}
    scoped = incidents.loc[incidents["incident_id"].astype(str).isin(ids)]
    normalized_query = query.casefold()

    if re.search(r"\bhigh[- ](?:packet|packets|traffic)\s+volume\b", normalized_query):
        if "flow_packets_s" not in scoped.columns:
            raise ValueError(
                "Cannot resolve 'high packet volume': flow_packets_s is unavailable in incident data"
            )
        rates = pd.to_numeric(scoped["flow_packets_s"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if rates.empty:
            raise ValueError(
                "Cannot resolve 'high packet volume': no packet-rate values are available "
                "for the metadata-filtered incidents"
            )
        threshold = float(rates.quantile(0.75))
        scope_label = (
            str(scoped["label"].iloc[0])
            if "label" in scoped.columns and scoped["label"].nunique(dropna=True) == 1
            else "metadata-filtered"
        )
        conditions.append(
            {
                "feature": "flow_packets_per_sec",
                "column": "flow_packets_s",
                "operator": ">=",
                "value": threshold,
                "unit": "packets/s",
                "source": "qualitative",
                "concept": "high_packet_volume",
                "threshold_source": f"{scope_label} P75",
                "percentile": 75,
                "sample_count": int(len(rates)),
            }
        )

    explicit_patterns = [
        re.compile(
            rf"\b(?P<operator>{_OPERATOR})\s+(?P<value>{_NUMBER})\s*"
            rf"(?P<unit>packets\s+per\s+second|packet\s+per\s+second|"
            rf"packets/sec|packet/sec|packets/s|pps|packets?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<feature>packet\s+rate|packet\s+volume|packets\s+per\s+second|"
            rf"packets/sec|packets/s|pps|packets)\s+"
            rf"(?P<operator>{_OPERATOR})\s*(?P<value>{_NUMBER})\s*"
            rf"(?P<unit>packets\s+per\s+second|packet\s+per\s+second|"
            rf"packets/sec|packet/sec|packets/s|pps|packets?)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\bduration\s+(?P<operator>{_OPERATOR})\s*(?P<value>{_NUMBER})\s*"
            r"(?P<unit>microseconds?|milliseconds?|seconds?|minutes?|hours?|us|ms|s)\b",
            re.IGNORECASE,
        ),
    ]

    for pattern in explicit_patterns:
        for match in pattern.finditer(query):
            groups = match.groupdict()
            unit = (groups.get("unit") or "").casefold()
            if "duration" in match.group(0).casefold():
                feature = "flow_duration"
                scale = {
                    "microsecond": 1,
                    "microseconds": 1,
                    "us": 1,
                    "millisecond": 1_000,
                    "milliseconds": 1_000,
                    "ms": 1_000,
                    "second": 1_000_000,
                    "seconds": 1_000_000,
                    "s": 1_000_000,
                    "minute": 60_000_000,
                    "minutes": 60_000_000,
                    "hour": 3_600_000_000,
                    "hours": 3_600_000_000,
                }[unit]
                condition = _make_condition(
                    feature,
                    _operator(groups["operator"]),
                    _parse_number(groups["value"]) * scale,
                    "microseconds",
                    "explicit",
                )
            else:
                feature_text = groups.get("feature", "").casefold()
                is_rate = (
                    any(rate_unit in unit for rate_unit in _RATE_UNITS)
                    or "rate" in feature_text
                    or "per second" in feature_text
                    or "pps" in feature_text
                )
                feature = "flow_packets_per_sec" if is_rate else "total_packets"
                condition = _make_condition(
                    feature,
                    _operator(groups["operator"]),
                    _parse_number(groups["value"]),
                    "packets/s" if is_rate else "packets",
                    "explicit",
                )
            if not any(
                existing["feature"] == condition["feature"]
                and existing["operator"] == condition["operator"]
                and existing["value"] == condition["value"]
                for existing in conditions
            ):
                conditions.append(condition)

    return conditions


def _feature_values(frame: pd.DataFrame, feature: str) -> pd.Series:
    if feature == "total_packets":
        required = {"total_fwd_packets", "total_bwd_packets"}
        if not required.issubset(frame.columns):
            raise ValueError("Cannot filter total_packets: packet count columns are unavailable")
        forward = pd.to_numeric(frame["total_fwd_packets"], errors="coerce")
        backward = pd.to_numeric(frame["total_bwd_packets"], errors="coerce")
        return (forward + backward).where(forward.notna() & backward.notna())

    column = "flow_packets_s" if feature == "flow_packets_per_sec" else feature
    if column not in frame.columns:
        raise ValueError(f"Cannot filter numeric feature {feature!r}: column {column!r} is unavailable")
    return pd.to_numeric(frame[column], errors="coerce")


def condition_matches(value: Any, condition: Mapping[str, Any]) -> bool:
    if value is None or pd.isna(value):
        return False
    operator = condition["operator"]
    threshold = condition["value"]
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    raise ValueError(f"Unsupported numeric filter operator: {operator}")


def apply_numeric_filters(
    incidents: pd.DataFrame,
    conditions: List[Mapping[str, Any]],
    candidate_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Apply every numeric condition with AND semantics and return eligible IDs."""
    if candidate_ids is None:
        eligible = set(incidents["incident_id"].astype(str))
    else:
        eligible = {str(incident_id) for incident_id in candidate_ids}

    scoped = incidents.loc[incidents["incident_id"].astype(str).isin(eligible)]
    mask = pd.Series(True, index=scoped.index)
    for condition in conditions:
        values = _feature_values(scoped, str(condition["feature"]))
        mask &= values.map(lambda value: condition_matches(value, condition))

    candidate_result = scoped.loc[mask, "incident_id"].astype(str).tolist()
    return {
        "candidate_ids": candidate_result,
        "matched_count": len(candidate_result),
        "conditions": [dict(condition) for condition in conditions],
    }


__all__ = ["apply_numeric_filters", "condition_matches", "resolve_numeric_conditions"]
