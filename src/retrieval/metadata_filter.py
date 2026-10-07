"""Validate structured filters against the metadata available in an index."""

from numbers import Real
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from retrieval.filters import filter_field_names, matches_metadata


FIELD_ALIASES = {
    "attack_type": "label",
    "destination_port": "dst_port",
}
SUPPORTED_OPERATORS = {"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"}


class MetadataFilter:
    """Validate filters and find eligible incident IDs from stored metadata."""

    def __init__(self, metadata: List[Dict[str, Any]]):
        self.metadata = metadata
        self.filterable_fields = {
            field
            for record in metadata
            for field, value in record.items()
            if field != "incident_id" and value is not None
        }
        self._sample_values = {
            field: next(
                record[field]
                for record in metadata
                if field in record and record[field] is not None
            )
            for field in self.filterable_fields
        }

    def validate(
        self, filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return canonical filters, coercing values to the stored field types."""
        if filters is None:
            return {}
        if not isinstance(filters, dict):
            raise ValueError("Metadata filters must be a field-to-value mapping")

        normalized: Dict[str, Any] = {}
        for requested_field, condition in filters.items():
            if not isinstance(requested_field, str):
                raise ValueError("Metadata filter field names must be strings")
            field = FIELD_ALIASES.get(requested_field, requested_field)
            if field not in self.filterable_fields:
                supported = ", ".join(sorted(self.filterable_fields)) or "none"
                raise ValueError(
                    f"Metadata filter field {requested_field!r} is unavailable; "
                    f"available fields: {supported}"
                )
            if field in normalized:
                raise ValueError(
                    f"Metadata field {field!r} was specified more than once"
                )

            sample = self._sample_values[field]
            if isinstance(condition, dict):
                if not condition:
                    raise ValueError(f"Filter for {requested_field!r} cannot be empty")
                unknown_operators = set(condition) - SUPPORTED_OPERATORS
                if unknown_operators:
                    raise ValueError(
                        "Unsupported metadata filter operator(s): "
                        f"{', '.join(sorted(unknown_operators))}"
                    )
                normalized[field] = {
                    operator: self._coerce_value(field, sample, value, operator)
                    for operator, value in condition.items()
                }
            else:
                normalized[field] = self._coerce_value(
                    field, sample, condition, "$eq"
                )
        return normalized

    def eligible_ids(
        self, filters: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Return validated filters and IDs satisfying all filter conditions."""
        normalized = self.validate(filters)
        eligible = [
            str(record["incident_id"])
            for record in self.metadata
            if matches_metadata(record, normalized)
        ]
        return normalized, eligible

    @staticmethod
    def _coerce_value(
        field: str, sample: Any, value: Any, operator: str
    ) -> Any:
        if operator in {"$in", "$nin"}:
            if not isinstance(value, (list, tuple, set)):
                raise ValueError(f"{operator} filter values must be a list")
            return [
                MetadataFilter._coerce_value(field, sample, item, "$eq")
                for item in value
            ]

        if value is None:
            raise ValueError("Metadata filter values cannot be null")
        if isinstance(sample, (int, float, np.number)) and not isinstance(
            sample, (bool, np.bool_)
        ):
            if isinstance(value, bool):
                raise ValueError(f"Metadata filter {field!r} must be numeric")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Metadata filter {field!r} must be numeric"
                ) from exc
            if not np.isfinite(numeric):
                raise ValueError(f"Metadata filter {field!r} must be finite")
            if isinstance(sample, (int, np.integer)) and numeric.is_integer():
                return int(numeric)
            return numeric

        if isinstance(sample, str):
            if not isinstance(value, str):
                raise ValueError(f"Metadata filter {field!r} must be text")
            return value

        if isinstance(sample, bool):
            if not isinstance(value, bool):
                raise ValueError(f"Metadata filter {field!r} must be boolean")
            return value

        if isinstance(sample, Real) and not isinstance(sample, bool):
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Metadata filter {field!r} must be numeric"
                ) from exc

        if not isinstance(value, type(sample)):
            raise ValueError(
                f"Metadata filter {field!r} must have type {type(sample).__name__}"
            )
        return value
