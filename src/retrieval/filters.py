"""
Build metadata filters for vector and hybrid search queries.
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


def to_chroma_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize scalar equality filters to Chroma's explicit where syntax."""
    if not filters:
        return None

    normalized: Dict[str, Any] = {}
    for key, condition in filters.items():
        if key in ("$and", "$or"):
            normalized[key] = [to_chroma_where(item) for item in condition]
        elif isinstance(condition, dict):
            normalized[key] = condition
        else:
            normalized[key] = {"$eq": condition}
    if len(normalized) > 1:
        return {"$and": [{key: value} for key, value in normalized.items()]}
    return normalized


def filter_field_names(filters: Optional[Dict[str, Any]]) -> set:
    """Return metadata field names referenced by a filter expression."""
    fields = set()
    if not filters:
        return fields
    for key, condition in filters.items():
        if key in ("$and", "$or"):
            for item in condition:
                fields.update(filter_field_names(item))
        else:
            fields.add(key)
    return fields


def matches_metadata(metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    """Evaluate the equality and comparison filters supported by Chroma."""
    if not filters:
        return True

    for key, condition in filters.items():
        if key == "$and":
            if not all(matches_metadata(metadata, item) for item in condition):
                return False
            continue
        if key == "$or":
            if not any(matches_metadata(metadata, item) for item in condition):
                return False
            continue

        actual = metadata.get(key)
        operators = condition if isinstance(condition, dict) else {"$eq": condition}
        for operator, expected in operators.items():
            if operator not in {"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"}:
                raise ValueError(f"Unsupported metadata filter operator: {operator}")
            if operator == "$eq" and actual != expected:
                return False
            if operator == "$ne" and actual == expected:
                return False
            if operator == "$in" and actual not in expected:
                return False
            if operator == "$nin" and actual in expected:
                return False
            if operator in ("$gt", "$gte", "$lt", "$lte"):
                if actual is None:
                    return False
                try:
                    if operator == "$gt" and not actual > expected:
                        return False
                    if operator == "$gte" and not actual >= expected:
                        return False
                    if operator == "$lt" and not actual < expected:
                        return False
                    if operator == "$lte" and not actual <= expected:
                        return False
                except TypeError:
                    return False
    return True


class FilterBuilder:
    """Build Chroma-compatible metadata filters."""

    @staticmethod
    def attack_type_filter(attack_type: str) -> Dict[str, Any]:
        """
        Filter by attack type (label).

        Args:
            attack_type: Attack label (e.g., 'DoS Hulk', 'BENIGN').

        Returns:
            Chroma filter dict.
        """
        return {"label": {"$eq": attack_type}}

    @staticmethod
    def protocol_filter(protocol: int) -> Dict[str, Any]:
        """
        Filter by protocol number (6=TCP, 17=UDP, etc.).

        Args:
            protocol: Protocol number.

        Returns:
            Chroma filter dict.
        """
        return {"protocol": {"$eq": protocol}}

    @staticmethod
    def port_filter(port: int, direction: str = "dst") -> Dict[str, Any]:
        """
        Filter by port number.

        Args:
            port: Port number.
            direction: 'src' for source, 'dst' for destination.

        Returns:
            Chroma filter dict.
        """
        if direction not in ("src", "dst"):
            raise ValueError("direction must be 'src' or 'dst'")
        port_col = "src_port" if direction == "src" else "dst_port"
        return {port_col: {"$eq": port}}

    @staticmethod
    def ip_filter(ip: str, direction: str = "src") -> Dict[str, Any]:
        """
        Filter by IP address.

        Args:
            ip: IP address string.
            direction: 'src' for source, 'dst' for destination.

        Returns:
            Chroma filter dict.
        """
        if direction not in ("src", "dst"):
            raise ValueError("direction must be 'src' or 'dst'")
        ip_col = "src_ip" if direction == "src" else "dst_ip"
        return {ip_col: {"$eq": ip}}

    @staticmethod
    def combine_filters(filters: List[Dict[str, Any]], operator: str = "$and") -> Dict[str, Any]:
        """
        Combine multiple filters with AND or OR logic.

        Args:
            filters: List of filter dicts.
            operator: '$and' or '$or'.

        Returns:
            Combined Chroma filter dict.
        """
        if not filters:
            return {}
        if operator not in ("$and", "$or"):
            raise ValueError("operator must be '$and' or '$or'")
        if len(filters) == 1:
            return filters[0]

        return {operator: filters}
