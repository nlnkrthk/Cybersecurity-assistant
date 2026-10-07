"""Canonical incident metadata and validation shared by retrieval indexes."""

from datetime import date, datetime
from numbers import Real
from typing import Any, Dict, List

import numpy as np
import pandas as pd


REQUIRED_INDEX_COLUMNS = ("incident_id", "label", "incident_text", "dst_port")
OPTIONAL_METADATA_COLUMNS = (
    "protocol",
    "src_port",
    "src_ip",
    "dst_ip",
    "timestamp",
)


def validate_incident_frame(frame: pd.DataFrame) -> None:
    """Validate the canonical columns and ID invariants used by both indexes."""
    missing = [column for column in REQUIRED_INDEX_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Incident data is missing required retrieval columns: "
            f"{', '.join(missing)}. Re-run scripts/data/prepare_incident_data.py."
        )
    if frame["incident_id"].isna().any() or frame["incident_id"].astype(str).str.strip().eq("").any():
        raise ValueError("incident_id values must be present and non-empty")
    if frame["incident_id"].duplicated().any():
        raise ValueError("incident_id values must be unique before indexing")
    if frame["label"].isna().any():
        raise ValueError("label values must be present before indexing")
    if frame["incident_text"].isna().any() or frame["incident_text"].astype(str).str.strip().eq("").any():
        raise ValueError("incident_text values must be present and non-empty before indexing")
    ports = pd.to_numeric(frame["dst_port"], errors="coerce")
    invalid_ports = frame["dst_port"].notna() & ports.isna()
    if invalid_ports.any():
        raise ValueError("dst_port values must be numeric when present")


def _metadata_value(value: Any):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            return None
        if numeric_value.is_integer():
            return int(numeric_value)
        return numeric_value
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def build_incident_metadata(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build metadata records in frame order, preserving only present values."""
    validate_incident_frame(frame)
    metadata_list: List[Dict[str, Any]] = []

    for _, row in frame.iterrows():
        metadata: Dict[str, Any] = {
            "incident_id": str(row["incident_id"]),
            "label": str(row["label"]),
        }
        for column in OPTIONAL_METADATA_COLUMNS:
            if column not in frame.columns:
                continue
            value = _metadata_value(row[column])
            if value is not None:
                metadata[column] = value

        destination_port = _metadata_value(row["dst_port"])
        if destination_port is not None:
            metadata["dst_port"] = destination_port
        metadata_list.append(metadata)

    return metadata_list
