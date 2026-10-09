"""
Turns partial network-flow metrics into the 24-feature vector the ML models expect.

Users (or the LLM extracting metrics from a chat message) rarely provide all 24
features, so this module:
  1. parses numbers and unit strings ("2 s", "1.5 KB", "43k") into model units,
  2. validates values (finite, non-negative, flags 0/1) as an input guardrail,
  3. derives features that can be computed from the ones given (rates, mean sizes),
  4. asks for a follow-up when key features are missing (duration, packets, bytes),
  5. otherwise fills the remaining gaps with BENIGN medians and flags them.

It also looks up the real features of a historical incident by incident ID.
"""
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED_INCIDENTS_FILE, FEATURE_REFERENCE_FILE

logger = logging.getLogger(__name__)


class FlowInputError(ValueError):
    """Raised when a provided value cannot be parsed or is out of range."""


# ============================================================================
# Feature definitions (order matches the trained models)
# ============================================================================
FEATURE_NAMES = (
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_len_fwd",
    "total_len_bwd",
    "flow_packets_s",
    "fwd_packets_s",
    "bwd_packets_s",
    "fwd_packet_length_mean",
    "bwd_packet_length_mean",
    "min_packet_length",
    "max_packet_length",
    "packet_length_mean",
    "packet_length_std",
    "syn_flag_count",
    "ack_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "fin_flag_count",
    "down_up_ratio",
    "init_win_bytes_forward",
    "init_win_bytes_backward",
    "active_mean",
    "idle_mean",
)

# Unit kind per feature: decides which unit strings are accepted.
DURATION_FEATURES = {"flow_duration", "active_mean", "idle_mean"}  # microseconds
BYTE_FEATURES = {
    "total_len_fwd", "total_len_bwd", "fwd_packet_length_mean", "bwd_packet_length_mean",
    "min_packet_length", "max_packet_length", "packet_length_mean", "packet_length_std",
}
WINDOW_FEATURES = {"init_win_bytes_forward", "init_win_bytes_backward"}  # bytes, -1 = not set
COUNT_FEATURES = {"total_fwd_packets", "total_bwd_packets"}
RATE_FEATURES = {"flow_packets_s", "fwd_packets_s", "bwd_packets_s"}  # packets per second
FLAG_FEATURES = {"syn_flag_count", "ack_flag_count", "rst_flag_count", "psh_flag_count", "fin_flag_count"}

# A follow-up question is needed if any of these groups is entirely missing.
KEY_FEATURE_GROUPS = {
    "flow duration": ("flow_duration",),
    "packet counts (forward/backward)": ("total_fwd_packets", "total_bwd_packets"),
    "byte counts (forward/backward)": ("total_len_fwd", "total_len_bwd"),
}

# input_quality thresholds on the number of provided + derived features.
FULL_QUALITY_MIN = 20
REDUCED_QUALITY_MIN = 12

# Friendly names accepted from API callers (normalized key -> feature name).
FEATURE_ALIASES = {
    "duration": "flow_duration",
    "fwd_packets": "total_fwd_packets",
    "forward_packets": "total_fwd_packets",
    "bwd_packets": "total_bwd_packets",
    "backward_packets": "total_bwd_packets",
    "fwd_bytes": "total_len_fwd",
    "forward_bytes": "total_len_fwd",
    "bwd_bytes": "total_len_bwd",
    "backward_bytes": "total_len_bwd",
    "packets_per_second": "flow_packets_s",
    "packet_rate": "flow_packets_s",
    "flow_packets_per_second": "flow_packets_s",
    "average_packet_size": "packet_length_mean",
    "mean_packet_size": "packet_length_mean",
}

# Unit multipliers per kind (input unit -> model unit).
_DURATION_UNITS = {
    "": 1, "us": 1, "µs": 1, "μs": 1, "microsecond": 1, "microseconds": 1,
    "ms": 1e3, "millisecond": 1e3, "milliseconds": 1e3,
    "s": 1e6, "sec": 1e6, "secs": 1e6, "second": 1e6, "seconds": 1e6,
    "min": 6e7, "mins": 6e7, "minute": 6e7, "minutes": 6e7,
}
_BYTE_UNITS = {
    "": 1, "b": 1, "byte": 1, "bytes": 1,
    "kb": 1e3, "k": 1e3, "mb": 1e6, "m": 1e6, "gb": 1e9,
}
_SCALE_SUFFIXES = {"": 1, "k": 1e3, "m": 1e6}
_RATE_UNIT_WORDS = ("packets/s", "packets/sec", "pkts/s", "pkt/s", "pps", "/s", "per second")
_COUNT_UNIT_WORDS = ("packets", "packet", "pkts", "pkt")
_NUMBER_RE = re.compile(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)\s*(.*)$")


# ============================================================================
# Input schema (also used as the LLM tool-call schema by the agents)
# ============================================================================
Number = Optional[Union[float, str]]


def _f(description: str):
    return Field(default=None, description=description)


class FlowFeatures(BaseModel):
    """Network-flow metrics. Every field is optional; give only what is known.
    Numbers are in the stated unit; strings with a unit (e.g. '2 s', '1.5 KB', '43k') are also accepted."""

    model_config = ConfigDict(extra="allow")

    flow_duration: Number = _f("Total flow duration in microseconds (e.g. '2 s' or 2000000).")
    total_fwd_packets: Number = _f("Number of packets sent from source to destination.")
    total_bwd_packets: Number = _f("Number of packets sent from destination back to source.")
    total_len_fwd: Number = _f("Total bytes sent forward (source to destination).")
    total_len_bwd: Number = _f("Total bytes sent backward (destination to source).")
    flow_packets_s: Number = _f("Packets per second over the whole flow.")
    fwd_packets_s: Number = _f("Forward packets per second.")
    bwd_packets_s: Number = _f("Backward packets per second.")
    fwd_packet_length_mean: Number = _f("Mean size of forward packets in bytes.")
    bwd_packet_length_mean: Number = _f("Mean size of backward packets in bytes.")
    min_packet_length: Number = _f("Smallest packet size in bytes.")
    max_packet_length: Number = _f("Largest packet size in bytes.")
    packet_length_mean: Number = _f("Mean packet size in bytes (both directions).")
    packet_length_std: Number = _f("Standard deviation of packet size in bytes.")
    syn_flag_count: Number = _f("1 if the TCP SYN flag was seen, else 0.")
    ack_flag_count: Number = _f("1 if the TCP ACK flag was seen, else 0.")
    rst_flag_count: Number = _f("1 if the TCP RST flag was seen, else 0.")
    psh_flag_count: Number = _f("1 if the TCP PSH flag was seen, else 0.")
    fin_flag_count: Number = _f("1 if the TCP FIN flag was seen, else 0.")
    down_up_ratio: Number = _f("Backward packets divided by forward packets (rounded down).")
    init_win_bytes_forward: Number = _f("Initial TCP window size in the forward direction, bytes (-1 if not set).")
    init_win_bytes_backward: Number = _f("Initial TCP window size in the backward direction, bytes (-1 if not set).")
    active_mean: Number = _f("Mean time the flow was active before going idle, microseconds.")
    idle_mean: Number = _f("Mean time the flow was idle before becoming active, microseconds.")


# ============================================================================
# Output
# ============================================================================
@dataclass
class NormalizedFlow:
    """A complete 24-feature flow plus a record of where each value came from."""

    features: Dict[str, float]
    provided: List[str] = field(default_factory=list)
    derived: List[str] = field(default_factory=list)
    filled: List[str] = field(default_factory=list)
    missing_key_features: List[str] = field(default_factory=list)
    needs_follow_up: bool = False
    input_quality: str = "full"
    warnings: List[str] = field(default_factory=list)

    @property
    def known_count(self) -> int:
        return len(self.provided) + len(self.derived)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["known_count"] = self.known_count
        return data


@dataclass
class IncidentFlow:
    """Features of a historical incident looked up by ID."""

    incident_id: str
    historical_label: str
    dst_port: Optional[int]
    flow: NormalizedFlow


# ============================================================================
# Reference data
# ============================================================================
@lru_cache(maxsize=1)
def load_feature_reference() -> Dict[str, Any]:
    """Benign medians, quantiles, display names and units (built by ml/explainability.ipynb)."""
    if not FEATURE_REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Feature reference not found: {FEATURE_REFERENCE_FILE}")
    reference = json.loads(FEATURE_REFERENCE_FILE.read_text(encoding="utf-8"))
    missing = set(FEATURE_NAMES) - set(reference["features"])
    if missing:
        raise ValueError(f"Feature reference is missing features: {sorted(missing)}")
    return reference


def benign_medians() -> Dict[str, float]:
    features = load_feature_reference()["features"]
    return {name: float(features[name]["benign_median"]) for name in FEATURE_NAMES}


# ============================================================================
# Parsing and validation
# ============================================================================
def _canonical_key(key: str) -> str:
    normalized = re.sub(r"[\s\-/]+", "_", str(key).strip().lower())
    return FEATURE_ALIASES.get(normalized, normalized)


def parse_quantity(feature: str, value: Any) -> Optional[float]:
    """Convert a number or a unit string into the model unit for `feature`.

    Returns None for None/empty input. Raises FlowInputError for anything unparseable.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise FlowInputError(f"{feature}: value must be a finite number, got {value!r}")
        return number

    text = str(value).strip().lower().replace(",", "")
    if text == "":
        return None
    match = _NUMBER_RE.match(text)
    if not match:
        raise FlowInputError(f"{feature}: could not read a number from {value!r}")
    number, unit = float(match.group(1)), match.group(2).strip()

    if feature in DURATION_FEATURES:
        units = _DURATION_UNITS
    elif feature in BYTE_FEATURES or feature in WINDOW_FEATURES:
        units = _BYTE_UNITS
    else:
        for word in _RATE_UNIT_WORDS + _COUNT_UNIT_WORDS:
            if unit.endswith(word):
                unit = unit[: -len(word)].strip()
                break
        units = _SCALE_SUFFIXES

    if unit not in units:
        raise FlowInputError(f"{feature}: unknown unit {match.group(2)!r} in {value!r}")
    result = number * units[unit]
    if not math.isfinite(result):
        raise FlowInputError(f"{feature}: value must be a finite number, got {value!r}")
    return result


def _check_range(feature: str, number: float, warnings: List[str]) -> float:
    if feature in WINDOW_FEATURES:
        if number < -1:
            raise FlowInputError(f"{feature}: must be >= -1 (-1 means not set), got {number}")
        return number
    if number < 0:
        raise FlowInputError(f"{feature}: must be >= 0, got {number}")
    if feature in FLAG_FEATURES and number > 1:
        warnings.append(f"{feature}: flags are 0/1 in this model; {number:g} treated as 1")
        return 1.0
    return number


# ============================================================================
# Derivation
# ============================================================================
def _derive(known: Dict[str, float], warnings: List[str]) -> Dict[str, float]:
    """Compute missing features from known ones (formulas verified against CIC-IDS2017)."""
    derived: Dict[str, float] = {}

    def has(*names: str) -> bool:
        return all(n in known or n in derived for n in names)

    def get(name: str) -> float:
        return known[name] if name in known else derived[name]

    if has("flow_duration") and get("flow_duration") > 0:
        seconds = get("flow_duration") / 1e6
        if "fwd_packets_s" not in known and has("total_fwd_packets"):
            derived["fwd_packets_s"] = get("total_fwd_packets") / seconds
        if "bwd_packets_s" not in known and has("total_bwd_packets"):
            derived["bwd_packets_s"] = get("total_bwd_packets") / seconds
        if "flow_packets_s" not in known and has("total_fwd_packets", "total_bwd_packets"):
            derived["flow_packets_s"] = (get("total_fwd_packets") + get("total_bwd_packets")) / seconds

    if "fwd_packet_length_mean" not in known and has("total_len_fwd", "total_fwd_packets") and get("total_fwd_packets") > 0:
        derived["fwd_packet_length_mean"] = get("total_len_fwd") / get("total_fwd_packets")
    if "bwd_packet_length_mean" not in known and has("total_len_bwd", "total_bwd_packets"):
        bwd = get("total_bwd_packets")
        derived["bwd_packet_length_mean"] = get("total_len_bwd") / bwd if bwd > 0 else 0.0

    if "down_up_ratio" not in known and has("total_fwd_packets", "total_bwd_packets") and get("total_fwd_packets") > 0:
        derived["down_up_ratio"] = float(math.floor(get("total_bwd_packets") / get("total_fwd_packets")))

    if "packet_length_mean" not in known and has("total_len_fwd", "total_len_bwd", "total_fwd_packets", "total_bwd_packets"):
        packets = get("total_fwd_packets") + get("total_bwd_packets")
        # CIC-IDS2017 averages over packets + 1; matches most but not all rows.
        derived["packet_length_mean"] = (get("total_len_fwd") + get("total_len_bwd")) / (packets + 1)
        warnings.append("packet_length_mean was approximated from total bytes and packets")

    return derived


def _consistency_warnings(known: Dict[str, float], warnings: List[str]) -> None:
    duration = known.get("flow_duration")
    fwd, bwd = known.get("total_fwd_packets"), known.get("total_bwd_packets")
    if duration and duration > 0 and fwd is not None and bwd is not None and "flow_packets_s" in known:
        expected = (fwd + bwd) / (duration / 1e6)
        given = known["flow_packets_s"]
        if expected > 0 and abs(given - expected) / expected > 0.5:
            warnings.append(
                f"flow_packets_s ({given:,.2f}) differs from packets / duration ({expected:,.2f}); check the inputs"
            )

    lo, mean, hi = known.get("min_packet_length"), known.get("packet_length_mean"), known.get("max_packet_length")
    if lo is not None and hi is not None and lo > hi:
        warnings.append("min_packet_length is larger than max_packet_length")
    if mean is not None and hi is not None and mean > hi:
        warnings.append("packet_length_mean is larger than max_packet_length")
    if mean is not None and lo is not None and mean < lo:
        warnings.append("packet_length_mean is smaller than min_packet_length")


# ============================================================================
# Public API
# ============================================================================
FlowInput = Union[Dict[str, Any], FlowFeatures, NormalizedFlow]


def normalize_flow(data: FlowInput, *, lenient: bool = False) -> NormalizedFlow:
    """Build a complete, validated 24-feature flow from partial input.

    Args:
        data: dict of feature -> value (numbers or unit strings), a FlowFeatures,
            or an already normalized flow (returned unchanged).
        lenient: for stored dataset rows. NaN/invalid values become "missing"
            with a warning instead of raising FlowInputError.
    """
    if isinstance(data, NormalizedFlow):
        return data
    if isinstance(data, FlowFeatures):
        raw = {**data.model_dump(exclude_none=True), **(data.model_extra or {})}
    elif isinstance(data, dict):
        raw = dict(data)
    else:
        raise FlowInputError(f"Unsupported flow input type: {type(data).__name__}")

    warnings: List[str] = []
    known: Dict[str, float] = {}

    for key, value in raw.items():
        feature = _canonical_key(key)
        if feature not in FEATURE_NAMES:
            warnings.append(f"Ignored unknown field: {key}")
            continue
        try:
            if lenient and isinstance(value, float) and not math.isfinite(value):
                raise FlowInputError(f"{feature}: missing value in source data")
            number = parse_quantity(feature, value)
            if number is None:
                continue
            known[feature] = _check_range(feature, number, warnings)
        except FlowInputError as exc:
            if not lenient:
                raise
            warnings.append(f"{exc}; treated as missing")

    _consistency_warnings(known, warnings)
    derived = _derive(known, warnings)
    available = {**known, **derived}

    missing_key = [
        group for group, names in KEY_FEATURE_GROUPS.items()
        if not any(name in available for name in names)
    ]

    medians = benign_medians()
    filled = [name for name in FEATURE_NAMES if name not in available]
    features = {name: float(available.get(name, medians[name])) for name in FEATURE_NAMES}

    known_count = len(known) + len(derived)
    if known_count >= FULL_QUALITY_MIN:
        quality = "full"
    elif known_count >= REDUCED_QUALITY_MIN:
        quality = "reduced"
    else:
        quality = "low"
    if filled:
        warnings.append(f"{len(filled)} feature(s) not provided were filled with normal-traffic medians")

    return NormalizedFlow(
        features=features,
        provided=[name for name in FEATURE_NAMES if name in known],
        derived=[name for name in FEATURE_NAMES if name in derived],
        filled=filled,
        missing_key_features=missing_key,
        needs_follow_up=bool(missing_key),
        input_quality=quality,
        warnings=warnings,
    )


def follow_up_question(flow: NormalizedFlow) -> Optional[str]:
    """Plain-language question asking for the missing key features, or None."""
    if not flow.needs_follow_up:
        return None
    missing = ", ".join(flow.missing_key_features)
    return (
        f"To analyse this flow I still need the {missing}. "
        "Could you share them (for example from your flow logs or NetFlow/Zeek records)?"
    )


@lru_cache(maxsize=1)
def _incident_table() -> pd.DataFrame:
    columns = ["incident_id", "label", "dst_port", *FEATURE_NAMES]
    logger.info(f"Loading incident features from {CLEANED_INCIDENTS_FILE}")
    return pd.read_parquet(CLEANED_INCIDENTS_FILE, columns=columns).set_index("incident_id")


def normalize_incident_id(incident_id: Union[str, int]) -> str:
    text = str(incident_id).strip().upper()
    if text.isdigit():
        return f"INC-{int(text):06d}"
    match = re.fullmatch(r"INC-?(\d+)", text)
    if not match:
        raise FlowInputError(f"Invalid incident ID {incident_id!r}; expected a form like INC-000123")
    return f"INC-{int(match.group(1)):06d}"


def get_incident_features(incident_id: Union[str, int]) -> IncidentFlow:
    """Look up the stored flow features of a historical incident."""
    key = normalize_incident_id(incident_id)
    table = _incident_table()
    if key not in table.index:
        raise KeyError(f"Incident {key} not found")
    row = table.loc[key]
    flow = normalize_flow({name: row[name] for name in FEATURE_NAMES}, lenient=True)
    dst_port = row["dst_port"]
    return IncidentFlow(
        incident_id=key,
        historical_label=str(row["label"]),
        dst_port=None if pd.isna(dst_port) else int(dst_port),
        flow=flow,
    )
