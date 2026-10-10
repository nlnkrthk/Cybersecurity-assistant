"""
Convert incident flow rows into natural-language text for semantic embedding.
Designed to work with both Spark DataFrames and Pandas Series.
"""
import logging
import math
from datetime import date, datetime
from numbers import Real
from typing import Union
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, concat_ws, lit
import pandas as pd

logger = logging.getLogger(__name__)

_TEXT_FEATURES = [
    ("flow_duration", "Flow duration", "microseconds"),
    ("total_fwd_packets", "Forward packets", None),
    ("total_bwd_packets", "Backward packets", None),
    ("total_len_fwd", "Forward bytes", None),
    ("total_len_bwd", "Backward bytes", None),
    ("flow_packets_s", "Flow packets per second", None),
    ("fwd_packets_s", "Forward packets per second", None),
    ("bwd_packets_s", "Backward packets per second", None),
    ("fwd_packet_length_mean", "Forward packet length mean", "bytes"),
    ("bwd_packet_length_mean", "Backward packet length mean", "bytes"),
    ("min_packet_length", "Minimum packet length", "bytes"),
    ("max_packet_length", "Maximum packet length", "bytes"),
    ("packet_length_mean", "Packet length mean", "bytes"),
    ("packet_length_std", "Packet length standard deviation", "bytes"),
    ("syn_flag_count", "SYN flags", None),
    ("ack_flag_count", "ACK flags", None),
    ("rst_flag_count", "RST flags", None),
    ("psh_flag_count", "PSH flags", None),
    ("fin_flag_count", "FIN flags", None),
    ("down_up_ratio", "Down/up ratio", None),
    ("init_win_bytes_forward", "Initial forward window size", "bytes"),
    ("init_win_bytes_backward", "Initial backward window size", "bytes"),
    ("active_mean", "Active mean", "microseconds"),
    ("idle_mean", "Idle mean", "microseconds"),
    ("fwd_syn_flags", "Forward SYN flags", None),
    ("bwd_syn_flags", "Backward SYN flags", None),
    ("fwd_ack_flags", "Forward ACK flags", None),
    ("bwd_ack_flags", "Backward ACK flags", None),
    ("fwd_rst_flags", "Forward RST flags", None),
    ("bwd_rst_flags", "Backward RST flags", None),
    ("fwd_psh_flags", "Forward PSH flags", None),
    ("bwd_psh_flags", "Backward PSH flags", None),
    ("fwd_urg_flags", "Forward URG flags", None),
    ("bwd_urg_flags", "Backward URG flags", None),
]


def _present_value(row: Union[pd.Series, dict], name: str):
    value = row.get(name)
    if value is None or pd.isna(value):
        return None
    return value


def _format_value(value) -> str:
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            if numeric_value.is_integer():
                return str(int(numeric_value))
            return f"{numeric_value:.6g}"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value).strip()


def flow_to_text(row: Union[pd.Series, dict]) -> str:
    """
    Convert a single incident row into natural-language text for embedding.

    Describes attack type, protocol, ports, packet volumes, duration, and TCP flags.

    Args:
        row: Pandas Series or dict-like object with incident data.

    Returns:
        Natural-language text description.
    """
    parts = []

    label = _present_value(row, "label")
    if label is not None:
        if str(label).strip().casefold() == "benign":
            parts.append("Traffic label: BENIGN.")
        else:
            parts.append(f"Attack: {_format_value(label)}.")

    protocol = _present_value(row, "protocol")
    if protocol is not None:
        protocol_text = _format_value(protocol)
        if isinstance(protocol, Real) and not isinstance(protocol, bool):
            protocol_number = int(protocol)
            protocol_text = {1: "ICMP", 6: "TCP", 17: "UDP"}.get(
                protocol_number, protocol_text
            )
        parts.append(f"Protocol: {protocol_text}.")

    for column, label_text in (("src_port", "Source port"), ("dst_port", "Destination port")):
        value = _present_value(row, column)
        if value is not None:
            parts.append(f"{label_text}: {_format_value(value)}.")

    for column, label_text, unit in _TEXT_FEATURES:
        value = _present_value(row, column)
        if value is None:
            continue
        suffix = f" {unit}" if unit else ""
        parts.append(f"{label_text}: {_format_value(value)}{suffix}.")

    return " ".join(parts) if parts else "No flow details available."


def build_incident_text_column(df: DataFrame) -> DataFrame:
    """
    Add incident_text column to Spark DataFrame.

    Note: This uses pandas UDF for efficient distributed processing.

    Args:
        df: Spark DataFrame.

    Returns:
        Spark DataFrame with incident_text column added.
    """
    logger.info("Building incident_text column...")

    # For Spark, we'll use a simple approach: convert to pandas, process, convert back
    # This is acceptable for ~100K records
    # For larger datasets, use pandas_udf with grouped processing

    try:
        pdf = df.toPandas()
        logger.info(f"Converted to pandas: {len(pdf):,} rows")

        pdf['incident_text'] = pdf.apply(flow_to_text, axis=1)
        logger.info(f"Generated incident_text for {len(pdf):,} incidents")

        # Convert back to Spark
        df_with_text = df.sparkSession.createDataFrame(pdf)

        return df_with_text

    except Exception as e:
        logger.error(f"Error building incident_text: {e}")
        raise


def build_text_corpus_pandas(df: pd.DataFrame) -> pd.Series:
    """
    Build text corpus for Pandas DataFrame (legacy function for compatibility).

    Args:
        df: Pandas DataFrame with incident data.

    Returns:
        Series of incident texts indexed by incident_id.
    """
    logger.info(f"Building text corpus for {len(df)} incidents (Pandas)")

    texts = df.apply(flow_to_text, axis=1)
    texts.index = df['incident_id']

    logger.info(f"Built {len(texts)} incident texts")
    return texts
