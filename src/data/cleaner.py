"""
PySpark-based data cleaning for CIC-IDS2017 dataset.
Handles: column normalization, numeric casting, null/infinity handling, deduplication, label standardization.
"""
import logging
import re
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, concat_ws, lit, regexp_replace, sha2, trim, when

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_TO_INTERNAL_COLUMNS,
    LABEL_STANDARDIZATION,
    LABEL_COLUMN,
    ML_FEATURE_COLUMNS,
    CONTEXT_COLUMNS,
)

logger = logging.getLogger(__name__)


def normalize_column_name(raw_name: str) -> str:
    """Normalizes a CIC-IDS2017 header to a stable canonical identifier."""
    if raw_name is None:
        return ""

    normalized = raw_name.strip()
    normalized = re.sub(r"[\s_\-]+", " ", normalized)
    normalized = re.sub(r"[^0-9A-Za-z ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.lower()


def normalize_column_names(df: DataFrame) -> DataFrame:
    """
    Normalize raw column names: strip whitespace and rename to internal schema.

    Args:
        df: Spark DataFrame with raw column names.

    Returns:
        Spark DataFrame with normalized column names.
    """
    logger.info("Normalizing column names")

    rename_map = {}
    for old_name in list(df.columns):
        canonical_key = normalize_column_name(old_name)
        if canonical_key:
            for raw_key, internal_name in RAW_TO_INTERNAL_COLUMNS.items():
                if normalize_column_name(raw_key) == canonical_key:
                    rename_map[old_name] = internal_name
                    break
            else:
                rename_map[old_name] = old_name.strip()

    for old_name, new_name in rename_map.items():
        if old_name != new_name:
            df = df.withColumnRenamed(old_name, new_name)

    logger.info(f"Renamed {len(rename_map)} columns to the internal schema")
    return df


def cast_numeric_columns(df: DataFrame, numeric_cols: list) -> DataFrame:
    """
    Cast specified columns to double (floating-point) type.

    Args:
        df: Spark DataFrame.
        numeric_cols: List of column names to cast.

    Returns:
        Spark DataFrame with numeric columns cast.
    """
    logger.info(f"Casting {len(numeric_cols)} columns to double")

    for col_name in numeric_cols:
        if col_name in df.columns:
            try:
                df = df.withColumn(col_name, col(col_name).cast("double"))
            except Exception as e:
                logger.warning(f"  Could not cast {col_name}: {e}")

    return df


def handle_invalid_numeric_values(df: DataFrame, numeric_cols: list) -> DataFrame:
    """
    Replace infinity values with null in numeric columns.
    Log count of affected values per column.

    Args:
        df: Spark DataFrame.
        numeric_cols: List of numeric column names.

    Returns:
        Spark DataFrame with infinity/NaN handled.
    """
    logger.info("Handling invalid numeric values (inf, NaN)")

    for col_name in numeric_cols:
        if col_name in df.columns:
            # Replace inf with null
            df = df.withColumn(
                col_name,
                when(
                    (col(col_name) == float('inf')) | (col(col_name) == float('-inf')),
                    None
                ).otherwise(col(col_name))
            )

            logger.debug(f"  {col_name}: replaced inf → null")

    return df


def standardize_labels(df: DataFrame) -> DataFrame:
    """
    Standardize attack labels using LABEL_STANDARDIZATION mapping.

    Args:
        df: Spark DataFrame.

    Returns:
        Spark DataFrame with standardized labels.
    """
    logger.info("Standardizing attack labels")

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' not found")

    # Raw Web Attack labels use a non-ASCII dash ("Web Attack \x96 XSS"), which
    # Spark decodes as U+FFFD; without this they never match the mapping and the
    # classes are silently dropped during sampling.
    df = df.withColumn(LABEL_COLUMN, trim(col(LABEL_COLUMN)))
    df = df.withColumn(
        LABEL_COLUMN,
        regexp_replace(col(LABEL_COLUMN), r"^Web Attack[^A-Za-z0-9]+", "Web Attack - "),
    )

    # Apply label standardization mapping
    for raw_label, std_label in LABEL_STANDARDIZATION.items():
        df = df.withColumn(
            LABEL_COLUMN,
            when(col(LABEL_COLUMN) == raw_label, std_label).otherwise(col(LABEL_COLUMN))
        )

    return df


def remove_duplicates(df: DataFrame) -> tuple:
    """
    Remove duplicate logical flow records without constructing a full-row
    generated comparison plan. The hash key is built from the canonical flow
    identity fields, which are deterministic, stable, and independent from any
    later synthetic identifiers.

    Args:
        df: Spark DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, count of removed rows).
    """
    logger.info("Deduplicating rows using a deterministic flow hash")

    dedup_columns = [
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "flow_duration",
        "total_fwd_packets",
        "total_bwd_packets",
        "total_len_fwd",
        "total_len_bwd",
        LABEL_COLUMN,
    ]
    dedup_columns = [column for column in dedup_columns if column in df.columns]

    if not dedup_columns:
        raise ValueError("No deduplication key columns were available after normalization")

    before_count = df.count()
    hash_parts = []
    for column_name in dedup_columns:
        normalized = when(col(column_name).isNull(), lit("__NULL__")).otherwise(
            col(column_name).cast("string")
        )
        hash_parts.append(normalized)

    df_with_hash = df.withColumn("row_hash", sha2(concat_ws("|", *hash_parts), 256))
    df_dedup = df_with_hash.dropDuplicates(["row_hash"]).drop("row_hash")
    after_count = df_dedup.count()
    removed = before_count - after_count

    logger.info(f"Deduplication: {before_count:,} → {after_count:,} (removed {removed:,})")
    return df_dedup, removed


def drop_null_rows(df: DataFrame, required_cols: list) -> DataFrame:
    """
    Drop rows missing any of the required columns.

    Args:
        df: Spark DataFrame.
        required_cols: Columns that must be present for retrieval.

    Returns:
        Spark DataFrame with null rows removed.
    """
    logger.info("Dropping rows with null values in required columns")

    before_count = df.count()
    df_clean = df.dropna(how="any", subset=required_cols)
    after_count = df_clean.count()

    logger.info(f"After null removal: {before_count:,} → {after_count:,} rows")

    return df_clean


def clean_and_prepare(df: DataFrame) -> DataFrame:
    """
    Main entry point: apply all cleaning steps in sequence.

    Args:
        df: Raw Spark DataFrame from CSV load.

    Returns:
        Cleaned Spark DataFrame ready for sampling.
    """
    logger.info("=" * 80)
    logger.info("STARTING DATA CLEANING PIPELINE")
    logger.info("=" * 80)

    initial_rows = df.count()
    logger.info(f"Initial rows: {initial_rows:,}")
    logger.info(f"Initial columns: {len(df.columns)}")

    # Step 1: Normalize columns
    logger.info("\n--- Step 1: Normalizing columns ---")
    df = normalize_column_names(df)

    # Step 2: Cast numeric columns
    logger.info("\n--- Step 2: Casting numeric columns ---")
    numeric_cols = ML_FEATURE_COLUMNS + CONTEXT_COLUMNS
    numeric_cols = [c for c in numeric_cols if c not in [LABEL_COLUMN, "src_ip", "dst_ip", "timestamp"]]
    df = cast_numeric_columns(df, numeric_cols)

    # Step 3: Handle invalid numeric values
    logger.info("\n--- Step 3: Handling invalid numeric values ---")
    df = handle_invalid_numeric_values(df, numeric_cols)

    # Step 4: Standardize labels
    logger.info("\n--- Step 4: Standardizing labels ---")
    df = standardize_labels(df)

    # Step 5: Remove duplicates
    logger.info("\n--- Step 5: Removing duplicates ---")
    df, removed = remove_duplicates(df)

    # Step 6: Drop null rows
    logger.info("\n--- Step 6: Dropping null values ---")
    df = drop_null_rows(df, [LABEL_COLUMN, "dst_port"])

    final_rows = df.count()
    logger.info(f"\n--- Cleaning Summary ---")
    logger.info(f"Initial rows: {initial_rows:,}")
    logger.info(f"Final rows: {final_rows:,}")
    logger.info(f"Rows removed: {initial_rows - final_rows:,}")

    return df
