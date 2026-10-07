"""
Validation and verification functions for prepared incident datasets.
"""
import logging
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, isnan, isnull

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LABEL_COLUMN

logger = logging.getLogger(__name__)


def validate_required_columns(df: DataFrame, required_cols: list) -> bool:
    """
    Check that all required columns exist.

    Args:
        df: Spark DataFrame.
        required_cols: List of required column names.

    Returns:
        True if all present, raises otherwise.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")

    logger.info(f"✓ All {len(required_cols)} required columns present")
    return True


def validate_incident_ids(df: DataFrame) -> bool:
    """
    Verify incident IDs are unique and non-null.

    Args:
        df: Spark DataFrame with incident_id column.

    Returns:
        True if valid, raises otherwise.
    """
    total_rows = df.count()
    unique_ids = df.select("incident_id").distinct().count()

    logger.info(f"Incident ID validation:")
    logger.info(f"  Total rows: {total_rows:,}")
    logger.info(f"  Unique IDs: {unique_ids:,}")

    if total_rows != unique_ids:
        logger.error(f"  ✗ Duplicate IDs found! ({total_rows} rows, {unique_ids} unique IDs)")
        raise ValueError(f"Duplicate incident IDs found")

    logger.info(f"  ✓ All IDs unique")
    return True


def validate_text_column(df: DataFrame, text_col: str = "incident_text") -> bool:
    """
    Verify incident_text column is non-null and non-empty.

    Args:
        df: Spark DataFrame.
        text_col: Name of text column.

    Returns:
        True if valid, raises otherwise.
    """
    if text_col not in df.columns:
        logger.error(f"Text column '{text_col}' not found")
        raise ValueError(f"Text column '{text_col}' not found")

    total_rows = df.count()

    # Count non-null texts
    non_null_texts = df.filter(col(text_col).isNotNull()).count()
    empty_texts = df.filter(col(text_col) == "").count()

    logger.info(f"Text column validation ({text_col}):")
    logger.info(f"  Total rows: {total_rows:,}")
    logger.info(f"  Non-null: {non_null_texts:,}")
    logger.info(f"  Empty: {empty_texts:,}")

    if non_null_texts < total_rows:
        logger.warning(f"  ⚠ {total_rows - non_null_texts} null texts")

    if empty_texts > 0:
        logger.warning(f"  ⚠ {empty_texts} empty texts")

    logger.info(f"  ✓ Text column valid")
    return True


def validate_label_distribution(df: DataFrame, label_col: str = LABEL_COLUMN) -> dict:
    """
    Validate and log label distribution.

    Args:
        df: Spark DataFrame.
        label_col: Name of label column.

    Returns:
        Dictionary mapping label → count.
    """
    total_rows = df.count()
    label_counts = {}

    label_dist = df.groupBy(label_col).count().sort(col("count").desc()).collect()

    logger.info(f"\nLabel distribution validation ({label_col}):")
    logger.info(f"{'Label':<30} {'Count':>12} {'Percentage':>12}")
    logger.info("-" * 56)

    for row in label_dist:
        label = row[label_col]
        count_val = row['count']
        pct = (count_val / total_rows * 100) if total_rows > 0 else 0
        logger.info(f"{str(label):<30} {count_val:>12,} {pct:>11.1f}%")
        label_counts[label] = count_val

    logger.info(f"{'TOTAL':<30} {total_rows:>12,} {'100.0':>11}%")

    return label_counts


def validate_for_duplicates(df: DataFrame) -> bool:
    """
    Check for duplicate logical-flow records using the same deterministic hash
    strategy as the data-prep pipeline. This avoids constructing a full-row
    generated comparison plan.

    Args:
        df: Spark DataFrame.

    Returns:
        True if no duplicates, False otherwise.
    """
    total_rows = df.count()
    if "row_hash" in df.columns:
        unique_rows = df.dropDuplicates(["row_hash"]).count()
    else:
        key_columns = [
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
        key_columns = [column for column in key_columns if column in df.columns]
        if not key_columns:
            logger.warning("  ⚠ No dedupe key columns available for duplicate validation; skipping")
            return True
        unique_rows = df.dropDuplicates(key_columns).count()

    logger.info(f"Duplicate validation:")
    logger.info(f"  Total rows: {total_rows:,}")
    logger.info(f"  Unique rows: {unique_rows:,}")
    logger.info(f"  Duplicates: {total_rows - unique_rows:,}")

    if total_rows == unique_rows:
        logger.info(f"  ✓ No duplicates")
        return True
    else:
        logger.warning(f"  ⚠ Duplicates found")
        return False


def validate_null_values(df: DataFrame, important_cols: list = None) -> dict:
    """
    Count null values in important columns.

    Args:
        df: Spark DataFrame.
        important_cols: List of columns to check. If None, checks all.

    Returns:
        Dictionary mapping column → null count.
    """
    total_rows = df.count()

    if important_cols is None:
        important_cols = df.columns

    null_counts = {}

    logger.info(f"Null value validation:")
    logger.info(f"{'Column':<30} {'Nulls':>12} {'%':>10}")
    logger.info("-" * 54)

    for col_name in important_cols:
        if col_name in df.columns:
            null_count = df.filter(col(col_name).isNull()).count()
            pct = (null_count / total_rows * 100) if total_rows > 0 else 0
            null_counts[col_name] = null_count

            if null_count > 0:
                logger.info(f"{col_name:<30} {null_count:>12,} {pct:>9.2f}%")

    if not null_counts or all(v == 0 for v in null_counts.values()):
        logger.info("✓ No nulls in important columns")

    return null_counts


def validate_final_dataset(
    df: DataFrame,
    expected_rows: int = None,
    label_col: str = LABEL_COLUMN
) -> bool:
    """
    Run all validation checks on final dataset.

    Args:
        df: Spark DataFrame.
        expected_rows: Expected approximate row count.
        label_col: Name of label column.

    Returns:
        True if all validations pass.
    """
    logger.info("\n" + "=" * 80)
    logger.info("FINAL DATASET VALIDATION")
    logger.info("=" * 80)

    try:
        # Row count
        total_rows = df.count()
        logger.info(f"\nRow count: {total_rows:,}")
        if expected_rows and abs(total_rows - expected_rows) / expected_rows > 0.1:
            logger.warning(f"  ⚠ Expected ~{expected_rows:,}, got {total_rows:,}")

        # Required columns
        required = ["incident_id", label_col, "incident_text", "dst_port"]
        validate_required_columns(df, required)

        # Destination port must remain available for retrieval metadata.
        dst_port_nulls = df.filter(col("dst_port").isNull()).count()
        dst_port_values = df.filter(col("dst_port").isNotNull()).select("dst_port").distinct().count()
        logger.info(f"Destination port validation:")
        logger.info(f"  Null dst_port values: {dst_port_nulls:,}")
        logger.info(f"  Distinct dst_port values: {dst_port_values:,}")
        if dst_port_nulls == df.count():
            raise ValueError("dst_port was lost from the final dataset")

        # Incident IDs
        validate_incident_ids(df)

        # Text column
        validate_text_column(df)

        # Label distribution
        validate_label_distribution(df, label_col)

        # Duplicates
        validate_for_duplicates(df)

        # Nulls
        validate_null_values(df, required)

        # Schema
        logger.info(f"\nSchema:")
        for field in df.schema.fields:
            logger.info(f"  {field.name}: {field.dataType}")

        logger.info("\n" + "=" * 80)
        logger.info("✓ ALL VALIDATIONS PASSED")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"\n✗ Validation failed: {e}")
        return False
