"""
PySpark-based loader for CIC-IDS2017 CSV files.
Loads all 8 raw CSV files and combines them into a single Spark DataFrame.
"""
import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_RAW_DIR,
    RAW_CSV_FILES,
    LABEL_COLUMN,
    ML_FEATURE_COLUMNS,
    CONTEXT_COLUMNS,
    RAW_TO_INTERNAL_COLUMNS,
)
from data.cleaner import normalize_column_names

logger = logging.getLogger(__name__)


def align_dataframe_to_schema(df: DataFrame, required_columns: list[str] | None = None) -> DataFrame:
    """Ensure a CSV file schema matches the canonical CIC-IDS2017 field set."""
    expected_columns = []
    for raw_name, internal_name in RAW_TO_INTERNAL_COLUMNS.items():
        if internal_name not in expected_columns:
            expected_columns.append(internal_name)

    expected_columns.extend([column for column in ML_FEATURE_COLUMNS + CONTEXT_COLUMNS if column not in expected_columns])
    expected_columns = list(dict.fromkeys(expected_columns))

    missing_required = []
    for column_name in expected_columns:
        if column_name in df.columns:
            continue
        if required_columns and column_name in required_columns:
            missing_required.append(column_name)
            continue

        if column_name in {"src_ip", "dst_ip", "timestamp", "label", "protocol"}:
            df = df.withColumn(column_name, F.lit(None).cast("string"))
        else:
            df = df.withColumn(column_name, F.lit(None).cast("double"))

    if missing_required:
        raise ValueError(f"Missing required column(s) in file schema: {missing_required}")

    return df


def create_spark_session(app_name: str = "APCIRA-DataPrep") -> SparkSession:
    """
    Create and configure Spark session.

    Args:
        app_name: Name of the Spark application.

    Returns:
        Configured SparkSession.
    """
    logger.info(f"Creating Spark session: {app_name}")

    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "50") \
        .getOrCreate()

    logger.info(f"Spark version: {spark.version}")
    return spark


def load_csv_file(spark: SparkSession, csv_path: Path) -> tuple:
    """
    Load a single CSV file.

    Args:
        spark: SparkSession.
        csv_path: Path to CSV file.

    Returns:
        Tuple of (DataFrame or None, row count).
    """
    if not csv_path.exists():
        logger.warning(f"File not found: {csv_path.name}")
        return None, 0

    try:
        logger.info(f"  Loading: {csv_path.name}")
        df = spark.read.csv(
            str(csv_path),
            header=True,
            inferSchema=True,
            mode="PERMISSIVE"
        )
        logger.info(f"    Raw columns before normalization: {len(df.columns)}")
        df = normalize_column_names(df)
        df = align_dataframe_to_schema(df, required_columns=[LABEL_COLUMN, "dst_port"])
        row_count = df.count()
        logger.info(f"    {row_count:,} rows, {len(df.columns)} columns")
        return df, row_count
    except Exception as e:
        logger.error(f"    Error loading {csv_path.name}: {e}")
        return None, 0


def load_all_cic_ids_files(spark: SparkSession, raw_dir: Path = None) -> tuple:
    """
    Load all 8 CIC-IDS2017 CSV files and combine them.

    Args:
        spark: SparkSession.
        raw_dir: Directory containing CSV files. Defaults to DATA_RAW_DIR.

    Returns:
        Tuple of (combined DataFrame, total row count).
    """
    if raw_dir is None:
        raw_dir = DATA_RAW_DIR

    logger.info("=" * 80)
    logger.info("LOADING CIC-IDS2017 CSV FILES")
    logger.info("=" * 80)

    dfs = []
    total_rows = 0

    for csv_file in RAW_CSV_FILES:
        csv_path = raw_dir / csv_file
        df, row_count = load_csv_file(spark, csv_path)

        if df is not None:
            dfs.append(df)
            total_rows += row_count

    if not dfs:
        logger.error("No CSV files loaded!")
        raise ValueError("No CSV files found in DATA_RAW_DIR")

    # Align columns by canonical name; raw CSV headers vary slightly.
    logger.info(f"\nCombining {len(dfs)} files...")
    combined_df = dfs[0]
    for idx, df in enumerate(dfs[1:], start=1):
        logger.info(f"  Aligning file {idx}/{len(dfs)-1} to canonical schema before UNION")
        combined_df = combined_df.unionByName(df, allowMissingColumns=True)

    combined_df = combined_df.localCheckpoint(eager=True)
    final_count = combined_df.count()
    logger.info(f"Combined: {final_count:,} rows, {len(combined_df.columns)} columns")

    return combined_df, final_count


def verify_required_columns(df: DataFrame, required_cols: list = None) -> bool:
    """
    Verify that required columns exist in DataFrame.

    Args:
        df: Spark DataFrame.
        required_cols: List of required column names. Defaults to [LABEL_COLUMN].

    Returns:
        True if all required columns present, raises otherwise.
    """
    if required_cols is None:
        required_cols = [LABEL_COLUMN]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.error(f"Available columns: {df.columns}")
        raise ValueError(f"Missing columns: {missing_cols}")

    logger.info(f"✓ All required columns present: {required_cols}")
    return True
