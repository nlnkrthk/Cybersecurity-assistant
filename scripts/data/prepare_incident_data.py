#!/usr/bin/env python3
"""
Unified Data Preparation Pipeline for CIC-IDS2017

Orchestrates modular data processing steps:
1. Load all 8 CIC-IDS2017 CSV files (PySpark)
2. Clean and normalize (PySpark)
3. Stratified sampling to ~100K (PySpark)
4. Generate incident IDs (Pandas for ~100K records only)
5. Build incident text (Pandas)
6. Save to Parquet
7. Verify output

Reusable functions and classes are in src/data/:
- loader.py: CSV loading
- cleaner.py: Data cleaning
- sampler.py: Stratified sampling
- text_builder.py: Text generation
- validator.py: Verification
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
project_root = Path(__file__).resolve().parents[2]
log_dir = project_root / "data" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"data_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(project_root / "src"))

# Import configuration
try:
    from config import (
        DATA_PROCESSED_DIR,
        LABEL_COLUMN,
        ML_FEATURE_COLUMNS,
        CONTEXT_COLUMNS,
        RETRIEVAL_TEXT_COLUMNS,
        RANDOM_SEED,
        CLEANED_INCIDENTS_FILE,
    )
except ImportError as e:
    logger.error(f"Failed to import config: {e}")
    sys.exit(1)

# Import reusable modules
try:
    from data.loader import create_spark_session, load_all_cic_ids_files, verify_required_columns
    from data.cleaner import clean_and_prepare, normalize_column_names
    from data.sampler import stratified_sample, verify_sample_distribution
    from data.text_builder import flow_to_text
    from data.validator import validate_final_dataset
except ImportError as e:
    logger.error(f"Failed to import data modules: {e}")
    logger.error("Ensure all modules are in src/data/")
    sys.exit(1)

# Import pandas for final processing
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Install with: pip install pandas numpy")
    sys.exit(1)

# ============================================================================
# Main Pipeline
# ============================================================================

logger.info("=" * 80)
logger.info("CIC-IDS2017 DATA PREPARATION PIPELINE (Modular)")
logger.info("=" * 80)
logger.info(f"Output: {CLEANED_INCIDENTS_FILE}")
logger.info(f"Log: {log_file}")

# Step 1: Initialize Spark
logger.info("\n" + "=" * 80)
logger.info("STEP 1: Initializing Spark Session")
logger.info("=" * 80)

try:
    spark = create_spark_session("APCIRA-DataPrep-100K")
    logger.info(f"✓ Spark session created")
except Exception as e:
    logger.error(f"✗ Failed to create Spark session: {e}")
    sys.exit(1)

# Step 2: Load CSV files
logger.info("\n" + "=" * 80)
logger.info("STEP 2: Loading 8 CIC-IDS2017 CSV Files")
logger.info("=" * 80)

try:
    df_raw, total_rows = load_all_cic_ids_files(spark)
    logger.info(f"✓ Loaded {total_rows:,} rows from {len(df_raw.columns)} columns")
except Exception as e:
    logger.error(f"✗ Failed to load CSVs: {e}")
    spark.stop()
    sys.exit(1)

# Step 3: Verify required columns
logger.info("\n--- Verifying required columns ---")
try:
    df_raw = normalize_column_names(df_raw)
    verify_required_columns(df_raw, [LABEL_COLUMN, "dst_port"])
except Exception as e:
    logger.error(f"✗ Column verification failed: {e}")
    spark.stop()
    sys.exit(1)

# Step 4: Clean and prepare data
logger.info("\n" + "=" * 80)
logger.info("STEP 3: Cleaning and Preparing Data")
logger.info("=" * 80)

try:
    df_cleaned = clean_and_prepare(df_raw)
    logger.info(f"✓ Data cleaned and prepared")
except Exception as e:
    logger.error(f"✗ Cleaning failed: {e}")
    spark.stop()
    sys.exit(1)

# Step 5: Stratified sampling
logger.info("\n" + "=" * 80)
logger.info("STEP 4: Stratified Sampling to ~100K")
logger.info("=" * 80)

try:
    df_sampled, sampled_rows, before_dist, after_dist = stratified_sample(
        df_cleaned,
        label_column=LABEL_COLUMN,
        seed=RANDOM_SEED
    )
    verify_sample_distribution(before_dist, after_dist)
    logger.info(f"✓ Sampled to {sampled_rows:,} rows")
except Exception as e:
    logger.error(f"✗ Sampling failed: {e}")
    spark.stop()
    sys.exit(1)

# Step 6: Convert to Pandas and add incident IDs + text
logger.info("\n" + "=" * 80)
logger.info("STEP 5: Converting to Pandas (for ID and Text Generation)")
logger.info("=" * 80)

try:
    # Select required columns for text generation
    requested_cols = list(dict.fromkeys(RETRIEVAL_TEXT_COLUMNS + CONTEXT_COLUMNS))
    available_cols = [c for c in requested_cols if c in df_sampled.columns]
    missing_required = {"label", "dst_port"} - set(available_cols)
    if missing_required:
        raise ValueError(
            "Required retrieval columns are missing after CSV normalization: "
            f"{', '.join(sorted(missing_required))}"
        )
    df_pd = df_sampled.select(available_cols).toPandas()
    logger.info(f"✓ Converted to Pandas: {len(df_pd):,} rows")

    # Shuffle and assign incident IDs
    df_pd = df_pd.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    df_pd['incident_id'] = [f"INC-{i+1:06d}" for i in range(len(df_pd))]
    logger.info(f"✓ Assigned {len(df_pd):,} incident IDs")

    # Generate incident_text
    logger.info("  Generating incident_text descriptions...")
    df_pd['incident_text'] = df_pd.apply(flow_to_text, axis=1)
    non_empty = df_pd['incident_text'].str.len().gt(0).sum()
    logger.info(f"✓ Generated {non_empty:,} incident texts")

except Exception as e:
    logger.error(f"✗ Pandas processing failed: {e}")
    spark.stop()
    sys.exit(1)

# Step 7: Save to Parquet
logger.info("\n" + "=" * 80)
logger.info("STEP 6: Saving to Parquet")
logger.info("=" * 80)

try:
    # Create parent directory if needed
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save using pandas
    df_pd.to_parquet(str(CLEANED_INCIDENTS_FILE), index=False)
    logger.info(f"✓ Saved to {CLEANED_INCIDENTS_FILE}")

    if CLEANED_INCIDENTS_FILE.exists():
        file_size_mb = CLEANED_INCIDENTS_FILE.stat().st_size / 1024 / 1024
        logger.info(f"  File size: {file_size_mb:.2f} MB")

except Exception as e:
    logger.error(f"✗ Save failed: {e}")
    spark.stop()
    sys.exit(1)

# Step 8: Verify output
logger.info("\n" + "=" * 80)
logger.info("STEP 7: Verifying Output")
logger.info("=" * 80)

try:
    df_verify = spark.read.parquet(str(CLEANED_INCIDENTS_FILE))
    logger.info(f"✓ Parquet file readable")

    # Run validation
    validate_final_dataset(
        df_verify,
        expected_rows=len(df_pd),
        label_col=LABEL_COLUMN
    )

except Exception as e:
    logger.error(f"✗ Verification failed: {e}")
    spark.stop()
    sys.exit(1)

# Finish
logger.info("\n" + "=" * 80)
logger.info("[SUCCESS] DATA PREPARATION COMPLETED")
logger.info("=" * 80)
logger.info(f"\nOutput Parquet: {CLEANED_INCIDENTS_FILE}")
logger.info(f"Rows: {len(df_pd):,}")
logger.info(f"Columns: {len(df_pd.columns)}")
logger.info(f"Log: {log_file}")
logger.info("\nModular Architecture:")
logger.info("  src/data/loader.py      → CSV loading")
logger.info("  src/data/cleaner.py     → Data cleaning")
logger.info("  src/data/sampler.py     → Stratified sampling")
logger.info("  src/data/text_builder.py → Text generation")
logger.info("  src/data/validator.py   → Verification")
logger.info("\nNext steps:")
logger.info("  1. Build search indexes: python scripts/indexing/build_search_indexes.py")
logger.info("  2. Run interactive search: python scripts/search/interactive_hybrid_search.py")

spark.stop()
logger.info("\n[OK] Spark session closed")
