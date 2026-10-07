"""
PySpark-based stratified sampling for incident dataset.
Samples data to approximately 100K records while preserving class distribution.
"""
import logging
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LABEL_COLUMN,
    SAMPLES_PER_CLASS_100K,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)


def get_label_distribution(df: DataFrame) -> dict:
    """
    Calculate label distribution in DataFrame.

    Args:
        df: Spark DataFrame.

    Returns:
        Dictionary mapping label → count.
    """
    label_counts = {}
    for row in df.groupBy(LABEL_COLUMN).count().collect():
        label_counts[row[LABEL_COLUMN]] = row['count']

    return label_counts


def log_label_distribution(df: DataFrame, title: str = "Label Distribution") -> dict:
    """
    Log label distribution to logger.

    Args:
        df: Spark DataFrame.
        title: Title for log output.

    Returns:
        Dictionary of label counts.
    """
    total_rows = df.count()
    label_counts = get_label_distribution(df)

    logger.info(f"\n{title}:")
    logger.info(f"{'Label':<30} {'Count':>12} {'Percentage':>12}")
    logger.info("-" * 56)

    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = (count / total_rows * 100) if total_rows > 0 else 0
        logger.info(f"{label:<30} {count:>12,} {pct:>11.1f}%")

    logger.info(f"{'TOTAL':<30} {total_rows:>12,} {'100.0':>11}%")

    return label_counts


def calculate_sampling_fractions(label_counts: dict, target_samples: dict) -> dict:
    """
    Calculate sampling fractions for each label.

    Args:
        label_counts: Current counts per label.
        target_samples: Target counts per label (from config).

    Returns:
        Dictionary mapping label → sampling fraction.
    """
    fractions = {}

    logger.info("Calculating sampling fractions:")
    logger.info(f"{'Label':<30} {'Current':>12} {'Target':>12} {'Fraction':>10}")
    logger.info("-" * 66)

    for label, target_count in target_samples.items():
        if label not in label_counts:
            logger.warning(f"  {label}: NOT FOUND in dataset")
            continue

        current_count = label_counts[label]

        if target_count is None:
            # Keep all samples for rare classes
            fraction = 1.0
            logger.info(f"{label:<30} {current_count:>12,} {'(all)':>12} {fraction:>10.4f}")
        else:
            # Calculate fraction needed
            if current_count <= target_count:
                # Already under target, keep all
                fraction = 1.0
                actual_target = current_count
            else:
                # Need to sample down
                fraction = target_count / current_count
                actual_target = target_count

            logger.info(f"{label:<30} {current_count:>12,} {actual_target:>12,} {fraction:>10.4f}")

        fractions[label] = fraction

    return fractions


def stratified_sample(
    df: DataFrame,
    label_column: str = LABEL_COLUMN,
    target_samples: dict = None,
    seed: int = RANDOM_SEED
) -> tuple:
    """
    Perform stratified sampling using Spark's sampleBy method.

    Args:
        df: Spark DataFrame with incident data.
        label_column: Name of label column.
        target_samples: Dict mapping label → target count. Defaults to SAMPLES_PER_CLASS_100K.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sampled DataFrame, total sampled rows, original distribution, final distribution).
    """
    if target_samples is None:
        target_samples = SAMPLES_PER_CLASS_100K

    logger.info("\n" + "=" * 80)
    logger.info("STRATIFIED SAMPLING")
    logger.info("=" * 80)

    # Get current distribution
    logger.info("\nBEFORE SAMPLING:")
    before_dist = log_label_distribution(df)

    # Calculate fractions
    fractions = calculate_sampling_fractions(before_dist, target_samples)

    if not fractions:
        logger.error("No sampling fractions calculated!")
        raise ValueError("No labels matched between data and target samples")

    # Perform stratified sampling
    logger.info("\nSampling...")
    df_sampled = df.sampleBy(label_column, fractions, seed=seed)

    # Verify result
    sampled_rows = df_sampled.count()
    logger.info(f"Sampled: {sampled_rows:,} rows")

    logger.info("\nAFTER SAMPLING:")
    after_dist = log_label_distribution(df_sampled, "Final Distribution")

    return df_sampled, sampled_rows, before_dist, after_dist


def verify_sample_distribution(
    before_dist: dict,
    after_dist: dict,
    target_samples: dict = None
) -> bool:
    """
    Verify that sample distribution is reasonable.

    Args:
        before_dist: Distribution before sampling.
        after_dist: Distribution after sampling.
        target_samples: Expected targets.

    Returns:
        True if reasonable, logs warnings otherwise.
    """
    if target_samples is None:
        target_samples = SAMPLES_PER_CLASS_100K

    logger.info("\n" + "=" * 80)
    logger.info("VERIFYING SAMPLE DISTRIBUTION")
    logger.info("=" * 80)

    all_good = True

    for label in after_dist.keys():
        actual = after_dist[label]
        target = target_samples.get(label)

        if target is None:
            target = before_dist.get(label, 0)

        if target > 0:
            deviation = abs(actual - target) / target * 100
            if deviation > 5:
                logger.warning(f"  {label}: deviation {deviation:.1f}% (actual={actual}, target={target})")
                all_good = False
            else:
                logger.info(f"  ✓ {label}: {actual:,} (target {target:,})")

    if all_good:
        logger.info("\n✓ Sample distribution verified")
    else:
        logger.warning("\n⚠ Some labels deviated from target")

    return all_good
