"""
Shared pytest setup: puts src/ on the import path (same as the scripts) and
provides real CIC-IDS2017 flows for tests.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from services.feature_input import FEATURE_NAMES  # noqa: E402

CLEANED_CSV = PROJECT_ROOT / "data" / "cleaned_network_data" / "cleaned_network_data.csv"


@pytest.fixture(scope="session")
def flows_df() -> pd.DataFrame:
    """Cleaned flows with duplicate feature vectors removed."""
    return pd.read_csv(CLEANED_CSV).drop_duplicates(subset=list(FEATURE_NAMES)).reset_index(drop=True)


@pytest.fixture(scope="session")
def one_per_class(flows_df) -> pd.DataFrame:
    """One real flow per traffic type (15 rows)."""
    return flows_df.groupby("label").head(1).reset_index(drop=True)


@pytest.fixture(scope="session")
def full_flow(flows_df) -> dict:
    """A complete 24-feature DDoS flow with non-zero duration and packets in both directions."""
    rows = flows_df[
        (flows_df["label"] == "DDoS")
        & (flows_df["flow_duration"] > 0)
        & (flows_df["total_bwd_packets"] > 0)
    ]
    return {name: float(rows.iloc[0][name]) for name in FEATURE_NAMES}


@pytest.fixture(scope="session")
def ml_service():
    from services.ml_service import get_ml_service

    return get_ml_service()
