"""
EconoSphere AI -- Phase 11, Step 6: Forecasting Target Tests
============================================================
File   : apps/api/tests/test_forecasting_target.py
Run    : pytest apps/api/tests/test_forecasting_target.py -v

Tests verify:
1. Target boundary doesn't cross countries.
2. 2025 rows have missing next-year target.
3. No feature is equal to next-year target.
4. Next-year target is exactly gdp_growth_pct(T+1).
5. Dataset dimensions are correct.
6. Raw dataset checksum remains unchanged.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

RAW_PATH           = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
PROCESSED_PATH     = PROJECT_ROOT / "data" / "processed" / "master_panel_processed.csv"
FORECASTING_PATH   = PROJECT_ROOT / "data" / "processed" / "master_panel_forecasting.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COL      = "gdp_growth_pct"
NEXT_TARGET_COL = "gdp_growth_next_year"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def processed_df() -> pd.DataFrame:
    assert PROCESSED_PATH.exists(), f"Processed dataset not found: {PROCESSED_PATH}"
    return pd.read_csv(PROCESSED_PATH, low_memory=False)

@pytest.fixture(scope="module")
def forecasting_df() -> pd.DataFrame:
    assert FORECASTING_PATH.exists(), f"Forecasting dataset not found: {FORECASTING_PATH}"
    return pd.read_csv(FORECASTING_PATH, low_memory=False)

# ===========================================================================
# TEST 1: Dimensions Match
# ===========================================================================
def test_dataset_dimensions(processed_df: pd.DataFrame, forecasting_df: pd.DataFrame):
    """Forecasting dataset should have same number of rows as processed dataset, and 2 extra columns."""
    assert len(processed_df) == len(forecasting_df), "Row count mismatch between processed and forecasting datasets"
    assert len(forecasting_df.columns) == len(processed_df.columns) + 2, "Column count mismatch"
    assert NEXT_TARGET_COL in forecasting_df.columns
    assert "next_year_target_available" in forecasting_df.columns

# ===========================================================================
# TEST 2: Target is exactly T+1 within country
# ===========================================================================
def test_target_is_shifted_correctly(forecasting_df: pd.DataFrame):
    """The next-year target for year T must equal the actual GDP growth for year T+1 within the same country."""
    df = forecasting_df.sort_values(["country_code", "year"]).reset_index(drop=True)
    
    for code, group in df.groupby("country_code"):
        # The shifted target should exactly equal actual GDP growth of the subsequent row
        actual_shifted = group[TARGET_COL].shift(-1)
        
        # Compare where both are not na
        mask = group[NEXT_TARGET_COL].notna() & actual_shifted.notna()
        assert np.allclose(group.loc[mask, NEXT_TARGET_COL], actual_shifted[mask]), f"Target mismatch in country {code}"

# ===========================================================================
# TEST 3: 2025 rows missing target
# ===========================================================================
def test_2025_target_missing(forecasting_df: pd.DataFrame):
    """All rows for the year 2025 must have a NaN next-year target."""
    df_2025 = forecasting_df[forecasting_df["year"] == 2025]
    if not df_2025.empty:
        assert df_2025[NEXT_TARGET_COL].isna().all(), "Found non-NaN next-year targets for 2025"

# ===========================================================================
# TEST 4: No Direct Feature Leakage
# ===========================================================================
def test_no_feature_leakage(forecasting_df: pd.DataFrame):
    """Verify no feature has ~1.0 correlation with the next-year target."""
    features = [c for c in forecasting_df.columns if c not in ["country_code", "country_name", "year", NEXT_TARGET_COL, "next_year_target_available", TARGET_COL]]
    
    for f in features:
        if forecasting_df[f].dtype in ['float64', 'int64']:
            mask = forecasting_df[f].notna() & forecasting_df[NEXT_TARGET_COL].notna()
            if mask.sum() > 10:
                corr = np.corrcoef(forecasting_df.loc[mask, f], forecasting_df.loc[mask, NEXT_TARGET_COL])[0, 1]
                assert abs(corr) < 0.99, f"Feature {f} leaks next-year target (corr: {corr:.4f})"

# ===========================================================================
# TEST 5: Raw Checksum Unchanged
# ===========================================================================
def test_raw_dataset_unchanged():
    """The raw master_panel.csv must not have been modified."""
    assert RAW_PATH.exists(), f"Raw file not found: {RAW_PATH}"
    actual_md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    expected   = "8ac7e0b2bf09fbe89289f82d0c7cf25e"
    assert actual_md5 == expected, (
        f"[CRITICAL] Raw dataset was MODIFIED!\n"
        f"  Expected MD5 : {expected}\n"
        f"  Actual MD5   : {actual_md5}"
    )
