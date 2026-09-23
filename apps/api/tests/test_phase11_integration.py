"""
EconoSphere AI -- Phase 11 Pre-Training Integration Tests
=========================================================
File: apps/api/tests/test_phase11_integration.py
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

RAW_PATH           = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
PROCESSED_PATH     = PROJECT_ROOT / "data" / "processed" / "master_panel_processed.csv"
FORECASTING_PATH   = PROJECT_ROOT / "data" / "processed" / "master_panel_forecasting.csv"

@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(FORECASTING_PATH, low_memory=False)

def test_raw_integrity():
    actual_md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    assert actual_md5 == "8ac7e0b2bf09fbe89289f82d0c7cf25e", "Raw data changed!"

def test_country_year_uniqueness(df: pd.DataFrame):
    assert df.duplicated(["country_code", "year"]).sum() == 0, "Duplicate country-year pairs found"

def test_t_plus_1_target_correctness(df: pd.DataFrame):
    for code, group in df.groupby("country_code"):
        shifted = group["gdp_growth_pct"].shift(-1)
        mask = group["gdp_growth_next_year"].notna() & shifted.notna()
        assert np.allclose(group.loc[mask, "gdp_growth_next_year"], shifted[mask]), f"Target mismatch for {code}"

def test_2025_target_missingness(df: pd.DataFrame):
    y2025 = df[df["year"] == 2025]
    if not y2025.empty:
        assert y2025["gdp_growth_next_year"].isna().all(), "2025 target must be NaN"
        assert (y2025["next_year_target_available"] == 0).all(), "next_year_target_available must be 0 for 2025"

def test_lag_correctness(df: pd.DataFrame):
    # Check that lag1 of T is actual of T-1
    for code, group in df.groupby("country_code"):
        shifted = group["gdp_growth_pct"].shift(1)
        mask = group["gdp_growth_lag1"].notna() & shifted.notna()
        assert np.allclose(group.loc[mask, "gdp_growth_lag1"], shifted[mask]), f"Lag1 mismatch for {code}"

def test_rolling_feature_correctness(df: pd.DataFrame):
    for code, group in df.groupby("country_code"):
        # rolling_mean_3 for T should be mean of T-1, T-2, T-3
        s_shifted = group["gdp_growth_pct"].shift(1)
        expected_rolling = s_shifted.rolling(window=3, min_periods=2).mean()
        mask = group["gdp_growth_rolling_mean_3"].notna() & expected_rolling.notna()
        assert np.allclose(group.loc[mask, "gdp_growth_rolling_mean_3"], expected_rolling[mask]), f"Rolling mean 3 mismatch for {code}"

def test_temporal_split_correctness(df: pd.DataFrame):
    # Verify split logic ranges
    train_years = range(2000, 2019)
    val_years = range(2019, 2023)
    test_years = [2023, 2024]
    
    assert set(df[df["year"].isin(train_years)]["year"]) == set(train_years)
    assert set(df[df["year"].isin(val_years)]["year"]) == set(val_years)
    assert set(df[df["year"].isin(test_years)]["year"]) == set(test_years)
