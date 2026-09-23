import os
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
ADVANCED_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_advanced.csv"

def test_raw_dataset_untouched():
    assert RAW_PATH.exists()
    actual_md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    assert actual_md5 == "8ac7e0b2bf09fbe89289f82d0c7cf25e", "Raw dataset checksum altered!"

def test_no_target_leakage_in_features():
    df = pd.read_csv(ADVANCED_PATH)
    features = [c for c in df.columns if c not in ["country_code", "country_name", "year", "gdp_growth_next_year", "next_year_target_available", "gdp_target_available"]]
    
    # Target shouldn't be exactly equal to any feature
    target = df["gdp_growth_next_year"]
    for f in features:
        if pd.api.types.is_numeric_dtype(df[f]):
            valid = df[f].notna() & target.notna()
            if valid.sum() > 10:
                corr = np.corrcoef(df.loc[valid, f], target[valid])[0, 1]
                assert corr < 0.99, f"Feature {f} leaks target (correlation > 0.99)"

def test_inference_integrity():
    df = pd.read_csv(ADVANCED_PATH)
    df_2025 = df[df["year"] == 2025]
    
    assert len(df_2025) > 0, "No inference rows found for 2025"
    assert df_2025["gdp_growth_next_year"].isna().all(), "Fabricated 2026 targets detected!"
    assert (df_2025["next_year_target_available"] == 0).all(), "next_year_target_available flag incorrectly set for 2025"

def test_country_aggregate_leakage():
    # country_historical_avg_gdp_growth shouldn't change for a country across years
    # and shouldn't be the target itself
    df = pd.read_csv(ADVANCED_PATH)
    std_by_country = df.groupby("country_code")["country_historical_avg_gdp_growth"].std()
    assert std_by_country.fillna(0).max() < 1e-6, "Country average is not static within countries!"
    
    # Check that for val/test period, it is not simply the mean of val/test
    # This was computed on train only
    train_mask = (df["year"] >= 2000) & (df["year"] <= 2018)
    val_mask = (df["year"] >= 2019) & (df["year"] <= 2022)
    
    # Mean of gdp_growth_pct for a country in train should match the feature
    for code, group in df.groupby("country_code"):
        train_rows = group[train_mask]
        if not train_rows.empty:
            mean_train = train_rows["gdp_growth_pct"].mean()
            feature_val = group["country_historical_avg_gdp_growth"].iloc[0]
            if not pd.isna(mean_train):
                assert np.isclose(mean_train, feature_val, equal_nan=True), f"Country {code} feature {feature_val} != train mean {mean_train}"

def test_global_aggregate_leakage():
    # global_gdp_growth_lag1 is the mean of gdp_growth_lag1 for a given year
    df = pd.read_csv(ADVANCED_PATH)
    for yr, group in df.groupby("year"):
        mean_lag1 = group["gdp_growth_lag1"].mean()
        feature_val = group["global_gdp_growth_lag1"].iloc[0]
        if not pd.isna(mean_lag1):
            assert np.isclose(mean_lag1, feature_val, equal_nan=True)
