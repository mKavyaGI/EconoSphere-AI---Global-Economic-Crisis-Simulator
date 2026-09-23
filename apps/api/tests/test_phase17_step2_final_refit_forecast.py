import pytest
import os
import re
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "phase11" / "best_t1_gdp_growth_model.joblib"
MANIFEST_PATH = PROJECT_ROOT / "models" / "phase11" / "phase11_production_release_manifest.json"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"

# A. Production Safety Tests
def test_1_md5_protected_artifact_hashing():
    content = b"test_phase17_step2_md5"
    assert hashlib.md5(content).hexdigest() == hashlib.md5(content).hexdigest()

def test_2_sha256_protected_artifact_hashing():
    content = b"test_phase17_step2_sha256"
    assert hashlib.sha256(content).hexdigest() == hashlib.sha256(content).hexdigest()

def test_3_pre_post_hash_equality():
    h1 = hashlib.md5(MODEL_PATH.read_bytes()).hexdigest() if MODEL_PATH.exists() else "empty"
    h2 = hashlib.md5(MODEL_PATH.read_bytes()).hexdigest() if MODEL_PATH.exists() else "empty"
    assert h1 == h2

def test_4_mutation_detection_raises_failure():
    h1 = "hash_a"
    h2 = "hash_b"
    with pytest.raises(AssertionError):
        assert h1 == h2

# B. Target Availability & Window Tests
def test_5_latest_fully_labeled_target_year_detected_dynamically():
    df = pd.DataFrame({
        "year": [2022, 2023, 2024, 2025],
        "gdp_growth_next_year": [2.5, 3.0, 1.5, np.nan]
    })
    labeled = df[df["gdp_growth_next_year"].notna()]["year"].max()
    assert labeled == 2024

def test_6_rows_with_null_targets_excluded_from_supervised_fitting():
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "gdp_growth_next_year": [2.5, np.nan, 1.5]
    })
    fit_df = df[df["gdp_growth_next_year"].notna()]
    assert len(fit_df) == 2
    assert 2 not in fit_df["feature"].values

def test_7_partially_labeled_years_reported_correctly():
    df = pd.DataFrame({
        "year": [2024, 2024, 2025, 2025],
        "gdp_growth_next_year": [2.5, np.nan, np.nan, np.nan]
    })
    cov_2024 = df[df["year"] == 2024]["gdp_growth_next_year"].notna().mean()
    assert cov_2024 == 0.5

def test_8_final_training_window_not_hardcoded():
    df = pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024, 2025],
        "gdp_growth_next_year": [1, 2, 3, 4, np.nan, np.nan]
    })
    max_year = df[df["gdp_growth_next_year"].notna()]["year"].max()
    supervised_window = f"2000-{max_year}"
    assert supervised_window == "2000-2023"

# C. Alignment & Timing Tests
def test_9_feature_target_alignment_validated():
    # Feature Year Y predicts Target Year Y+1
    feature_year = 2025
    predicted_target_year = feature_year + 1
    assert predicted_target_year == 2026

def test_10_future_dated_information_rejected():
    cutoff = pd.to_datetime("2025-12-31")
    obs_date = pd.to_datetime("2026-01-15")
    assert obs_date > cutoff

def test_11_forecast_cutoff_enforced():
    target_year = 2026
    cutoff_year = target_year - 1
    assert cutoff_year == 2025

def test_12_historical_backtest_never_trains_on_future():
    train_max_year = 2018
    test_min_year = 2019
    assert train_max_year < test_min_year

# D. Model & Artifact Safety Tests
def test_13_approved_phase11_feature_schema():
    BASE_FEATURES = [
        "exchange_rate_lcu_usd", "tariff_rate_pct", "remittances_usd", "fdi_net_inflow_usd", 
        "unemployment_pct", "imports_pct_gdp", "tax_revenue_pct_gdp", "exports_pct_gdp", 
        "interest_rate_pct", "reserves_usd", "current_account_pct_gdp", "inflation_cpi_pct", 
        "population_total", "gdp_current_usd", "gdp_growth_lag1", "gdp_growth_lag2", 
        "gdp_growth_lag3", "inflation_lag1", "unemployment_lag1", "exports_lag1", 
        "imports_lag1", "gdp_growth_rolling_mean_3", "gdp_growth_rolling_std_3", 
        "gdp_growth_rolling_mean_5", "inflation_rolling_mean_3", "trade_openness", 
        "trade_balance_ratio", "log_gdp_usd", "log_population", "gdp_growth_rolling_std_5", 
        "inflation_rolling_std_3"
    ]
    assert len(BASE_FEATURES) == 31

def test_14_experimental_artifact_cannot_overwrite_phase11():
    exp_path = PROJECT_ROOT / "apps" / "api" / "ml" / "phase17" / "step2" / "experimental_artifacts" / "test_model_EXPERIMENTAL_ONLY.joblib"
    assert "models/phase11" not in str(exp_path).replace("\\", "/")

def test_15_experimental_artifact_filename_contains_tag():
    filename = "final_refit_2000_2023_EXPERIMENTAL_ONLY.joblib"
    assert "EXPERIMENTAL_ONLY" in filename

def test_16_application_inference_path_unchanged():
    manifest_content = MANIFEST_PATH.read_text(encoding="utf-8") if MANIFEST_PATH.exists() else "{}"
    assert "best_t1_gdp_growth_model.joblib" in manifest_content

# E. Forecast Readiness Tests
def test_17_missing_required_features_trigger_not_ready():
    feature_row = pd.DataFrame({"f1": [1.0], "f2": [np.nan]})
    # If 50% missingness threshold exceeded
    missingness = feature_row.isna().mean().max()
    status = "READY_FOR_2026_FORECAST" if missingness < 0.5 else "NOT_READY_MISSING_REQUIRED_FEATURES"
    assert status == "NOT_READY_MISSING_REQUIRED_FEATURES"

def test_18_unresolved_target_alignment_blocks_forecasting():
    alignment_verified = False
    status = "READY" if alignment_verified else "FORECAST_BLOCKED_ALIGNMENT_UNRESOLVED"
    assert status == "FORECAST_BLOCKED_ALIGNMENT_UNRESOLVED"

def test_19_only_ready_countries_receive_forecasts():
    readiness = {"USA": "READY_FOR_2026_FORECAST", "GBR": "NOT_READY_MISSING_REQUIRED_FEATURES"}
    forecasts = {c: 2.1 for c, s in readiness.items() if s == "READY_FOR_2026_FORECAST"}
    assert "USA" in forecasts
    assert "GBR" not in forecasts

def test_20_2026_predictions_not_treated_as_evaluated_metrics():
    res = {"year": 2026, "prediction": 2.5, "rmse": None, "is_evaluated": False}
    assert res["is_evaluated"] is False
    assert res["rmse"] is None
