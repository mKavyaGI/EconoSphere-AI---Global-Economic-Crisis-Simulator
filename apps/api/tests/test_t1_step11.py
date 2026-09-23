import pytest
import pandas as pd
import hashlib
from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
SCRIPT_PATH = PROJECT_ROOT / "apps" / "api" / "ml" / "train_t1_model_benchmark.py"
PRED_PATH = PROJECT_ROOT / "data" / "processed" / "t1_step11_predictions.csv"
FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "t1_step11_2026_forecasts.csv"
META_PATH = PROJECT_ROOT / "models" / "phase11" / "t1_step11_model_metadata.json"

def test_raw_checksum_unchanged():
    assert RAW_PATH.exists()
    md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    assert md5 == "8ac7e0b2bf09fbe89289f82d0c7cf25e"

def test_no_target_in_features():
    with open(SCRIPT_PATH, "r") as f:
        content = f.read()
    assert "gdp_growth_next_year" not in content.split("base_features = [")[1].split("]")[0]
    assert "next_year_target_available" not in content.split("base_features = [")[1].split("]")[0]
    assert "gdp_growth_pct" not in content.split("base_features = [")[1].split("]")[0]

def test_no_random_split():
    with open(SCRIPT_PATH, "r") as f:
        content = f.read()
    assert "train_test_split" not in content

def test_inference_2025_target_nan():
    df = pd.read_csv(INPUT_PATH)
    inf_2025 = df[df["year"] == 2025]
    assert inf_2025["gdp_growth_next_year"].isna().all()

def test_imputation_fitted_on_train_only():
    with open(SCRIPT_PATH, "r") as f:
        content = f.read()
    assert "pipe.fit(X_train, y_train)" in content

def test_predictions_have_intervals():
    assert PRED_PATH.exists()
    df = pd.read_csv(PRED_PATH)
    assert "lower_bound_90" in df.columns
    assert "upper_bound_90" in df.columns
    # Check finite intervals
    assert np.isfinite(df["lower_bound_90"]).all()
    assert np.isfinite(df["upper_bound_90"]).all()
    # Check lower <= predicted <= upper
    assert (df["lower_bound_90"] <= df["predicted_gdp_growth_next_year"]).all()
    assert (df["predicted_gdp_growth_next_year"] <= df["upper_bound_90"]).all()

def test_forecasts_have_intervals():
    assert FORECAST_PATH.exists()
    df = pd.read_csv(FORECAST_PATH)
    assert "lower_bound_90" in df.columns
    assert "upper_bound_90" in df.columns
    assert np.isfinite(df["lower_bound_90"]).all()
    assert np.isfinite(df["upper_bound_90"]).all()
    assert (df["lower_bound_90"] <= df["predicted_gdp_growth"]).all()
    assert (df["predicted_gdp_growth"] <= df["upper_bound_90"]).all()

def test_uncertainty_calculated_from_val():
    # Ensure test target was not used to calculate interval
    with open(SCRIPT_PATH, "r") as f:
        content = f.read()
    # We want to see it use y_val for calibration
    assert "val_abs_residuals = np.abs(y_val - val_predictions)" in content
    assert "q90 = np.quantile(val_abs_residuals, 0.90)" in content
    assert "y_test - test_predictions" not in content

def test_metadata_has_correct_model():
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    assert meta["selected_model"] == "A. HistGradientBoosting (Locked Control)"
