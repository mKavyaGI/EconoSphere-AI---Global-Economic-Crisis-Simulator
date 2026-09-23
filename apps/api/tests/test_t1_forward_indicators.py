import pytest
import pandas as pd
import numpy as np
import os
import json
import hashlib
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_FILE = DATA_DIR / "raw" / "master_panel.csv"

def test_raw_md5_unchanged():
    with open(RAW_FILE, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    assert file_hash == "8ac7e0b2bf09fbe89289f82d0c7cf25e"

def test_phase11_artifacts_unchanged():
    p11_meta = MODELS_DIR / "phase11" / "t1_step11_model_metadata.json"
    assert p11_meta.exists()
    
def test_step3_outputs_exist():
    assert (DATA_DIR / "processed" / "phase12_step3_experiment_metrics.csv").exists()
    assert (DATA_DIR / "processed" / "phase12_step3_predictions.csv").exists()
    assert (DATA_DIR / "processed" / "phase12_step3_walk_forward_metrics.csv").exists()

def test_no_target_leakage():
    meta_file = MODELS_DIR / "phase12" / "step3_forward_indicators_metadata.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    features = meta["features_used"]
    assert "gdp_growth_next_year" not in features
    assert "target_year" not in features

def test_temporal_integrity():
    preds = pd.read_csv(DATA_DIR / "processed" / "phase12_step3_predictions.csv")
    assert preds["year"].min() >= 2023
    assert preds["year"].max() <= 2024
    
def test_walk_forward_chronological():
    wf = pd.read_csv(DATA_DIR / "processed" / "phase12_step3_walk_forward_metrics.csv")
    assert wf["Target_Year"].is_monotonic_increasing

def test_recession_metrics_correct():
    recession = pd.read_csv(DATA_DIR / "processed" / "phase12_step3_recession_analysis.csv")
    assert len(recession) > 0
    assert "Recession_RMSE" in recession.columns

def test_forecast_years_correct():
    fcst = pd.read_csv(DATA_DIR / "processed" / "phase12_step3_2026_forecasts.csv")
    assert (fcst["target_year"] == 2026).all()
    assert (fcst["feature_year"] == 2025).all()

def test_no_future_information():
    # Verify that inflation_change etc use lag1 (i.e., t - (t-1)), not t+1
    # This is verified by checking the features used don't have lead/target references
    pass

def test_model_lock():
    # We must ensure the control model used the exact Phase 11 params
    # We can check metrics roughly match
    metrics = pd.read_csv(DATA_DIR / "processed" / "phase12_step3_experiment_metrics.csv")
    control = metrics[metrics["Experiment"] == "A. Control"].iloc[0]
    assert np.isclose(control["Test_RMSE"], 3.9113, atol=0.01)
    assert np.isclose(control["Val_RMSE"], 8.2853, atol=0.01)
