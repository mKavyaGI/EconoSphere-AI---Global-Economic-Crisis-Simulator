"""
EconoSphere AI -- Phase 11, Step 7: Baseline T+1 Training Leakage Tests
"""
import json
import hashlib
from pathlib import Path
import pandas as pd

THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

METADATA_PATH    = PROJECT_ROOT / "models" / "phase11" / "t1_model_metadata.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "t1_baseline_predictions.csv"
FORECASTS_PATH   = PROJECT_ROOT / "data" / "processed" / "t1_2026_forecasts.csv"
RAW_PATH         = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"

def test_leakage_target_exclusion():
    assert METADATA_PATH.exists(), "Metadata not found"
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
        
    features = meta.get("feature_columns", [])
    assert "gdp_growth_next_year" not in features, "Target included in predictors!"
    assert "next_year_target_available" not in features, "Target availability flag included in predictors!"
    assert "gdp_growth_pct" not in features, "Current year GDP growth included in predictors (potential leakage/confusion)!"
    
def test_leakage_raw_dataset_untouched():
    assert RAW_PATH.exists(), "Raw dataset not found"
    raw_md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    assert raw_md5 == "8ac7e0b2bf09fbe89289f82d0c7cf25e", "Raw dataset was modified!"

def test_forecasts_validity():
    assert FORECASTS_PATH.exists(), "2026 Forecasts not found"
    fc = pd.read_csv(FORECASTS_PATH)
    assert not fc.empty, "Forecasts file is empty"
    assert (fc["feature_year"] == 2025).all(), "Inference used non-2025 feature year"
    assert (fc["target_year"] == 2026).all(), "Inference target year is not 2026"
    assert "actual_gdp_growth" not in fc.columns, "Actual GDP growth should not be present for 2026"

def test_predictions_validity():
    assert PREDICTIONS_PATH.exists(), "Test predictions not found"
    preds = pd.read_csv(PREDICTIONS_PATH)
    assert not preds.empty, "Predictions file is empty"
    
    # Feature years should be 2023 or 2024
    assert preds["feature_year"].isin([2023, 2024]).all(), "Test predictions contain non-test feature years"
    
    # Target years should be 2024 or 2025
    assert preds["target_year"].isin([2024, 2025]).all(), "Test predictions contain non-test target years"
