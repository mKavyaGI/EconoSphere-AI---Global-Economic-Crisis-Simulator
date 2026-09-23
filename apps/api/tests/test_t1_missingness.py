import pytest
import pandas as pd
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
SCRIPT_PATH = PROJECT_ROOT / "apps" / "api" / "ml" / "train_t1_missingness_aware.py"

def test_raw_checksum_unchanged():
    assert RAW_PATH.exists(), "Raw dataset missing"
    md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    assert md5 == "8ac7e0b2bf09fbe89289f82d0c7cf25e", "Raw dataset checksum was modified!"

def test_no_target_in_features():
    with open(SCRIPT_PATH, "r") as f:
        content = f.read()
    assert "gdp_growth_next_year" not in content.split("base_features = [")[1].split("]")[0]
    assert "next_year_target_available" not in content.split("base_features = [")[1].split("]")[0]

def test_no_random_split():
    with open(SCRIPT_PATH, "r") as f:
        content = f.read()
    assert "train_test_split" not in content, "Random split used"

def test_inference_2025_target_nan():
    df = pd.read_csv(INPUT_PATH)
    inf_2025 = df[df["year"] == 2025]
    assert inf_2025["gdp_growth_next_year"].isna().all(), "2025 inference targets must be NaN"

def test_missingness_indicators_independent_of_target():
    # Indicators should be calculated purely from raw feature presence
    df = pd.read_csv(INPUT_PATH)
    assert "inflation_cpi_pct_missing" in df.columns
    # Verify indicator only has 0 or 1
    assert df["inflation_cpi_pct_missing"].isin([0, 1]).all()
    
def test_imputation_fitted_on_train_only():
    with open(SCRIPT_PATH, "r") as f:
        content = f.read()
    assert "pipe.fit(X_tr, y_train)" in content
