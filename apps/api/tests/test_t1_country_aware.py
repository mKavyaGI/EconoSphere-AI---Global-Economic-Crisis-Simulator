import pytest
import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_advanced.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "phase11"
SCRIPT_PATH = PROJECT_ROOT / "apps" / "api" / "ml" / "train_t1_country_aware.py"

def test_raw_checksum_unchanged():
    assert RAW_PATH.exists(), "Raw dataset missing"
    md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    assert md5 == "8ac7e0b2bf09fbe89289f82d0c7cf25e", "Raw dataset checksum was modified!"

def test_t1_advanced_target_not_a_predictor():
    with open(SCRIPT_PATH, "r") as f:
        content = f.read()
    assert "gdp_growth_next_year" not in content.split("base_features = [")[1].split("]")[0], "Target in features"

def test_no_random_split():
    with open(SCRIPT_PATH, "r") as f:
        content = f.read()
    assert "train_test_split" not in content, "Random split used"

def test_inference_2025_target_nan():
    df = pd.read_csv(INPUT_PATH)
    inf_2025 = df[df["year"] == 2025]
    assert inf_2025["gdp_growth_next_year"].isna().all(), "2025 inference targets must be NaN"
    
def test_historical_features_train_only():
    with open(SCRIPT_PATH, "r") as f:
        content = f.read()
    # verify that train_mask is used for country_stats
    assert 'train_mask = (df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1])' in content
    assert 'train_df = df[train_mask]' in content
    assert 'country_stats = train_df.groupby("country_code")' in content

def test_one_hot_encoder_fit_on_train():
    with open(SCRIPT_PATH, "r") as f:
        content = f.read()
    # Check that pipe.fit(X_train) is called, meaning encoder is fit only on train
    assert "pipe.fit(X_train, y_train)" in content
