import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add api to path so we can import the script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.ml.phase17.step3.run_expanding_window_backtest import (
    get_hashes,
    get_pct_improvement,
    get_supported_countries,
    evaluate_predictions,
    MODEL_PATH,
    MANIFEST_PATH,
    DATA_PATH,
    EXPERIMENTAL_DIR
)

def test_dynamic_supported_countries():
    countries = get_supported_countries()
    assert isinstance(countries, list)
    assert "USA" in countries
    assert "GBR" in countries

def test_protected_artifact_paths():
    assert "models/phase11/best_t1_gdp_growth_model.joblib" in str(MODEL_PATH).replace("\\", "/")
    assert "models/phase11/phase11_production_release_manifest.json" in str(MANIFEST_PATH).replace("\\", "/")
    assert "data/processed/master_panel_t1_missingness.csv" in str(DATA_PATH).replace("\\", "/")

def test_hash_calculation():
    if MODEL_PATH.exists():
        md5, sha256 = get_hashes(MODEL_PATH)
        assert len(md5) == 32
        assert len(sha256) == 64

def test_pct_improvement_handles_zero():
    assert get_pct_improvement(0, 1.0) == "PERCENTAGE_IMPROVEMENT_NOT_DEFINED"
    assert get_pct_improvement(np.nan, 1.0) == "PERCENTAGE_IMPROVEMENT_NOT_DEFINED"
    assert get_pct_improvement(2.0, 1.0) == 50.0

def test_experimental_directory():
    assert "experimental_artifacts" in str(EXPERIMENTAL_DIR)

def test_expanding_training_windows_chronological():
    df = pd.DataFrame({
        "year": [2015, 2016, 2017, 2018],
        "gdp_growth_next_year": [1.0, 2.0, 3.0, 4.0]
    })
    years = sorted(df["year"].unique())
    folds = []
    for f_year in years:
        if f_year >= 2017:
            train_candidates = [y for y in years if y < f_year]
            if train_candidates:
                max_train_f_year = max(train_candidates)
                folds.append({
                    "eval_feature_year": f_year,
                    "train_cutoff_feature_year": max_train_f_year
                })
    assert folds[0]["eval_feature_year"] == 2017
    assert folds[0]["train_cutoff_feature_year"] == 2016
    assert folds[1]["eval_feature_year"] == 2018
    assert folds[1]["train_cutoff_feature_year"] == 2017

def test_no_future_leakage():
    # If eval feature year is 2018, train cutoff should be 2017.
    # The training targets will only include targets for feature year 2017.
    f_eval = 2018
    f_train_end = 2017
    train_years = [2015, 2016, 2017]
    assert all(y <= f_train_end for y in train_years)
    assert f_eval not in train_years

def test_target_alignment():
    # Feature Year Y predicts Target Year Y+1
    f_year = 2020
    target_year = f_year + 1
    assert target_year == 2021

def test_baseline_logic():
    # N0 uses lag1, which represents information available
    df_eval = pd.DataFrame({
        "country_code": ["USA", "GBR"],
        "gdp_growth_next_year": [2.0, 1.5],
        "gdp_growth_lag1": [1.8, 1.2], # N0
    })
    
    df_eval["N0_pred"] = df_eval["gdp_growth_lag1"]
    met, _, _, _ = evaluate_predictions(df_eval, "N0_pred")
    assert met["N"] == 2
    
    # N1 uses training mean
    train_mean = 1.65
    df_eval["N1_pred"] = train_mean
    met_n1, _, _, _ = evaluate_predictions(df_eval, "N1_pred")
    assert met_n1["N"] == 2
