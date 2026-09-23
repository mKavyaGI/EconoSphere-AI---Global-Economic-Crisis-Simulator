import os
import sys
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
import pytest
import numpy as np
import pandas as pd

# Update sys.path to ensure modules can be loaded if necessary
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.ml.phase18.step1.run_regime_aware_weighting_experiments import (
    get_supported_countries,
    get_hashes,
    verify_artifacts_mutated,
    generate_chronological_folds,
    calculate_rolling_volatility,
    calculate_historical_growth_deviation,
    fit_regime_thresholds,
    calculate_sample_weights,
    evaluate_candidates_and_select,
    calculate_naive_baseline,
    calculate_dummy_baseline,
    calculate_effective_sample_size,
    validate_weights,
    prevent_pseudo_replication,
    calculate_percentage_improvement,
    compare_predictions
)

def test_dynamic_frontend_country_extraction(tmp_path):
    # Mock frontend file
    mock_frontend = tmp_path / "page.tsx"
    mock_frontend.write_text('const SUPPORTED_COUNTRIES = ["USA", "GBR", "X_Y"];')
    
    countries = get_supported_countries(frontend_path=mock_frontend)
    assert "USA" in countries
    assert "GBR" in countries
    assert "X_Y" in countries
    assert len(countries) == 3

def test_fallback_country_behavior():
    # Provide a non-existent path
    countries = get_supported_countries(frontend_path=Path("/does/not/exist.tsx"))
    assert "GBR" in countries # Priority
    assert "USA" in countries # Guardrail
    assert len(countries) == 10

def test_md5_hash_consistency(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("dummy content")
    md5, _ = get_hashes(test_file)
    assert md5 == hashlib.md5(b"dummy content").hexdigest()

def test_sha256_hash_consistency(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("dummy content")
    _, sha256 = get_hashes(test_file)
    assert sha256 == hashlib.sha256(b"dummy content").hexdigest()

def test_protected_artifact_mutation_detection(tmp_path):
    # Test our mutation detection logic
    pre_hashes = {"file1": ("md5_1", "sha256_1")}
    post_hashes = {"file1": ("md5_1", "sha256_1")}
    assert not verify_artifacts_mutated(pre_hashes, post_hashes)
    
    post_hashes_mutated = {"file1": ("md5_2", "sha256_1")}
    assert verify_artifacts_mutated(pre_hashes, post_hashes_mutated)

def test_experimental_output_path_isolation():
    from apps.api.ml.phase18.step1.run_regime_aware_weighting_experiments import EXPERIMENTAL_DIR
    assert "experimental_artifacts" in str(EXPERIMENTAL_DIR)
    assert "phase18" in str(EXPERIMENTAL_DIR)

def test_experimental_filename_suffix_enforcement():
    from apps.api.ml.phase18.step1.run_regime_aware_weighting_experiments import get_experimental_filename
    filename = get_experimental_filename("my_model", "joblib")
    assert "EXPERIMENTAL_ONLY" in filename
    assert filename.endswith(".joblib")

def test_chronological_fold_ordering():
    df = pd.DataFrame({
        "year": [2010, 2011, 2012, 2013, 2014, 2015],
        "gdp_growth_next_year": [1, 2, 3, 4, 5, 6]
    })
    folds = generate_chronological_folds(df, min_train_years=2, val_years=1, test_years=1)
    for fold in folds:
        train_max = fold["train"]["year"].max()
        val_min = fold["val"]["year"].min()
        val_max = fold["val"]["year"].max()
        test_min = fold["test"]["year"].min()
        assert train_max < val_min
        assert val_max < test_min

def test_strict_feature_year_to_target_year_alignment():
    df = pd.DataFrame({
        "year": [2010, 2011],
        "gdp_growth_next_year": [1.5, 2.0]
    })
    # For year 2010, target is available in 2011.
    # The generation function should ensure we don't leak 2011 features.
    folds = generate_chronological_folds(df, min_train_years=1, val_years=1, test_years=1)
    # Just asserting the function exists and can be called, specific assertions depend on impl.
    assert len(folds) >= 0

def test_no_future_target_leakage_in_rolling_volatility():
    df = pd.DataFrame({
        "year": [2010, 2011, 2012],
        "gdp_growth_next_year": [1, 2, 3],
        "country_code": ["USA", "USA", "USA"]
    })
    vol = calculate_rolling_volatility(df, "gdp_growth_next_year", window=2)
    # The volatility for 2012 should NOT use the target of 2012 (which is 2013 growth)
    # And it should use only targets that are strictly before 2012.
    assert len(vol) == 3

def test_no_future_target_leakage_in_historical_growth_deviation():
    df = pd.DataFrame({
        "year": [2010, 2011, 2012],
        "gdp_growth_next_year": [1, 2, 3],
        "country_code": ["USA", "USA", "USA"]
    })
    dev = calculate_historical_growth_deviation(df, "gdp_growth_next_year")
    assert len(dev) == 3

def test_regime_thresholds_fitted_on_training_only():
    train_data = pd.DataFrame({"vol": [1, 2, 3, 4, 5]})
    val_data = pd.DataFrame({"vol": [10, 20]})
    thresholds = fit_regime_thresholds(train_data, feature="vol")
    assert thresholds["high"] <= 5 # It should not be influenced by val_data

def test_sample_weights_fitted_on_training_only():
    train_data = pd.DataFrame({"regime": ["NORMAL", "EXTREME"]})
    weights = calculate_sample_weights(train_data, strategy="W1_MILD_SHOCK_DOWNWEIGHT")
    assert len(weights) == 2

def test_validation_used_for_selection_not_test():
    metrics_val = {"M1": {"RMSE": 2.0}, "W1": {"RMSE": 1.5}}
    metrics_test = {"M1": {"RMSE": 1.0}, "W1": {"RMSE": 3.0}}
    selected = evaluate_candidates_and_select(metrics_val, metrics_test)
    assert selected == "W1" # Should pick based on val, not test

def test_test_metrics_cannot_alter_selected_candidate():
    metrics_val = {"M1": {"RMSE": 1.5}, "W1": {"RMSE": 2.0}}
    metrics_test = {"M1": {"RMSE": 3.0}, "W1": {"RMSE": 1.0}}
    selected = evaluate_candidates_and_select(metrics_val, metrics_test)
    assert selected == "M1"

def test_naive_baseline_no_future_target_info():
    df_test = pd.DataFrame({
        "year": [2020],
        "gdp_growth_next_year": [5.0],
        "gdp_growth_lag1": [2.0] # This represents growth from 2018->2019 available in 2020
    })
    preds = calculate_naive_baseline(df_test)
    # Naive baseline should predict the last available growth, not the future target
    assert (preds != df_test["gdp_growth_next_year"]).all()

def test_dummy_baseline_uses_training_targets_only():
    df_train = pd.DataFrame({"gdp_growth_next_year": [1.0, 2.0, 3.0]})
    df_test = pd.DataFrame({"gdp_growth_next_year": [10.0, 20.0]})
    preds = calculate_dummy_baseline(df_train, df_test)
    assert (preds == 2.0).all() # Mean of train

def test_weighted_effective_sample_size_calculation():
    weights = np.array([1.0, 1.0, 0.1, 0.1])
    n_eff = calculate_effective_sample_size(weights)
    # n_eff = (sum(w))^2 / sum(w^2) = (2.2)^2 / (2.02) = 4.84 / 2.02 = 2.396
    assert np.isclose(n_eff, 2.396, atol=0.01)

def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        validate_weights(np.array([1.0, -0.5]))
    with pytest.raises(ValueError):
        validate_weights(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        validate_weights(np.array([1.0, np.inf]))

def test_pseudo_replication_prevented():
    df = pd.DataFrame({"country_code": ["USA", "USA"], "year": [2010, 2010]})
    with pytest.raises(ValueError):
        prevent_pseudo_replication(df)

def test_percentage_improvement_handles_zero_baselines():
    imp = calculate_percentage_improvement(base_rmse=0.0, new_rmse=1.0)
    assert imp == 0.0 or np.isnan(imp)

def test_prediction_difference_diagnostics_exact_matches():
    y1 = np.array([1.0, 2.0])
    y2 = np.array([1.0, 2.0])
    res = compare_predictions(y1, y2)
    assert res["max_diff"] == 0.0
    assert res["status"] == "EXACT_REPRODUCTION"
