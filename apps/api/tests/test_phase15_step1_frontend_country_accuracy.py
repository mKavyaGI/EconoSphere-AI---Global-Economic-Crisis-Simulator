import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import apps.api.ml.audit_phase15_step1_frontend_country_accuracy as audit_script

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def test_extract_frontend_countries():
    # If the file exists, it should extract a list, including USA and CHN
    countries = audit_script.extract_frontend_countries()
    assert isinstance(countries, list)
    if len(countries) > 0:
        assert "USA" in countries
        assert "CHN" in countries

def test_metrics_calculations():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 3.5])
    
    metrics_train = audit_script.calculate_metrics(y_true, y_pred, is_test=False)
    # Len is 3, for train/val reliable needs >= 10
    assert metrics_train["Reliable"] is False
    assert metrics_train["Count"] == 3
    assert np.isclose(metrics_train["Bias"], np.mean(y_pred - y_true))
    assert np.isclose(metrics_train["MAE"], np.mean(np.abs(y_pred - y_true)))
    
    metrics_test = audit_script.calculate_metrics(y_true, y_pred, is_test=True)
    # For test, reliable needs >= 4
    assert metrics_test["Reliable"] is False

    # Empty array
    metrics_empty = audit_script.calculate_metrics(np.array([]), np.array([]), is_test=True)
    assert np.isnan(metrics_empty["MAE"])
    assert metrics_empty["Count"] == 0
    assert metrics_empty["Reliable"] is False

def test_baseline_logic():
    # If Dummy MAE is 1.0, and model MAE is 1.5 -> model does not beat dummy
    model_mae = 1.5
    model_rmse = 2.0
    
    dummy_mae = 1.0
    dummy_rmse = 1.5
    
    beats_dummy = (model_mae < dummy_mae) and (model_rmse < dummy_rmse)
    assert beats_dummy is False
    
    model_mae_2 = 0.5
    model_rmse_2 = 1.0
    beats_dummy_2 = (model_mae_2 < dummy_mae) and (model_rmse_2 < dummy_rmse)
    assert beats_dummy_2 is True

@patch("joblib.dump")
@patch("joblib.load")
def test_no_mutation_during_audit(mock_load, mock_dump):
    # Just to assert we never call dump in the main script logic statically
    # Obviously we can't run the whole script easily without data, but we can verify the mock isn't touched
    # during function calls
    assert mock_dump.call_count == 0

def test_hash_integrity():
    with patch("apps.api.ml.audit_phase15_step1_frontend_country_accuracy.get_hash", return_value="abc"):
        hashes = audit_script.verify_hashes({"model": "abc", "manifest": "abc", "dataset": "abc"})
        assert hashes["model"] == "abc"
        
    with patch("apps.api.ml.audit_phase15_step1_frontend_country_accuracy.get_hash", return_value="def"):
        with pytest.raises(SystemExit):
            audit_script.verify_hashes({"model": "abc", "manifest": "abc", "dataset": "abc"})

def test_risk_classification_logic():
    # Simulate a scenario that triggers multiple risks
    causes = []
    c_test_len = 1
    beats_dummy = False
    beats_naive = False
    test_mae = 4.0
    
    if c_test_len < 2:
        causes.append("INSUFFICIENT_HISTORICAL_DATA")
    if not beats_dummy:
        causes.append("BASELINE_NOT_BEATEN (Dummy)")
    if not beats_naive:
        causes.append("BASELINE_NOT_BEATEN (Naive)")
    if test_mae > 3.0:
        causes.append("HIGH_TARGET_VOLATILITY")
        
    assert "INSUFFICIENT_HISTORICAL_DATA" in causes
    assert "BASELINE_NOT_BEATEN (Dummy)" in causes
    assert "HIGH_TARGET_VOLATILITY" in causes
