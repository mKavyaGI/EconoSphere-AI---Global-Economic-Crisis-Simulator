import pytest
import os
from pathlib import Path
from unittest.mock import patch
import json

import pandas as pd
import numpy as np

# We'll import the script to test its functions
import apps.api.ml.audit_production_model_accuracy as audit_script

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def test_protected_artifacts_exist():
    assert audit_script.MODEL_PATH.exists()
    assert audit_script.MANIFEST_PATH.exists()
    assert audit_script.DATA_PATH.exists()

def test_no_mutating_calls_in_script():
    content = Path(audit_script.__file__).read_text(encoding="utf-8")
    assert "joblib.dump" not in content
    # The script shouldn't call fit on the model
    assert ".fit(" not in content
    assert "master_panel.csv" in content
    # Make sure we don't write to master_panel.csv
    import re
    assert not re.search(r'to_csv\([^)]*master_panel\.csv', content)

def test_metric_calculations():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    
    metrics = audit_script.calculate_metrics(y_true, y_pred)
    assert np.isclose(metrics["MAE"], 0.1333333)
    assert np.isclose(metrics["RMSE"], 0.14142135)
    assert "MAPE" not in metrics # Ensure MAPE is avoided due to zero handling as per requirements

def test_residual_calculations():
    y_true = np.array([1.0, -2.0, 3.0])
    y_pred = np.array([1.5, -2.5, 3.0])
    
    res = audit_script.analyze_residuals(y_true, y_pred)
    assert np.isclose(res["Mean Residual"], 0.0)
    assert np.isclose(res["Mean Residual"], 0.0)
    assert np.isclose(res["Max Positive Residual"], 0.5)
    assert np.isclose(res["Min Negative Residual"], -0.5)
    assert np.isclose(res["Max Absolute Error"], 0.5)

def test_api_vs_offline_consistency():
    # Mocking model prediction
    class MockModel:
        def predict(self, X):
            return np.array([2.6995696601100905])
            
    api_payload = audit_script.VALID_PAYLOAD.copy()
    api_payload.pop("country")
    api_payload.pop("year")
    
    X_df = pd.DataFrame([api_payload])
    
    # Check consistency logic
    model = MockModel()
    pred = model.predict(X_df)[0]
    
    assert audit_script.verify_api_consistency(pred, 2.6995696601100905) == True
    assert audit_script.verify_api_consistency(pred, 2.7) == False

def test_audit_fails_on_hash_mismatch():
    original = audit_script.AUTHORITATIVE_HASHES.copy()
    audit_script.AUTHORITATIVE_HASHES["model"] = "bad_hash"
    
    # We patch sys.exit so it doesn't actually exit the test suite
    with patch('sys.exit') as mock_exit:
        audit_script.verify_hashes(audit_script.AUTHORITATIVE_HASHES)
        mock_exit.assert_called_once()
        
    audit_script.AUTHORITATIVE_HASHES = original
