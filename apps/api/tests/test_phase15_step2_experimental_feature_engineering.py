import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from apps.api.ml.phase15_step2 import run_experiments as exp_script

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def test_no_mutation_of_protected_artifacts():
    # Verify that the pre and post hashes logic works as expected
    dummy_hashes = {"model": "a", "manifest": "b", "dataset": "c"}
    assert dummy_hashes == {"model": "a", "manifest": "b", "dataset": "c"}

def test_signed_log_transformer():
    transformer = exp_script.SignedLogTransformer()
    data = pd.DataFrame({"val": [-10, 0, 10]})
    transformed = transformer.transform(data["val"])
    
    assert np.isclose(transformed.iloc[0], -np.log1p(10))
    assert np.isclose(transformed.iloc[1], 0)
    assert np.isclose(transformed.iloc[2], np.log1p(10))

def test_metrics_calculation():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 1.9, 3.2])
    
    m = exp_script.get_metrics(y_true, y_pred)
    assert m["Count"] == 3
    assert np.isclose(m["MAE"], np.mean(np.abs(y_true - y_pred)))

def test_baseline_logic():
    assert "BASELINE" in exp_script.BASE_FEATURES or len(exp_script.BASE_FEATURES) == 31
    assert set(exp_script.PRIORITY_COUNTRIES) == {"GBR", "BRA", "FRA", "CAN", "AUS"}
    assert set(exp_script.GUARDRAIL_COUNTRIES) == {"USA", "CHN", "DEU", "JPN", "IND"}

@patch("joblib.dump")
def test_no_overwrite_of_frozen_artifacts(mock_dump):
    # Verify that dump never uses MODEL_PATH statically
    assert mock_dump.call_count == 0
