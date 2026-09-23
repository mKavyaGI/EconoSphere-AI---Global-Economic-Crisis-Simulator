import os
import json
import copy
import hashlib
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.ml.phase11_production_model import production_model, FrozenModelVerificationError
import math

client = TestClient(app)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
MANIFEST_PATH = os.path.join(BASE_DIR, 'models/phase11/phase11_production_release_manifest.json')
MODEL_PATH = os.path.join(BASE_DIR, 'models/phase11/best_t1_gdp_growth_model.joblib')
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data/raw/master_panel.csv')

def calculate_hashes(file_path):
    if not os.path.exists(file_path):
        return None, None
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
            sha256_hash.update(byte_block)
    return md5_hash.hexdigest(), sha256_hash.hexdigest()

@pytest.fixture(scope="module")
def valid_payload():
    return {
        "country": "USA",
        "year": 2025,
        "exchange_rate_lcu_usd": 1.0,
        "tariff_rate_pct": 2.5,
        "remittances_usd": 100000.0,
        "fdi_net_inflow_usd": 50000.0,
        "unemployment_pct": 4.0,
        "imports_pct_gdp": 15.0,
        "tax_revenue_pct_gdp": 25.0,
        "exports_pct_gdp": 12.0,
        "interest_rate_pct": 5.0,
        "reserves_usd": 1000000.0,
        "current_account_pct_gdp": -2.0,
        "inflation_cpi_pct": 3.0,
        "population_total": 330000000.0,
        "gdp_current_usd": 25000000000000.0,
        "gdp_growth_lag1": 2.5,
        "gdp_growth_lag2": 2.0,
        "gdp_growth_lag3": 1.5,
        "inflation_lag1": 4.0,
        "unemployment_lag1": 3.5,
        "exports_lag1": 11.5,
        "imports_lag1": 14.5,
        "gdp_growth_rolling_mean_3": 2.0,
        "gdp_growth_rolling_std_3": 0.5,
        "gdp_growth_rolling_mean_5": 2.2,
        "inflation_rolling_mean_3": 3.5,
        "trade_openness": 27.0,
        "trade_balance_ratio": 0.8,
        "log_gdp_usd": 30.8,
        "log_population": 19.6,
        "gdp_growth_rolling_std_5": 0.4,
        "inflation_rolling_std_3": 0.5
    }

def test_production_model_loads():
    assert production_model._model is not None

def test_model_class_is_correct():
    model_class = production_model._model.__class__.__name__
    if model_class == "Pipeline":
        model_class = production_model._model.steps[-1][1].__class__.__name__
    assert model_class == "HistGradientBoostingRegressor"

def test_model_has_31_features():
    assert len(production_model._feature_names) == 31

def test_hyperparameters():
    model = production_model._model
    if model.__class__.__name__ == "Pipeline":
        params = model.steps[-1][1].get_params()
    else:
        params = model.get_params()
    
    assert params.get('max_depth') == 5
    assert params.get('l2_regularization') == 5.0
    assert params.get('learning_rate') == 0.05
    assert params.get('max_iter') == 300
    assert params.get('random_state') == 42

def test_model_hashes():
    md5, sha256 = calculate_hashes(MODEL_PATH)
    assert md5 == "9c539735897eaf6e72e5f54f047390d7"
    assert sha256 == "748be64bcd3402375c4df4e45f9ba0c6858f52744714808ae8a2445f29248e78"
    assert production_model._manifest["model_md5"] == md5
    assert production_model._manifest["model_sha256"] == sha256

def test_release_id_and_status():
    assert production_model._manifest["release_id"] == "ECONOSPHERE-PHASE11-PROD-2026-08-21"
    assert production_model._manifest["production_status"] == "PHASE 11 PRODUCTION FROZEN"

def test_model_identity_endpoint():
    response = client.get("/api/v1/forecasts/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "ECONOSPHERE-PHASE11-PROD-2026-08-21"
    assert data["status"] == "PHASE 11 PRODUCTION FROZEN"
    assert data["feature_count"] == 31
    assert "release_fingerprint" in data

def test_valid_inference(valid_payload):
    response = client.post("/api/v1/forecasts/gdp", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["country"] == "USA"
    assert data["year"] == 2025
    assert "predicted_gdp_growth" in data
    assert math.isfinite(data["predicted_gdp_growth"])
    assert data["model_version"] == "ECONOSPHERE-PHASE11-PROD-2026-08-21"
    assert data["production_status"] == "PHASE 11 PRODUCTION FROZEN"

def test_target_rejected(valid_payload):
    payload = valid_payload.copy()
    payload["gdp_growth_next_year"] = 5.0
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422 # Pydantic extra=forbid
    
    payload = valid_payload.copy()
    payload["target"] = 5.0
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422
    
    payload = valid_payload.copy()
    payload["target_year"] = 2026
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422

def test_candidate_b_rejected(valid_payload):
    payload = valid_payload.copy()
    payload["growth_regime_num"] = 1
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422

    payload = valid_payload.copy()
    payload["stress_regime_num"] = 2
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422

def test_missing_required_feature_rejected(valid_payload):
    payload = valid_payload.copy()
    del payload["exchange_rate_lcu_usd"]
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422

def test_unknown_feature_rejected(valid_payload):
    payload = valid_payload.copy()
    payload["some_made_up_feature"] = 42.0
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422

def test_invalid_input_type(valid_payload):
    payload = valid_payload.copy()
    payload["exchange_rate_lcu_usd"] = "not_a_number"
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422

def test_post_inference_integrity(valid_payload):
    # Initial hashes
    md5_model_1, _ = calculate_hashes(MODEL_PATH)
    md5_data_1, _ = calculate_hashes(DATA_RAW_PATH)
    
    # Run inference
    client.post("/api/v1/forecasts/gdp", json=valid_payload)
    
    # Check hashes again
    md5_model_2, _ = calculate_hashes(MODEL_PATH)
    md5_data_2, _ = calculate_hashes(DATA_RAW_PATH)
    
    assert md5_model_1 == md5_model_2
    assert md5_data_1 == md5_data_2
    
    # Ensure it didn't change from frozen state
    assert md5_model_2 == "9c539735897eaf6e72e5f54f047390d7"
    assert md5_data_2 == "8ac7e0b2bf09fbe89289f82d0c7cf25e"

def test_hash_verification_fails_closed(monkeypatch):
    from app.ml.phase11_production_model import Phase11ProductionModel
    
    # Create a new instance for test and mock the manifest hash
    mock_manifest = copy.deepcopy(production_model._manifest)
    mock_manifest["model_md5"] = "invalid_hash"
    
    with monkeypatch.context() as m:
        m.setattr("json.load", lambda x: mock_manifest)
        try:
            # We must recreate the instance to trigger the init again
            temp_model = Phase11ProductionModel.__new__(Phase11ProductionModel)
            temp_model._initialize()
            assert False, "Should have raised FrozenModelVerificationError"
        except FrozenModelVerificationError as e:
            assert "Model MD5 mismatch" in str(e)
    
    # Restore outside monkeypatch context
    production_model._initialize()


def test_unexpected_configuration_fails_closed(monkeypatch):
    from app.ml.phase11_production_model import Phase11ProductionModel
    
    mock_manifest = copy.deepcopy(production_model._manifest)
    mock_manifest["hyperparameters"]["max_depth"] = 100 # Invalid
    
    with monkeypatch.context() as m:
        m.setattr("json.load", lambda x: mock_manifest)
        try:
            temp_model = Phase11ProductionModel.__new__(Phase11ProductionModel)
            temp_model._initialize()
            assert False, "Should have raised FrozenModelVerificationError"
        except FrozenModelVerificationError as e:
            assert "Hyperparameter mismatch" in str(e)

    # Restore outside monkeypatch context
    production_model._initialize()
