import os
import copy
import json
import math
import hashlib
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.ml.phase11_production_model import production_model, FrozenModelVerificationError
import app.services.production_health as ph

client = TestClient(app)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
MANIFEST_PATH = os.path.join(BASE_DIR, 'models/phase11/phase11_production_release_manifest.json')
MODEL_PATH = os.path.join(BASE_DIR, 'models/phase11/best_t1_gdp_growth_model.joblib')
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data/raw/master_panel.csv')

def get_hash(path):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

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

# --- CONFIGURATION TESTS ---

def test_1_valid_dev_config():
    assert settings.env == "development"

def test_2_production_rejects_wildcard():
    from app.core.config import Settings
    with pytest.raises(ValueError, match="wildcard"):
        Settings(env="production", allowed_origins=["*"])

def test_3_production_rejects_empty_origins():
    from app.core.config import Settings
    with pytest.raises(ValueError, match="must be explicitly set"):
        Settings(env="production", allowed_origins=[])

def test_4_safety_mode_rejects_wildcard():
    from app.core.config import Settings
    with pytest.raises(ValueError, match="wildcard"):
        Settings(env="development", production_safety_mode=True, allowed_origins=["*"])

def test_5_production_accepts_valid_origin():
    from app.core.config import Settings
    settings = Settings(env="production", allowed_origins=["https://frontend.example.com"])
    assert settings.allowed_origins == ["https://frontend.example.com"]

# --- STARTUP TESTS ---

def test_6_valid_production_preflight(monkeypatch):
    monkeypatch.setattr(settings, "env", "production")
    with TestClient(app) as _:
        pass # Should not raise RuntimeError

def test_7_missing_model_aborts_startup(monkeypatch):
    monkeypatch.setattr(settings, "env", "production")
    monkeypatch.setattr(ph, "_MODEL_PATH", "/tmp/fake.joblib")
    with pytest.raises(RuntimeError, match="integrity check failed"):
        with TestClient(app) as _:
            pass

def test_8_corrupted_model_aborts_startup(monkeypatch):
    mock_manifest = copy.deepcopy(production_model._manifest)
    mock_manifest["model_md5"] = "invalid"
    monkeypatch.setattr(production_model, "_manifest", mock_manifest)
    monkeypatch.setattr(settings, "env", "production")
    with pytest.raises(RuntimeError):
        with TestClient(app) as _:
            pass

def test_9_invalid_schema_aborts_startup(monkeypatch):
    mock_manifest = copy.deepcopy(production_model._manifest)
    mock_manifest["feature_names"][0] = "invalid_feature"
    monkeypatch.setattr(production_model, "_manifest", mock_manifest)
    monkeypatch.setattr(settings, "env", "production")
    with pytest.raises(RuntimeError):
        with TestClient(app) as _:
            pass

def test_10_invalid_candidate_b_aborts_startup(monkeypatch):
    mock_manifest = copy.deepcopy(production_model._manifest)
    mock_manifest["feature_names"].append("growth_regime_num")
    mock_feature_names = list(production_model._feature_names) + ["growth_regime_num"]
    monkeypatch.setattr(production_model, "_manifest", mock_manifest)
    monkeypatch.setattr(production_model, "_feature_names", mock_feature_names)
    monkeypatch.setattr(settings, "env", "production")
    with pytest.raises(RuntimeError):
        with TestClient(app) as _:
            pass

# --- ENDPOINT TESTS ---

def test_11_model_info_returns_200():
    response = client.get("/api/v1/forecasts/model-info")
    assert response.status_code == 200

def test_12_model_info_content():
    response = client.get("/api/v1/forecasts/model-info")
    data = response.json()
    assert data["model_version"] == "ECONOSPHERE-PHASE11-PROD-2026-08-21"
    assert data["status"] == "PHASE 11 PRODUCTION FROZEN"
    assert data["feature_count"] == 31

def test_13_health_returns_200():
    response = client.get("/api/v1/forecasts/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_14_readiness_returns_200():
    response = client.get("/api/v1/forecasts/readiness")
    assert response.status_code == 200
    assert response.json()["ready"] is True

def test_15_valid_gdp_inference(valid_payload):
    response = client.post("/api/v1/forecasts/gdp", json=valid_payload)
    assert response.status_code == 200
    assert "predicted_gdp_growth" in response.json()

def test_16_missing_feature(valid_payload):
    payload = valid_payload.copy()
    del payload["exchange_rate_lcu_usd"]
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422

def test_17_extra_feature(valid_payload):
    payload = valid_payload.copy()
    payload["extra_thing"] = 4.0
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422

# --- SECURITY TESTS ---

def test_18_reject_growth_regime(valid_payload):
    payload = valid_payload.copy()
    payload["growth_regime_num"] = 1.0
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422

def test_19_reject_stress_regime(valid_payload):
    payload = valid_payload.copy()
    payload["stress_regime_num"] = 2.0
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422

def test_20_reject_target_leakage_next_year(valid_payload):
    payload = valid_payload.copy()
    payload["gdp_growth_next_year"] = 5.0
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422

def test_21_reject_target_leakage_target_year(valid_payload):
    payload = valid_payload.copy()
    payload["target_year"] = 2026
    response = client.post("/api/v1/forecasts/gdp", json=payload)
    assert response.status_code == 422

def test_22_hash_non_leakage(monkeypatch):
    monkeypatch.setattr(ph, "_MODEL_PATH", "/tmp/fake.joblib")
    response = client.get("/api/v1/forecasts/readiness")
    # In API response, it should not expose the hash or full stack
    text = response.text
    assert "748be64bcd" not in text
    assert "9c53973589" not in text

def test_23_path_non_leakage(monkeypatch):
    monkeypatch.setattr(ph, "_MODEL_PATH", "/tmp/super_secret_path.joblib")
    response = client.get("/api/v1/forecasts/readiness")
    text = response.text
    assert "/tmp/super_secret_path.joblib" not in text

def test_24_stack_trace_non_leakage(monkeypatch):
    def bad_hash(*args): raise ValueError("Secret Stack Trace Error")
    monkeypatch.setattr(ph, "_compute_file_hashes", bad_hash)
    response = client.get("/api/v1/forecasts/readiness")
    assert "Secret Stack Trace Error" not in response.text
    assert "Traceback" not in response.text

# --- PREDICTION TESTS ---

def test_25_prediction_consistency(valid_payload):
    import pandas as pd
    response = client.post("/api/v1/forecasts/gdp", json=valid_payload)
    api_pred = response.json()["predicted_gdp_growth"]
    
    features_only = {k: v for k, v in valid_payload.items() if k not in ("country", "year")}
    df = pd.DataFrame([features_only])[production_model._feature_names]
    direct_pred = production_model._model.predict(df)[0]
    assert abs(api_pred - direct_pred) < 1e-12

def test_26_deterministic_repeated_inference(valid_payload):
    resp1 = client.post("/api/v1/forecasts/gdp", json=valid_payload)
    resp2 = client.post("/api/v1/forecasts/gdp", json=valid_payload)
    assert resp1.json()["predicted_gdp_growth"] == resp2.json()["predicted_gdp_growth"]

def test_27_response_metadata(valid_payload):
    resp = client.post("/api/v1/forecasts/gdp", json=valid_payload)
    data = resp.json()
    assert data["model_version"] == "ECONOSPHERE-PHASE11-PROD-2026-08-21"
    assert data["production_status"] == "PHASE 11 PRODUCTION FROZEN"

def test_28_finite_prediction(valid_payload):
    resp = client.post("/api/v1/forecasts/gdp", json=valid_payload)
    assert math.isfinite(resp.json()["predicted_gdp_growth"])

# --- IMMUTABILITY TESTS ---

def test_29_dataset_unchanged():
    assert get_hash(DATA_RAW_PATH) == "8ac7e0b2bf09fbe89289f82d0c7cf25e"

def test_30_model_unchanged():
    assert get_hash(MODEL_PATH) == "9c539735897eaf6e72e5f54f047390d7"

def test_31_manifest_unchanged():
    assert get_hash(MANIFEST_PATH) == "9c673e2d5bcd12adc171a55bd86ca34e"

def test_32_no_dump_during_inference(monkeypatch, valid_payload):
    dump_called = False
    def mock_dump(*args, **kwargs):
        nonlocal dump_called
        dump_called = True
    import joblib
    monkeypatch.setattr(joblib, "dump", mock_dump)
    client.post("/api/v1/forecasts/gdp", json=valid_payload)
    assert not dump_called

def test_33_no_fit_during_inference(monkeypatch, valid_payload):
    fit_called = False
    def mock_fit(*args, **kwargs):
        nonlocal fit_called
        fit_called = True
    monkeypatch.setattr(production_model._model, "fit", mock_fit)
    client.post("/api/v1/forecasts/gdp", json=valid_payload)
    assert not fit_called

def test_34_no_set_params_during_inference(monkeypatch, valid_payload):
    set_params_called = False
    def mock_set_params(*args, **kwargs):
        nonlocal set_params_called
        set_params_called = True
    monkeypatch.setattr(production_model._model, "set_params", mock_set_params)
    client.post("/api/v1/forecasts/gdp", json=valid_payload)
    assert not set_params_called

# --- DEPLOYMENT INTEGRATION TESTS ---

def test_35_production_like_settings_exist():
    # Verify .env.production.example exists
    env_path = os.path.join(BASE_DIR, "apps/api/.env.production.example")
    assert os.path.exists(env_path)

def test_36_lifespan_execution_works():
    with TestClient(app) as test_client:
        resp = test_client.get("/api/v1/forecasts/health")
        assert resp.status_code == 200

def test_37_cors_middleware_configuration():
    from fastapi.middleware.cors import CORSMiddleware
    cors_found = any(m.cls == CORSMiddleware for m in app.user_middleware)
    assert cors_found

def test_38_readiness_before_traffic(monkeypatch):
    # Ensure run_readiness_check is called in lifespan
    called = False
    original_check = ph.run_readiness_check
    def mock_check():
        nonlocal called
        called = True
        return original_check()
    monkeypatch.setattr(ph, "run_readiness_check", mock_check)
    monkeypatch.setattr(settings, "env", "production")
    with TestClient(app) as _:
        pass
    assert called

def test_39_missing_allowed_origins_aborts_in_production():
    from app.core.config import Settings
    with pytest.raises(ValueError):
        Settings(env="production", allowed_origins=[])

def test_40_model_info_does_not_leak_paths():
    response = client.get("/api/v1/forecasts/model-info")
    text = response.text
    assert "models/phase11" not in text
