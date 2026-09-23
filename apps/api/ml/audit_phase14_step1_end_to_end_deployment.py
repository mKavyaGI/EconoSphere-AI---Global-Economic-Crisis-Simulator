"""
Phase 14 Step 1: End-to-End Deployment Audit
Simulates a production startup and inference flow.
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.production_health import _compute_file_hashes, _MODEL_PATH, _DATASET_PATH, _MANIFEST_PATH

def get_snapshot():
    model_md5, model_sha256 = _compute_file_hashes(_MODEL_PATH)
    data_md5, data_sha256 = _compute_file_hashes(_DATASET_PATH)
    manifest_md5, manifest_sha256 = _compute_file_hashes(_MANIFEST_PATH)
    return {
        "model_md5": model_md5,
        "model_sha256": model_sha256,
        "data_md5": data_md5,
        "data_sha256": data_sha256,
        "manifest_md5": manifest_md5,
        "manifest_sha256": manifest_sha256
    }

def create_valid_payload():
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

async def run_audit():
    print("Starting Phase 14 Step 1 End-to-End Audit...\n")
    
    # 1. Hashes before
    hashes_before = get_snapshot()
    
    # Temporarily monkey patch settings for production validation
    from app.core.config import settings
    original_env = settings.env
    original_safety = settings.production_safety_mode
    original_origins = settings.allowed_origins
    
    settings.env = "production"
    settings.production_safety_mode = True
    settings.allowed_origins = ["https://frontend.example.com"]
    
    metadata = {
        "phase": "Phase 14",
        "step": "Step 1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment_mode": "production",
        "production_safety_mode": True,
        "dataset_hash_before": hashes_before["data_md5"],
        "model_hash_before": hashes_before["model_md5"],
        "manifest_hash_before": hashes_before["manifest_md5"]
    }
    
    print("--- 2. Configuration Validation ---")
    try:
        settings.validate_production_cors()
        print("PASS: Valid production CORS allowed.")
        metadata["cors_security_status"] = "PASS"
    except Exception as e:
        print(f"FAIL: {e}")
        metadata["cors_security_status"] = "FAIL"
        
    print("--- 3. Application Startup ---")
    from app.main import app, lifespan
    client = TestClient(app)
    
    try:
        # We start the client which runs the lifespan
        with TestClient(app) as test_client:
            print("PASS: Application started successfully with production settings.")
            metadata["deployment_validation_status"] = "PASS"
            
            # 4. Model Info
            resp = test_client.get("/api/v1/forecasts/model-info")
            data = resp.json()
            if "model_version" in data and data.get("status") == "PHASE 11 PRODUCTION FROZEN":
                print("PASS: Model info is correct.")
                metadata["model_info_status"] = "PASS"
            else:
                metadata["model_info_status"] = "FAIL"
                
            # 5. Health
            resp = test_client.get("/api/v1/forecasts/health")
            if resp.json().get("status") == "healthy":
                print("PASS: Health endpoint reports healthy.")
                metadata["health_status"] = "PASS"
            else:
                metadata["health_status"] = "FAIL"
                
            # 6. Readiness
            resp = test_client.get("/api/v1/forecasts/readiness")
            if resp.json().get("ready") is True:
                print("PASS: Readiness endpoint reports ready.")
                metadata["readiness_status"] = "PASS"
            else:
                metadata["readiness_status"] = "FAIL"
                
            # 7 & 8. Valid Inference & Consistency
            from app.ml.phase11_production_model import production_model
            payload = create_valid_payload()
            resp = test_client.post("/api/v1/forecasts/gdp", json=payload)
            if resp.status_code == 200:
                api_pred = resp.json()["predicted_gdp_growth"]
                
                # Direct prediction for consistency
                import pandas as pd
                features_only = {k: v for k, v in payload.items() if k not in ("country", "year")}
                # ensure proper order as expected by model (from Phase11ProductionModel features list)
                df = pd.DataFrame([features_only])[production_model._feature_names]
                direct_pred = production_model._model.predict(df)[0]
                
                if abs(api_pred - direct_pred) < 1e-12:
                    print(f"PASS: Inference successful and consistent. Prediction: {api_pred}")
                    metadata["inference_status"] = "PASS"
                    metadata["prediction_consistency_status"] = "PASS"
                else:
                    metadata["prediction_consistency_status"] = "FAIL"
            else:
                print(f"FAIL: Inference returned {resp.status_code}")
                metadata["inference_status"] = "FAIL"
                metadata["prediction_consistency_status"] = "FAIL"
                
            # 9. Candidate B Rejection
            cb_payload = create_valid_payload()
            cb_payload["growth_regime_num"] = 1.0
            resp = test_client.post("/api/v1/forecasts/gdp", json=cb_payload)
            if resp.status_code == 422:
                print("PASS: Candidate B features rejected.")
                metadata["candidate_b_isolation_status"] = "PASS"
            else:
                metadata["candidate_b_isolation_status"] = "FAIL"
                
            # 10. Target Leakage Rejection
            tl_payload = create_valid_payload()
            tl_payload["gdp_growth_next_year"] = 5.0
            resp = test_client.post("/api/v1/forecasts/gdp", json=tl_payload)
            if resp.status_code == 422:
                print("PASS: Target leakage features rejected.")
                metadata["target_leakage_protection_status"] = "PASS"
            else:
                metadata["target_leakage_protection_status"] = "FAIL"
                
    except Exception as e:
        print(f"FAIL: Application startup failed: {e}")
        metadata["deployment_validation_status"] = "FAIL"
        
    # 13. Fail-Closed Startup
    print("--- 13. Fail-Closed Startup ---")
    import app.services.production_health as ph
    original_model_path = ph._MODEL_PATH
    ph._MODEL_PATH = "/tmp/fake_path.joblib"
    try:
        with TestClient(app) as test_client:
            metadata["fail_closed_status"] = "FAIL"
    except RuntimeError as e:
        if "integrity check failed" in str(e):
            print("PASS: App failed closed correctly.")
            metadata["fail_closed_status"] = "PASS"
        else:
            metadata["fail_closed_status"] = "FAIL"
    ph._MODEL_PATH = original_model_path
    
    # Restore settings
    settings.env = original_env
    settings.production_safety_mode = original_safety
    settings.allowed_origins = original_origins
    
    # 14. Hashes after
    hashes_after = get_snapshot()
    metadata["dataset_hash_after"] = hashes_after["data_md5"]
    metadata["model_hash_after"] = hashes_after["model_md5"]
    metadata["manifest_hash_after"] = hashes_after["manifest_md5"]
    
    immu = True
    if hashes_before["data_md5"] != hashes_after["data_md5"]: immu = False
    if hashes_before["model_md5"] != hashes_after["model_md5"]: immu = False
    if hashes_before["manifest_md5"] != hashes_after["manifest_md5"]: immu = False
    metadata["immutability_status"] = "PASS" if immu else "FAIL"
    
    metadata["overall_status"] = "PASS" if all(v == "PASS" for k,v in metadata.items() if k.endswith("_status")) else "FAIL"
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/phase14"))
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "phase14_step1_end_to_end_deployment_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    csv_path = os.path.join(out_dir, "phase14_step1_end_to_end_deployment.csv")
    import pandas as pd
    df = pd.DataFrame([metadata])
    df.to_csv(csv_path, index=False)
    
    print("\n--- END-TO-END DEPLOYMENT VALIDATION RESULT ---")
    for k, v in metadata.items():
        if k.endswith("_status"):
            print(f"{k.replace('_status', '').replace('_', ' ').title():35}: {v}")

if __name__ == "__main__":
    asyncio.run(run_audit())
