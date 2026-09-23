"""
EconoSphere AI — Phase 13 Step 2 Operational Safety Test Suite
===============================================================
Tests for production reliability, observability, and operational safety.

Covers:
  - Health checks (model availability, artifact integrity, schema, configuration)
  - Readiness gate (all-or-nothing pass/fail)
  - Health endpoint (/api/v1/forecasts/health)
  - Readiness endpoint (/api/v1/forecasts/readiness)
  - Model-info consistency with health information
  - Configuration integrity (all 5 hyperparameters)
  - Schema protection (31 features, banned fields)
  - Error safety (fail-closed, no internal detail leakage)
  - Prediction consistency (1e-12 tolerance)
  - Artifact immutability (model, dataset, manifest unchanged after inference)
  - Candidate B isolation

Usage:
  python -X utf8 -m pytest apps/api/tests/test_phase13_step2_operational_safety.py -v
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.phase11_production_model import production_model, FrozenModelVerificationError
from app.services.production_health import (
    check_artifact_integrity,
    check_candidate_b_isolation,
    check_model_availability,
    check_model_configuration,
    check_schema_integrity,
    run_full_health_check,
    run_readiness_check,
    get_detailed_health_report,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Paths and hash constants
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parents[3]  # project root
_MODEL_PATH = _BASE_DIR / "models" / "phase11" / "best_t1_gdp_growth_model.joblib"
_MANIFEST_PATH = _BASE_DIR / "models" / "phase11" / "phase11_production_release_manifest.json"
_DATASET_PATH = _BASE_DIR / "data" / "raw" / "master_panel.csv"

_EXPECTED_MODEL_MD5 = "9c539735897eaf6e72e5f54f047390d7"
_EXPECTED_MODEL_SHA256 = "748be64bcd3402375c4df4e45f9ba0c6858f52744714808ae8a2445f29248e78"
_EXPECTED_DATASET_MD5 = "8ac7e0b2bf09fbe89289f82d0c7cf25e"

PREDICTION_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def valid_payload() -> Dict:
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
        "inflation_rolling_std_3": 0.5,
    }


@pytest.fixture(scope="module")
def valid_features(valid_payload) -> Dict:
    """Feature dict extracted from valid_payload (excludes routing fields)."""
    return {k: v for k, v in valid_payload.items() if k not in ("country", "year")}


def compute_file_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            md5.update(chunk)
    return md5.hexdigest()


# ===========================================================================
# 1. HEALTH CHECKS — model availability, health endpoint, status
# ===========================================================================

class TestHealthChecks:

    def test_01_healthy_model_reports_healthy_status(self):
        """Test 1: A verified, loaded model reports healthy status."""
        result = check_model_availability()
        assert result["status"] == "healthy", (
            f"Expected 'healthy', got '{result['status']}'"
        )
        assert result["model_version"] is not None
        assert result["production_status"] == "PHASE 11 PRODUCTION FROZEN"

    def test_02_readiness_returns_true_for_verified_frozen_baseline(self):
        """Test 2: Readiness check returns ready=True for the frozen Phase 11 baseline."""
        result = run_readiness_check()
        assert result["ready"] is True, f"Readiness failed: {result}"
        assert result["model_version"] == "phase11"
        assert result["production_status"] == "PHASE 11 PRODUCTION FROZEN"

    def test_03_model_info_consistent_with_health(self):
        """Test 3: /model-info is consistent with health check model_version and status."""
        info_resp = client.get("/api/v1/forecasts/model-info")
        assert info_resp.status_code == 200
        info = info_resp.json()

        health_result = check_model_availability()
        assert info["model_version"] == health_result["model_version"]
        assert info["status"] == health_result["production_status"]

    def test_04_health_checks_do_not_mutate_protected_artifacts(self, valid_payload):
        """Test 4: Running health checks does not change any protected artifact on disk."""
        md5_model_before = compute_file_md5(_MODEL_PATH)
        md5_dataset_before = compute_file_md5(_DATASET_PATH)
        md5_manifest_before = compute_file_md5(_MANIFEST_PATH)

        # Run all health sub-checks
        run_full_health_check()
        run_readiness_check()
        get_detailed_health_report()
        check_artifact_integrity()
        check_schema_integrity()
        check_model_configuration()
        check_candidate_b_isolation()

        md5_model_after = compute_file_md5(_MODEL_PATH)
        md5_dataset_after = compute_file_md5(_DATASET_PATH)
        md5_manifest_after = compute_file_md5(_MANIFEST_PATH)

        assert md5_model_before == md5_model_after, "Model file changed after health checks"
        assert md5_dataset_before == md5_dataset_after, "Dataset changed after health checks"
        assert md5_manifest_before == md5_manifest_after, "Manifest changed after health checks"

    def test_05_health_endpoint_returns_200_and_healthy(self):
        """Test 5: GET /api/v1/forecasts/health returns 200 and status=healthy."""
        resp = client.get("/api/v1/forecasts/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "econosphere-ai-production-inference"
        assert data["artifact_integrity"] == "verified"
        assert data["schema_integrity"] == "verified"
        assert data["configuration_integrity"] == "verified"
        assert "production_status" in data

    def test_06_readiness_endpoint_returns_200_and_ready(self):
        """Test 6: GET /api/v1/forecasts/readiness returns ready=true."""
        resp = client.get("/api/v1/forecasts/readiness")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert "checks" in data
        checks = data["checks"]
        assert checks["artifact_integrity"] is True
        assert checks["model_configuration"] is True
        assert checks["feature_schema"] is True
        assert checks["candidate_b_isolation"] is True
        assert checks["model_available"] is True

    def test_07_full_health_check_contains_all_fields(self):
        """Test 7: run_full_health_check() returns all required fields."""
        result = run_full_health_check()
        required = {
            "status", "service", "model_status", "artifact_integrity",
            "schema_integrity", "configuration_integrity",
            "candidate_b_isolation", "production_status",
        }
        for field in required:
            assert field in result, f"Missing field: {field}"


# ===========================================================================
# 2. CONFIGURATION INTEGRITY
# ===========================================================================

class TestConfigurationIntegrity:

    def test_08_model_class_is_histgradientboostingregressor(self):
        """Test 8: Model class is HistGradientBoostingRegressor."""
        result = check_model_configuration()
        assert result["model_class_correct"] is True
        assert result["status"] == "verified"

    def test_09_feature_count_equals_31(self):
        """Test 9: Feature count equals exactly 31."""
        result = check_schema_integrity()
        assert result["feature_count"] == 31
        assert result["feature_count_correct"] is True

    def test_10_max_depth_equals_5(self):
        """Test 10: max_depth == 5."""
        result = check_model_configuration()
        assert result["max_depth_correct"] is True
        # Verify directly from model params
        model = production_model._model
        regressor = model if model.__class__.__name__ != "Pipeline" else model.steps[-1][1]
        assert regressor.get_params()["max_depth"] == 5

    def test_11_l2_regularization_equals_5_0(self):
        """Test 11: l2_regularization == 5.0."""
        result = check_model_configuration()
        assert result["l2_regularization_correct"] is True
        model = production_model._model
        regressor = model if model.__class__.__name__ != "Pipeline" else model.steps[-1][1]
        assert regressor.get_params()["l2_regularization"] == 5.0

    def test_12_learning_rate_equals_0_05(self):
        """Test 12: learning_rate == 0.05."""
        result = check_model_configuration()
        assert result["learning_rate_correct"] is True
        model = production_model._model
        regressor = model if model.__class__.__name__ != "Pipeline" else model.steps[-1][1]
        assert regressor.get_params()["learning_rate"] == 0.05

    def test_13_max_iter_equals_300(self):
        """Test 13: max_iter == 300."""
        result = check_model_configuration()
        assert result["max_iter_correct"] is True
        model = production_model._model
        regressor = model if model.__class__.__name__ != "Pipeline" else model.steps[-1][1]
        assert regressor.get_params()["max_iter"] == 300

    def test_14_random_state_equals_42(self):
        """Test 14: random_state == 42."""
        result = check_model_configuration()
        assert result["random_state_correct"] is True
        model = production_model._model
        regressor = model if model.__class__.__name__ != "Pipeline" else model.steps[-1][1]
        assert regressor.get_params()["random_state"] == 42


# ===========================================================================
# 3. SCHEMA PROTECTION
# ===========================================================================

class TestSchemaProtection:

    def test_15_valid_payload_succeeds(self, valid_payload):
        """Test 15: A valid 31-feature payload returns 200 and a finite prediction."""
        resp = client.post("/api/v1/forecasts/gdp", json=valid_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_gdp_growth" in data
        assert math.isfinite(data["predicted_gdp_growth"])
        assert data["production_status"] == "PHASE 11 PRODUCTION FROZEN"

    def test_16_unknown_extra_field_rejected(self, valid_payload):
        """Test 16: An unknown extra field is rejected with 422."""
        payload = {**valid_payload, "some_made_up_feature": 99.0}
        resp = client.post("/api/v1/forecasts/gdp", json=payload)
        assert resp.status_code == 422

    def test_17_growth_regime_num_rejected(self, valid_payload):
        """Test 17: growth_regime_num (Candidate B) is rejected with 422."""
        payload = {**valid_payload, "growth_regime_num": 1}
        resp = client.post("/api/v1/forecasts/gdp", json=payload)
        assert resp.status_code == 422

    def test_18_stress_regime_num_rejected(self, valid_payload):
        """Test 18: stress_regime_num (Candidate B) is rejected with 422."""
        payload = {**valid_payload, "stress_regime_num": 2}
        resp = client.post("/api/v1/forecasts/gdp", json=payload)
        assert resp.status_code == 422

    def test_19_gdp_growth_next_year_rejected(self, valid_payload):
        """Test 19: gdp_growth_next_year (target leakage) is rejected with 422."""
        payload = {**valid_payload, "gdp_growth_next_year": 5.0}
        resp = client.post("/api/v1/forecasts/gdp", json=payload)
        assert resp.status_code == 422

    def test_20_target_year_rejected(self, valid_payload):
        """Test 20: target_year is rejected with 422."""
        payload = {**valid_payload, "target_year": 2026}
        resp = client.post("/api/v1/forecasts/gdp", json=payload)
        assert resp.status_code == 422

    def test_21_missing_required_feature_rejected(self, valid_payload):
        """Test 21: A missing required feature is rejected (422 from Pydantic)."""
        payload = {k: v for k, v in valid_payload.items() if k != "exchange_rate_lcu_usd"}
        resp = client.post("/api/v1/forecasts/gdp", json=payload)
        assert resp.status_code == 422

    def test_22_target_field_rejected(self, valid_payload):
        """Test 22: Generic 'target' field (leakage) is rejected with 422."""
        payload = {**valid_payload, "target": 3.5}
        resp = client.post("/api/v1/forecasts/gdp", json=payload)
        assert resp.status_code == 422

    def test_23_schema_integrity_verified_by_health_service(self):
        """Test 23: check_schema_integrity() returns verified=True for the production schema."""
        result = check_schema_integrity()
        assert result["status"] == "verified"
        assert result["feature_count_correct"] is True
        assert result["target_excluded"] is True
        assert result["target_year_excluded"] is True
        assert result["candidate_b_excluded"] is True
        assert result["feature_order_correct"] is True


# ===========================================================================
# 4. ERROR SAFETY
# ===========================================================================

class TestErrorSafety:

    def test_24_artifact_verification_failure_causes_fail_closed(self):
        """
        Test 24: When artifact integrity fails, check_artifact_integrity() returns
        status='failed', and the health endpoint reports non-healthy.
        We simulate this by patching the singleton's manifest md5.
        """
        original_md5 = production_model._manifest.get("model_md5")
        try:
            production_model._manifest["model_md5"] = "000000000000000000000000deadbeef"
            result = check_artifact_integrity()
            assert result["status"] == "failed"
            assert result["md5_match"] is False
        finally:
            # Always restore the correct value
            production_model._manifest["model_md5"] = original_md5

    def test_25_health_check_restored_after_tampered_manifest(self):
        """Test 25: After restoring manifest, health check returns to healthy."""
        # Sanity check that test_24's cleanup worked
        result = check_artifact_integrity()
        assert result["status"] == "verified"
        assert result["md5_match"] is True

    def test_26_model_info_does_not_expose_internal_exception_details(self):
        """
        Test 26: When /model-info returns normally, the response does NOT
        contain the secret model MD5 or SHA-256 hash values, or absolute paths.
        The release_fingerprint IS an intentional public identifier (not a secret hash).
        """
        resp = client.get("/api/v1/forecasts/model-info")
        assert resp.status_code == 200
        body = resp.text
        # No absolute filesystem paths
        assert "D:\\" not in body and "/Development/" not in body
        # The actual secret model hash values must not appear in the response
        assert _EXPECTED_MODEL_MD5 not in body, "Model MD5 hash leaked in /model-info"
        assert _EXPECTED_MODEL_SHA256 not in body, "Model SHA-256 hash leaked in /model-info"

    def test_27_gdp_endpoint_does_not_expose_stack_trace(self, valid_payload):
        """
        Test 27: A successful /gdp response does not contain Python stack trace
        indicators, absolute paths, or secret model hash values.
        """
        resp = client.post("/api/v1/forecasts/gdp", json=valid_payload)
        assert resp.status_code == 200
        body = resp.text
        assert "Traceback" not in body
        assert "D:\\" not in body and "/Development/" not in body
        # Actual hash values must not be present
        assert _EXPECTED_MODEL_MD5 not in body, "Model MD5 leaked in /gdp response"
        assert _EXPECTED_MODEL_SHA256 not in body, "Model SHA-256 leaked in /gdp response"

    def test_28_invalid_inference_returns_deterministic_422(self, valid_payload):
        """Test 28: Invalid inference requests (wrong type) return deterministic 422."""
        payload = {**valid_payload, "exchange_rate_lcu_usd": "not_a_number"}
        resp1 = client.post("/api/v1/forecasts/gdp", json=payload)
        resp2 = client.post("/api/v1/forecasts/gdp", json=payload)
        assert resp1.status_code == 422
        assert resp2.status_code == 422
        # Both responses should have the same structure
        assert resp1.json().get("code") == resp2.json().get("code")

    def test_29_readiness_fails_closed_when_model_unavailable(self):
        """
        Test 29: If model._model is patched to None, readiness returns ready=False.
        """
        original_model = production_model._model
        try:
            production_model._model = None
            result = run_readiness_check()
            assert result["ready"] is False
            assert result["checks"]["model_available"] is False
        finally:
            production_model._model = original_model

    def test_30_health_endpoint_fails_closed_when_model_unavailable(self):
        """
        Test 30: If model._model is patched to None, /health returns non-healthy status.
        """
        original_model = production_model._model
        try:
            production_model._model = None
            resp = client.get("/api/v1/forecasts/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] != "healthy"
        finally:
            production_model._model = original_model


# ===========================================================================
# 5. CONSISTENCY AND ARTIFACT INTEGRITY
# ===========================================================================

class TestConsistencyAndIntegrity:

    def test_31_production_prediction_matches_direct_model_prediction(self, valid_features):
        """
        Test 31: The prediction through the API matches the direct model call
        within tolerance 1e-12.
        """
        # Direct call
        direct_pred = production_model.predict(valid_features)

        # API call
        full_payload = {"country": "USA", "year": 2025, **valid_features}
        resp = client.post("/api/v1/forecasts/gdp", json=full_payload)
        assert resp.status_code == 200
        api_pred = resp.json()["predicted_gdp_growth"]

        diff = abs(direct_pred - api_pred)
        assert diff < PREDICTION_TOLERANCE, (
            f"Prediction mismatch: direct={direct_pred:.15f}, api={api_pred:.15f}, "
            f"|diff|={diff:.2e} >= 1e-12"
        )

    def test_32_model_hash_unchanged_after_inference(self, valid_payload):
        """Test 32: Model file hash is unchanged after running inference."""
        md5_before = compute_file_md5(_MODEL_PATH)

        client.post("/api/v1/forecasts/gdp", json=valid_payload)
        client.post("/api/v1/forecasts/gdp", json=valid_payload)

        md5_after = compute_file_md5(_MODEL_PATH)
        assert md5_before == md5_after == _EXPECTED_MODEL_MD5

    def test_33_dataset_hash_unchanged_after_inference(self, valid_payload):
        """Test 33: Raw dataset hash is unchanged after running inference."""
        md5_before = compute_file_md5(_DATASET_PATH)

        client.post("/api/v1/forecasts/gdp", json=valid_payload)

        md5_after = compute_file_md5(_DATASET_PATH)
        assert md5_before == md5_after == _EXPECTED_DATASET_MD5

    def test_34_manifest_hash_unchanged_after_inference(self, valid_payload):
        """Test 34: Release manifest hash is unchanged after running inference."""
        md5_before = compute_file_md5(_MANIFEST_PATH)

        client.post("/api/v1/forecasts/gdp", json=valid_payload)

        md5_after = compute_file_md5(_MANIFEST_PATH)
        assert md5_before == md5_after

    def test_35_candidate_b_not_loaded_for_production_inference(self):
        """
        Test 35: Candidate B features (growth_regime_num, stress_regime_num) are
        absent from the production model's feature schema. Candidate B is
        EXPERIMENTAL — NOT PRODUCTION.
        """
        result = check_candidate_b_isolation()
        assert result["status"] == "isolated"
        assert result["growth_regime_num_absent"] is True
        assert result["stress_regime_num_absent"] is True

        # Directly verify feature list
        feature_names = production_model._feature_names
        assert "growth_regime_num" not in feature_names
        assert "stress_regime_num" not in feature_names

        # Verify manifest declares it experimental
        manifest = production_model._manifest
        cb_status = manifest.get("candidate_b_status", "")
        assert "NOT PRODUCTION" in cb_status or "EXPERIMENTAL" in cb_status

    def test_36_direct_model_prediction_is_deterministic(self, valid_features):
        """
        Test 36: Multiple direct model predictions with identical input produce
        numerically identical outputs (determinism check).
        """
        predictions = [production_model.predict(valid_features) for _ in range(5)]
        for i in range(1, len(predictions)):
            diff = abs(predictions[0] - predictions[i])
            assert diff < PREDICTION_TOLERANCE, (
                f"Prediction non-determinism: pred[0]={predictions[0]:.15f}, "
                f"pred[{i}]={predictions[i]:.15f}, |diff|={diff:.2e}"
            )

    def test_37_feature_names_match_manifest_order(self):
        """
        Test 37: The model's feature_names_in_ exactly matches the manifest
        feature_names list in the same order.
        """
        manifest_features = production_model._manifest.get("feature_names", [])
        model_features = list(production_model._feature_names)
        assert model_features == manifest_features, (
            f"Feature order mismatch:\n  model={model_features}\n  manifest={manifest_features}"
        )

    def test_38_model_hashes_match_manifest(self):
        """Test 38: Live-computed model hashes match the manifest values."""
        result = check_artifact_integrity()
        assert result["md5_match"] is True
        assert result["sha256_match"] is True
        assert result["release_id_match"] is True
        assert result["production_status_match"] is True
        assert result["fingerprint_present"] is True

    def test_39_readiness_check_verifies_all_five_gates(self):
        """
        Test 39: Readiness check result includes all 5 required gate keys
        and all are True for the verified baseline.
        """
        result = run_readiness_check()
        required_keys = {
            "artifact_integrity",
            "model_configuration",
            "feature_schema",
            "candidate_b_isolation",
            "model_available",
        }
        assert required_keys.issubset(set(result["checks"].keys()))
        for key in required_keys:
            assert result["checks"][key] is True, f"Readiness gate failed: {key}"

    def test_40_inference_observability_does_not_alter_prediction(self, valid_payload, valid_features):
        """
        Test 40: Running inference through the logged API endpoint produces
        the exact same result as a direct model call (logging does not alter output).
        """
        # Direct prediction (no logging overhead)
        direct_pred = production_model.predict(valid_features)

        # API prediction (with structured logging)
        resp = client.post("/api/v1/forecasts/gdp", json=valid_payload)
        assert resp.status_code == 200
        api_pred = resp.json()["predicted_gdp_growth"]

        diff = abs(direct_pred - api_pred)
        assert diff < PREDICTION_TOLERANCE, (
            f"Logging altered the prediction: "
            f"direct={direct_pred:.15f}, api={api_pred:.15f}, |diff|={diff:.2e}"
        )

    def test_41_health_check_does_not_call_joblib_load(self, monkeypatch):
        """
        Test 41: run_full_health_check() and run_readiness_check() do not
        trigger joblib.load() (model is served from the existing singleton).
        """
        load_calls = []

        def mock_load(*args, **kwargs):
            load_calls.append(args)
            return MagicMock()

        monkeypatch.setattr("joblib.load", mock_load)

        # These should use the already-loaded singleton, not joblib.load
        run_full_health_check()
        run_readiness_check()

        assert len(load_calls) == 0, (
            f"Health checks triggered joblib.load() {len(load_calls)} time(s). "
            "Health checks must not re-load the model."
        )

    def test_42_configuration_mismatch_triggers_failed_status(self):
        """
        Test 42: Patching a hyperparameter on the singleton causes
        check_model_configuration() to return status='failed'.
        """
        model = production_model._model
        regressor = model if model.__class__.__name__ != "Pipeline" else model.steps[-1][1]
        original_depth = regressor.max_depth
        try:
            regressor.max_depth = 999  # invalid
            result = check_model_configuration()
            assert result["status"] == "failed"
            assert result["max_depth_correct"] is False
        finally:
            regressor.max_depth = original_depth

    def test_43_schema_check_detects_candidate_b_contamination(self):
        """
        Test 43: If Candidate B feature is injected into feature_names,
        check_schema_integrity() reports status='failed'.
        """
        original_features = list(production_model._feature_names)
        try:
            production_model._feature_names = original_features + ["growth_regime_num"]
            result = check_schema_integrity()
            assert result["candidate_b_excluded"] is False
            assert result["status"] == "failed"
        finally:
            production_model._feature_names = original_features

    def test_44_detailed_health_report_returns_all_check_categories(self):
        """
        Test 44: get_detailed_health_report() returns all 5 check categories.
        """
        report = get_detailed_health_report()
        expected_keys = {
            "model_availability",
            "artifact_integrity",
            "schema_integrity",
            "model_configuration",
            "candidate_b_isolation",
        }
        assert expected_keys.issubset(set(report.keys()))

    def test_45_health_endpoint_does_not_expose_hash_values(self):
        """
        Test 45: The /health endpoint response body does not contain the actual
        secret model MD5 or SHA-256 hash values, nor any absolute filesystem paths.
        """
        resp = client.get("/api/v1/forecasts/health")
        assert resp.status_code == 200
        body = resp.text
        # Actual secret hash values must not appear in the health response
        assert _EXPECTED_MODEL_MD5 not in body, "Model MD5 hash exposed in /health response"
        assert _EXPECTED_MODEL_SHA256 not in body, "Model SHA-256 hash exposed in /health response"
        # No filesystem paths
        assert "D:\\" not in body and "/Development/" not in body
