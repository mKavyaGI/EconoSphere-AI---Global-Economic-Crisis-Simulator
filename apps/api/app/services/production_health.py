"""
EconoSphere AI — Phase 13 Step 2 Production Health Service
===========================================================
Read-only health verification service for the frozen Phase 11 production model.

Design contract:
  - NEVER calls joblib.load() — model is already loaded and verified by the
    Phase11ProductionModel singleton at startup.
  - NEVER modifies the model, dataset, or release manifest.
  - NEVER exposes absolute filesystem paths, raw hashes, or stack traces
    in its public return values.
  - Recomputes artifact hashes live (read-only file I/O) for integrity checks.
  - Delegates configuration and schema state to the already-verified singleton.

Health statuses:
  "healthy"     — all checks pass; service is fit for production inference.
  "degraded"    — partial failure; service may be serving stale/unverified state.
  "unavailable" — critical failure; inference must not proceed.

Usage:
  from app.services.production_health import run_full_health_check, run_readiness_check
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Protected artifact paths — relative to project root, resolved at runtime.
# Paths are never returned to API callers.
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)
_MODEL_PATH = os.path.join(_BASE_DIR, "models", "phase11", "best_t1_gdp_growth_model.joblib")
_MANIFEST_PATH = os.path.join(_BASE_DIR, "models", "phase11", "phase11_production_release_manifest.json")
_DATASET_PATH = os.path.join(_BASE_DIR, "data", "raw", "master_panel.csv")

# ---------------------------------------------------------------------------
# Expected frozen baseline constants (immutable; match release manifest)
# ---------------------------------------------------------------------------
_EXPECTED_RELEASE_ID = "ECONOSPHERE-PHASE11-PROD-2026-08-21"
_EXPECTED_PRODUCTION_STATUS = "PHASE 11 PRODUCTION FROZEN"
_EXPECTED_FEATURE_COUNT = 31
_EXPECTED_MODEL_CLASS = "HistGradientBoostingRegressor"
_EXPECTED_HYPERPARAMETERS: Dict[str, Any] = {
    "max_depth": 5,
    "l2_regularization": 5.0,
    "learning_rate": 0.05,
    "max_iter": 300,
    "random_state": 42,
}
_CANDIDATE_B_FEATURES: List[str] = ["growth_regime_num", "stress_regime_num"]
_BANNED_TARGET_FEATURES: List[str] = [
    "gdp_growth_next_year",
    "target",
    "t1_gdp_growth",
    "target_year",
]
_SERVICE_NAME = "econosphere-ai-production-inference"
_MODEL_VERSION = "phase11"


# ---------------------------------------------------------------------------
# Internal helpers — never exposed directly in API responses
# ---------------------------------------------------------------------------

def _compute_file_hashes(path: str) -> tuple[str, str]:
    """Compute MD5 and SHA-256 of a file. Returns (md5_hex, sha256_hex)."""
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _get_production_model():
    """
    Return the Phase11ProductionModel singleton without triggering re-init.
    Import is deferred to avoid circular imports.
    """
    from app.ml.phase11_production_model import production_model
    return production_model


# ---------------------------------------------------------------------------
# Individual health checks
# ---------------------------------------------------------------------------

def check_model_availability() -> Dict[str, Any]:
    """
    Verify that the production model singleton is loaded and holds a non-None
    model object. Does not call joblib.load().

    Returns:
        {
            "status": "healthy" | "unavailable",
            "model_version": str | None,
            "production_status": str | None,
        }
    """
    try:
        model = _get_production_model()
        if model._model is None:
            return {
                "status": "unavailable",
                "model_version": None,
                "production_status": None,
            }
        manifest = model._manifest or {}
        return {
            "status": "healthy",
            "model_version": manifest.get("release_id", "unknown"),
            "production_status": manifest.get("production_status", "unknown"),
        }
    except Exception:
        logger.error("check_model_availability: unexpected error", exc_info=True)
        return {
            "status": "unavailable",
            "model_version": None,
            "production_status": None,
        }


def check_artifact_integrity() -> Dict[str, Any]:
    """
    Re-verify the model artifact against the release manifest by recomputing
    hashes. This is a live, read-only file integrity check.

    Returns:
        {
            "status": "verified" | "failed",
            "md5_match": bool,
            "sha256_match": bool,
            "release_id_match": bool,
            "production_status_match": bool,
            "fingerprint_present": bool,
        }

    NOTE: Actual hash values are NEVER included in the return dict.
    """
    result: Dict[str, Any] = {
        "status": "failed",
        "md5_match": False,
        "sha256_match": False,
        "release_id_match": False,
        "production_status_match": False,
        "fingerprint_present": False,
    }
    try:
        model = _get_production_model()
        manifest = model._manifest
        if manifest is None:
            logger.warning("check_artifact_integrity: manifest not loaded in singleton")
            return result

        # Verify manifest fields
        result["release_id_match"] = (
            manifest.get("release_id") == _EXPECTED_RELEASE_ID
        )
        result["production_status_match"] = (
            manifest.get("production_status") == _EXPECTED_PRODUCTION_STATUS
        )
        result["fingerprint_present"] = bool(manifest.get("release_fingerprint"))

        # Live hash computation — read-only
        if not os.path.exists(_MODEL_PATH):
            logger.warning("check_artifact_integrity: model file not found")
            return result

        actual_md5, actual_sha256 = _compute_file_hashes(_MODEL_PATH)
        expected_md5 = manifest.get("model_md5", "")
        expected_sha256 = manifest.get("model_sha256", "")

        result["md5_match"] = (actual_md5 == expected_md5)
        result["sha256_match"] = (actual_sha256 == expected_sha256)

        if all([
            result["md5_match"],
            result["sha256_match"],
            result["release_id_match"],
            result["production_status_match"],
            result["fingerprint_present"],
        ]):
            result["status"] = "verified"
        else:
            logger.warning(
                "check_artifact_integrity: integrity check failed",
                md5_match=result["md5_match"],
                sha256_match=result["sha256_match"],
                release_id_match=result["release_id_match"],
            )

    except Exception:
        logger.error("check_artifact_integrity: unexpected error", exc_info=True)

    return result


def check_schema_integrity() -> Dict[str, Any]:
    """
    Verify the production model's feature schema:
      - exactly 31 features
      - correct feature names in expected order
      - no target, target_year, growth_regime_num, stress_regime_num present

    Returns:
        {
            "status": "verified" | "failed",
            "feature_count": int | None,
            "feature_count_correct": bool,
            "feature_order_correct": bool,
            "target_excluded": bool,
            "target_year_excluded": bool,
            "candidate_b_excluded": bool,
        }
    """
    result: Dict[str, Any] = {
        "status": "failed",
        "feature_count": None,
        "feature_count_correct": False,
        "feature_order_correct": False,
        "target_excluded": False,
        "target_year_excluded": False,
        "candidate_b_excluded": False,
    }
    try:
        model = _get_production_model()
        feature_names = model._feature_names
        manifest = model._manifest

        if feature_names is None or manifest is None:
            return result

        count = len(feature_names)
        result["feature_count"] = count
        result["feature_count_correct"] = (count == _EXPECTED_FEATURE_COUNT)

        expected_features = manifest.get("feature_names", [])
        result["feature_order_correct"] = (list(feature_names) == list(expected_features))

        banned_targets = _BANNED_TARGET_FEATURES
        result["target_excluded"] = not any(f in feature_names for f in banned_targets)
        result["target_year_excluded"] = "target_year" not in feature_names
        result["candidate_b_excluded"] = not any(
            f in feature_names for f in _CANDIDATE_B_FEATURES
        )

        if all([
            result["feature_count_correct"],
            result["feature_order_correct"],
            result["target_excluded"],
            result["target_year_excluded"],
            result["candidate_b_excluded"],
        ]):
            result["status"] = "verified"
        else:
            logger.warning(
                "check_schema_integrity: schema check failed",
                feature_count=count,
                feature_count_correct=result["feature_count_correct"],
                candidate_b_excluded=result["candidate_b_excluded"],
            )

    except Exception:
        logger.error("check_schema_integrity: unexpected error", exc_info=True)

    return result


def check_model_configuration() -> Dict[str, Any]:
    """
    Verify the loaded model's hyperparameters match the expected frozen baseline.

    Returns:
        {
            "status": "verified" | "failed",
            "model_class_correct": bool,
            "max_depth_correct": bool,
            "l2_regularization_correct": bool,
            "learning_rate_correct": bool,
            "max_iter_correct": bool,
            "random_state_correct": bool,
        }
    """
    result: Dict[str, Any] = {
        "status": "failed",
        "model_class_correct": False,
        "max_depth_correct": False,
        "l2_regularization_correct": False,
        "learning_rate_correct": False,
        "max_iter_correct": False,
        "random_state_correct": False,
    }
    try:
        model = _get_production_model()
        loaded = model._model
        if loaded is None:
            return result

        # Resolve regressor (handles Pipeline wrapper)
        if loaded.__class__.__name__ == "Pipeline":
            regressor = loaded.steps[-1][1]
        else:
            regressor = loaded

        result["model_class_correct"] = (
            regressor.__class__.__name__ == _EXPECTED_MODEL_CLASS
        )

        params = regressor.get_params()
        result["max_depth_correct"] = (params.get("max_depth") == _EXPECTED_HYPERPARAMETERS["max_depth"])
        result["l2_regularization_correct"] = (
            params.get("l2_regularization") == _EXPECTED_HYPERPARAMETERS["l2_regularization"]
        )
        result["learning_rate_correct"] = (
            params.get("learning_rate") == _EXPECTED_HYPERPARAMETERS["learning_rate"]
        )
        result["max_iter_correct"] = (params.get("max_iter") == _EXPECTED_HYPERPARAMETERS["max_iter"])
        result["random_state_correct"] = (
            params.get("random_state") == _EXPECTED_HYPERPARAMETERS["random_state"]
        )

        if all([
            result["model_class_correct"],
            result["max_depth_correct"],
            result["l2_regularization_correct"],
            result["learning_rate_correct"],
            result["max_iter_correct"],
            result["random_state_correct"],
        ]):
            result["status"] = "verified"
        else:
            logger.warning(
                "check_model_configuration: configuration check failed",
                model_class_correct=result["model_class_correct"],
                max_depth_correct=result["max_depth_correct"],
            )

    except Exception:
        logger.error("check_model_configuration: unexpected error", exc_info=True)

    return result


def check_candidate_b_isolation() -> Dict[str, Any]:
    """
    Verify that Candidate B experimental features are not present in the
    production model's schema, confirming isolation.

    Returns:
        {
            "status": "isolated" | "contaminated",
            "candidate_b_status": str,
            "growth_regime_num_absent": bool,
            "stress_regime_num_absent": bool,
        }
    """
    result: Dict[str, Any] = {
        "status": "contaminated",
        "candidate_b_status": "EXPERIMENTAL — NOT PRODUCTION",
        "growth_regime_num_absent": False,
        "stress_regime_num_absent": False,
    }
    try:
        model = _get_production_model()
        feature_names = model._feature_names or []

        result["growth_regime_num_absent"] = "growth_regime_num" not in feature_names
        result["stress_regime_num_absent"] = "stress_regime_num" not in feature_names

        # Also verify the manifest declares candidate_b_status as experimental
        manifest = model._manifest or {}
        cb_status = manifest.get("candidate_b_status", "")
        # Accept any non-production declaration
        candidate_b_is_experimental = "NOT PRODUCTION" in cb_status or "EXPERIMENTAL" in cb_status
        result["candidate_b_status"] = cb_status if cb_status else "EXPERIMENTAL — NOT PRODUCTION"

        if (
            result["growth_regime_num_absent"]
            and result["stress_regime_num_absent"]
            and candidate_b_is_experimental
        ):
            result["status"] = "isolated"
        else:
            logger.warning(
                "check_candidate_b_isolation: Candidate B features detected in production schema"
            )

    except Exception:
        logger.error("check_candidate_b_isolation: unexpected error", exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Aggregated checks
# ---------------------------------------------------------------------------

def run_full_health_check() -> Dict[str, Any]:
    """
    Run all health checks and return an aggregated operational summary.

    Returns a dict safe for direct API response — no paths, hashes, or
    internal exception details.

    Status rules:
      "healthy"     — all checks pass
      "degraded"    — partial failure (artifact or schema issue)
      "unavailable" — model not loaded or critical integrity failure
    """
    t_start = time.monotonic()

    availability = check_model_availability()
    artifact = check_artifact_integrity()
    schema = check_schema_integrity()
    configuration = check_model_configuration()
    candidate_b = check_candidate_b_isolation()

    # Determine aggregate status
    if availability["status"] == "unavailable":
        overall = "unavailable"
    elif artifact["status"] != "verified" or configuration["status"] != "verified":
        overall = "unavailable"  # fail-closed for integrity/config failures
    elif schema["status"] != "verified" or candidate_b["status"] != "isolated":
        overall = "degraded"
    else:
        overall = "healthy"

    duration_ms = round((time.monotonic() - t_start) * 1000, 2)

    result = {
        "status": overall,
        "service": _SERVICE_NAME,
        "model_status": availability["status"],
        "artifact_integrity": artifact["status"],
        "schema_integrity": schema["status"],
        "configuration_integrity": configuration["status"],
        "candidate_b_isolation": candidate_b["status"],
        "production_status": availability.get("production_status") or _EXPECTED_PRODUCTION_STATUS,
        "health_check_duration_ms": duration_ms,
    }

    logger.info(
        "production_health_check_complete",
        status=overall,
        artifact_integrity=artifact["status"],
        schema_integrity=schema["status"],
        configuration_integrity=configuration["status"],
        candidate_b_isolation=candidate_b["status"],
        duration_ms=duration_ms,
    )

    return result


def run_readiness_check() -> Dict[str, Any]:
    """
    Strict readiness gate: can this service safely serve production predictions?

    Returns True only when ALL of the following pass:
      1. frozen artifact integrity
      2. manifest integrity (release ID + production status)
      3. model configuration verification
      4. 31-feature schema verification
      5. Candidate B isolation
      6. production model availability

    On any failure, returns {"ready": false} with safe structured check data.
    No internal exception details or paths are exposed.
    """
    t_start = time.monotonic()

    availability = check_model_availability()
    artifact = check_artifact_integrity()
    schema = check_schema_integrity()
    configuration = check_model_configuration()
    candidate_b = check_candidate_b_isolation()

    checks = {
        "artifact_integrity": artifact["status"] == "verified",
        "model_configuration": configuration["status"] == "verified",
        "feature_schema": (
            schema["status"] == "verified"
            and schema.get("feature_count_correct", False)
        ),
        "candidate_b_isolation": candidate_b["status"] == "isolated",
        "model_available": availability["status"] == "healthy",
    }

    ready = all(checks.values())
    duration_ms = round((time.monotonic() - t_start) * 1000, 2)

    manifest_status = availability.get("production_status") or _EXPECTED_PRODUCTION_STATUS

    logger.info(
        "production_readiness_check_complete",
        ready=ready,
        checks=checks,
        duration_ms=duration_ms,
    )

    return {
        "ready": ready,
        "model_version": _MODEL_VERSION,
        "production_status": manifest_status,
        "checks": checks,
        "readiness_check_duration_ms": duration_ms,
    }


def get_detailed_health_report() -> Dict[str, Any]:
    """
    Returns detailed per-check results for administrative/audit use.
    Still safe — no raw hash values or filesystem paths in output.
    """
    availability = check_model_availability()
    artifact = check_artifact_integrity()
    schema = check_schema_integrity()
    configuration = check_model_configuration()
    candidate_b = check_candidate_b_isolation()

    return {
        "model_availability": availability,
        "artifact_integrity": artifact,
        "schema_integrity": schema,
        "model_configuration": configuration,
        "candidate_b_isolation": candidate_b,
    }
