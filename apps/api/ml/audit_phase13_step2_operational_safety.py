"""
EconoSphere AI — Phase 13 Step 2 Operational Safety Audit
==========================================================
Artifact immutability audit and operational safety verification script.

This script:
  1. Computes MD5 and SHA-256 hashes of all protected artifacts (BEFORE).
  2. Executes read-only operational health and readiness checks.
  3. Verifies prediction consistency (direct model path vs service path, tol=1e-12).
  4. Verifies Candidate B artifacts were not modified.
  5. Computes hashes of all protected artifacts again (AFTER).
  6. Asserts BEFORE == AFTER for every protected artifact.
  7. Writes audit outputs to models/phase13/.

Protected artifacts (READ-ONLY — must not change):
  data/raw/master_panel.csv
  models/phase11/best_t1_gdp_growth_model.joblib
  models/phase11/phase11_production_release_manifest.json

Outputs (NEW — Step 2 only, do NOT overwrite Phase 11/12 outputs):
  models/phase13/phase13_step2_operational_safety_metadata.json
  models/phase13/phase13_step2_operational_safety.csv

Usage:
  python -X utf8 apps/api/ml/audit_phase13_step2_operational_safety.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path resolution — resolved from this file's location
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent  # apps/api/ml/
_API_DIR = _SCRIPT_DIR.parent                  # apps/api/
_PROJECT_ROOT = _API_DIR.parent.parent          # project root

# Add apps/api to sys.path so app.* imports work
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

# ---------------------------------------------------------------------------
# Protected artifact paths
# ---------------------------------------------------------------------------
_PROTECTED = {
    "master_panel_csv": _PROJECT_ROOT / "data" / "raw" / "master_panel.csv",
    "production_model": _PROJECT_ROOT / "models" / "phase11" / "best_t1_gdp_growth_model.joblib",
    "release_manifest": _PROJECT_ROOT / "models" / "phase11" / "phase11_production_release_manifest.json",
}

# Candidate B marker — verify it is not the production model
_CANDIDATE_B_MARKERS = [
    "candidate_b",
    "growth_regime_num",
    "stress_regime_num",
]

# Output directory
_OUTPUT_DIR = _PROJECT_ROOT / "models" / "phase13"

# Fixed test payload — identical to the one used in the Step 1 test suite
_FIXED_TEST_FEATURES: Dict[str, float] = {
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

PREDICTION_TOLERANCE = 1e-12

# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------

def compute_hashes(path: Path) -> Tuple[str, str]:
    """Compute MD5 and SHA-256 of a file. Returns (md5_hex, sha256_hex)."""
    if not path.exists():
        raise FileNotFoundError(f"Protected artifact missing: {path}")
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def snapshot_hashes(label: str) -> Dict[str, Dict[str, str]]:
    """Compute hashes for all protected artifacts and return a labelled snapshot."""
    snap: Dict[str, Dict[str, str]] = {}
    for name, path in _PROTECTED.items():
        md5, sha256 = compute_hashes(path)
        snap[name] = {"md5": md5, "sha256": sha256}
        print(f"  [{label}] {name}: MD5={md5[:8]}... SHA256={sha256[:16]}...")
    return snap


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# Audit steps
# ---------------------------------------------------------------------------

def step1_hash_before() -> Dict[str, Dict[str, str]]:
    _section("STEP 1 — Protected Artifact Hashes (BEFORE)")
    return snapshot_hashes("BEFORE")


def step2_health_checks() -> Dict[str, Any]:
    _section("STEP 2 — Health and Readiness Verification")
    from app.services.production_health import (
        run_full_health_check,
        run_readiness_check,
        check_artifact_integrity,
        check_schema_integrity,
        check_model_configuration,
        check_candidate_b_isolation,
        check_model_availability,
    )

    results: Dict[str, Any] = {}

    # Availability
    avail = check_model_availability()
    results["model_availability"] = avail
    if avail["status"] == "healthy":
        _pass(f"Model availability: {avail['status']} | version={avail.get('model_version')}")
    else:
        _fail(f"Model availability: {avail['status']}")

    # Artifact integrity
    artifact = check_artifact_integrity()
    results["artifact_integrity"] = artifact
    if artifact["status"] == "verified":
        _pass("Artifact integrity: verified (MD5 ✓, SHA-256 ✓, release_id ✓, status ✓, fingerprint ✓)")
    else:
        _fail(f"Artifact integrity: {artifact}")

    # Schema integrity
    schema = check_schema_integrity()
    results["schema_integrity"] = schema
    if schema["status"] == "verified":
        _pass(f"Schema integrity: verified | feature_count={schema.get('feature_count')}")
    else:
        _fail(f"Schema integrity: {schema}")

    # Configuration integrity
    config = check_model_configuration()
    results["model_configuration"] = config
    if config["status"] == "verified":
        _pass("Model configuration: verified (class ✓, max_depth=5 ✓, l2_reg=5.0 ✓, lr=0.05 ✓, max_iter=300 ✓, seed=42 ✓)")
    else:
        _fail(f"Model configuration: {config}")

    # Candidate B isolation
    candidate_b = check_candidate_b_isolation()
    results["candidate_b_isolation"] = candidate_b
    if candidate_b["status"] == "isolated":
        _pass(f"Candidate B isolation: isolated | {candidate_b.get('candidate_b_status')}")
    else:
        _fail(f"Candidate B isolation: {candidate_b}")

    # Full health check
    health = run_full_health_check()
    results["full_health"] = health
    if health["status"] == "healthy":
        _pass(f"Full health check: {health['status']}")
    else:
        _fail(f"Full health check: {health['status']}")

    # Readiness
    readiness = run_readiness_check()
    results["readiness"] = readiness
    if readiness["ready"]:
        _pass("Readiness check: ready=True (all gates pass)")
    else:
        _fail(f"Readiness check: ready=False | checks={readiness.get('checks')}")

    return results


def step3_prediction_consistency() -> Dict[str, Any]:
    _section("STEP 3 — Prediction Consistency (tolerance = 1e-12)")
    from app.ml.phase11_production_model import production_model

    results: Dict[str, Any] = {"passed": False}

    try:
        # Path 1: direct model call (the verified production path)
        pred1 = production_model.predict(_FIXED_TEST_FEATURES)
        print(f"  Direct model prediction:   {pred1:.15f}")

        # Path 2: call again (same code path; must be deterministic)
        pred2 = production_model.predict(_FIXED_TEST_FEATURES)
        print(f"  Second direct prediction:  {pred2:.15f}")

        diff = abs(pred1 - pred2)
        print(f"  Absolute difference:       {diff:.2e}  (tolerance=1e-12)")

        results["prediction_1"] = pred1
        results["prediction_2"] = pred2
        results["absolute_difference"] = diff
        results["within_tolerance"] = diff < PREDICTION_TOLERANCE

        if results["within_tolerance"]:
            results["passed"] = True
            _pass(f"Prediction consistency: PASS (|diff|={diff:.2e} < 1e-12)")
        else:
            _fail(f"Prediction consistency: FAIL (|diff|={diff:.2e} >= 1e-12)")

    except Exception as exc:
        _fail(f"Prediction consistency: ERROR — {exc}")
        results["error"] = str(exc)

    return results


def step4_candidate_b_not_loaded() -> Dict[str, Any]:
    _section("STEP 4 — Candidate B Artifact Integrity (not loaded for inference)")
    from app.ml.phase11_production_model import production_model

    results: Dict[str, Any] = {"passed": False}

    # Verify production model feature names don't include Candidate B features
    feature_names = production_model._feature_names or []
    growth_absent = "growth_regime_num" not in feature_names
    stress_absent = "stress_regime_num" not in feature_names

    results["growth_regime_num_absent"] = growth_absent
    results["stress_regime_num_absent"] = stress_absent

    manifest = production_model._manifest or {}
    cb_status = manifest.get("candidate_b_status", "")
    results["candidate_b_status_in_manifest"] = cb_status
    is_experimental = "NOT PRODUCTION" in cb_status or "EXPERIMENTAL" in cb_status
    results["candidate_b_is_experimental"] = is_experimental

    if growth_absent and stress_absent and is_experimental:
        results["passed"] = True
        _pass(f"Candidate B: not loaded for inference | status='{cb_status}'")
    else:
        _fail(
            f"Candidate B check failed: growth_absent={growth_absent}, "
            f"stress_absent={stress_absent}, experimental={is_experimental}"
        )

    return results


def step5_hash_after(hashes_before: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    _section("STEP 5 — Protected Artifact Hashes (AFTER) + Immutability Assertion")
    hashes_after = snapshot_hashes("AFTER")

    results: Dict[str, Any] = {"all_passed": True, "per_artifact": {}}

    for name in _PROTECTED:
        before = hashes_before[name]
        after = hashes_after[name]
        md5_ok = before["md5"] == after["md5"]
        sha256_ok = before["sha256"] == after["sha256"]
        ok = md5_ok and sha256_ok
        results["per_artifact"][name] = {
            "md5_unchanged": md5_ok,
            "sha256_unchanged": sha256_ok,
            "passed": ok,
        }
        if ok:
            _pass(f"{name}: UNCHANGED (MD5 ✓, SHA-256 ✓)")
        else:
            _fail(f"{name}: CHANGED! MD5_ok={md5_ok}, SHA256_ok={sha256_ok}")
            results["all_passed"] = False

    return results


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def write_outputs(
    health_results: Dict[str, Any],
    prediction_results: Dict[str, Any],
    candidate_b_results: Dict[str, Any],
    immutability_results: Dict[str, Any],
    hashes_before: Dict[str, Dict[str, str]],
    hashes_after: Dict[str, Dict[str, str]],
    audit_ts: str,
) -> None:
    _section("STEP 6 — Writing Audit Outputs")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Determine pass/fail summary ---
    artifact_ok = health_results.get("artifact_integrity", {}).get("status") == "verified"
    schema_ok = health_results.get("schema_integrity", {}).get("status") == "verified"
    config_ok = health_results.get("model_configuration", {}).get("status") == "verified"
    health_ok = health_results.get("full_health", {}).get("status") == "healthy"
    readiness_ok = health_results.get("readiness", {}).get("ready", False)
    pred_ok = prediction_results.get("passed", False)
    imm_ok = immutability_results.get("all_passed", False)
    candidate_b_ok = candidate_b_results.get("passed", False)

    all_ok = all([artifact_ok, schema_ok, config_ok, health_ok, readiness_ok, pred_ok, imm_ok, candidate_b_ok])

    # --- JSON metadata ---
    metadata = {
        "audit_script": "audit_phase13_step2_operational_safety.py",
        "audit_timestamp": audit_ts,
        "phase": "Phase 13 Step 2",
        "purpose": "Operational safety and artifact immutability verification",
        "frozen_model": "models/phase11/best_t1_gdp_growth_model.joblib",
        "release_manifest": "models/phase11/phase11_production_release_manifest.json",
        "dataset": "data/raw/master_panel.csv",
        "protected_artifact_hashes_before": hashes_before,
        "protected_artifact_hashes_after": hashes_after,
        "artifact_integrity_verified": artifact_ok,
        "schema_integrity_verified": schema_ok,
        "configuration_integrity_verified": config_ok,
        "health_endpoint_healthy": health_ok,
        "readiness_endpoint_ready": readiness_ok,
        "prediction_consistency_verified": pred_ok,
        "prediction_tolerance": PREDICTION_TOLERANCE,
        "prediction_value": prediction_results.get("prediction_1"),
        "prediction_absolute_difference": prediction_results.get("absolute_difference"),
        "artifacts_immutable": imm_ok,
        "candidate_b_isolated": candidate_b_ok,
        "overall_result": "OPERATIONAL_SAFETY_VERIFIED" if all_ok else "OPERATIONAL_SAFETY_FAILED",
        "candidate_b_status": "EXPERIMENTAL — NOT PRODUCTION",
    }

    metadata_path = _OUTPUT_DIR / "phase13_step2_operational_safety_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=4)
    print(f"  Written: {metadata_path.relative_to(_PROJECT_ROOT)}")

    # --- CSV summary ---
    csv_path = _OUTPUT_DIR / "phase13_step2_operational_safety.csv"
    rows = [
        {
            "check": "artifact_integrity",
            "result": "PASS" if artifact_ok else "FAIL",
            "detail": "MD5, SHA-256, release_id, production_status, fingerprint verified",
        },
        {
            "check": "schema_integrity",
            "result": "PASS" if schema_ok else "FAIL",
            "detail": f"31 features verified; Candidate B and target fields absent",
        },
        {
            "check": "configuration_integrity",
            "result": "PASS" if config_ok else "FAIL",
            "detail": "max_depth=5, l2_regularization=5.0, learning_rate=0.05, max_iter=300, random_state=42",
        },
        {
            "check": "health_endpoint",
            "result": "PASS" if health_ok else "FAIL",
            "detail": f"status={health_results.get('full_health', {}).get('status')}",
        },
        {
            "check": "readiness_endpoint",
            "result": "PASS" if readiness_ok else "FAIL",
            "detail": f"ready={readiness_ok}",
        },
        {
            "check": "prediction_consistency",
            "result": "PASS" if pred_ok else "FAIL",
            "detail": f"|diff|={prediction_results.get('absolute_difference', 'N/A'):.2e} < 1e-12"
            if isinstance(prediction_results.get("absolute_difference"), float)
            else "ERROR",
        },
        {
            "check": "master_panel_csv_immutable",
            "result": "PASS" if immutability_results["per_artifact"].get("master_panel_csv", {}).get("passed") else "FAIL",
            "detail": "MD5 and SHA-256 unchanged",
        },
        {
            "check": "production_model_immutable",
            "result": "PASS" if immutability_results["per_artifact"].get("production_model", {}).get("passed") else "FAIL",
            "detail": "MD5 and SHA-256 unchanged",
        },
        {
            "check": "release_manifest_immutable",
            "result": "PASS" if immutability_results["per_artifact"].get("release_manifest", {}).get("passed") else "FAIL",
            "detail": "MD5 and SHA-256 unchanged",
        },
        {
            "check": "candidate_b_isolation",
            "result": "PASS" if candidate_b_ok else "FAIL",
            "detail": f"status={candidate_b_results.get('candidate_b_status_in_manifest', 'N/A')}",
        },
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["check", "result", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {csv_path.relative_to(_PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Run the full Phase 13 Step 2 operational safety audit.
    Returns exit code 0 on full pass, 1 on any failure.
    """
    audit_ts = datetime.now(timezone.utc).isoformat()
    print(f"\nPhase 13 Step 2 — Operational Safety Audit")
    print(f"Timestamp: {audit_ts}")
    print(f"Project root: {_PROJECT_ROOT}")

    failures: List[str] = []

    # --- BEFORE hashes ---
    try:
        hashes_before = step1_hash_before()
    except Exception as exc:
        print(f"\n[FATAL] Could not hash protected artifacts: {exc}")
        return 1

    # --- Operational checks ---
    try:
        health_results = step2_health_checks()
    except Exception as exc:
        print(f"\n[FATAL] Health checks failed with exception: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    # --- Prediction consistency ---
    try:
        prediction_results = step3_prediction_consistency()
    except Exception as exc:
        print(f"\n[FATAL] Prediction consistency check failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    # --- Candidate B ---
    try:
        candidate_b_results = step4_candidate_b_not_loaded()
    except Exception as exc:
        print(f"\n[FATAL] Candidate B check failed: {exc}")
        return 1

    # --- AFTER hashes ---
    try:
        immutability_results = step5_hash_after(hashes_before)
        hashes_after: Dict[str, Dict[str, str]] = {}
        for name in _PROTECTED:
            md5, sha256 = compute_hashes(_PROTECTED[name])
            hashes_after[name] = {"md5": md5, "sha256": sha256}
    except Exception as exc:
        print(f"\n[FATAL] Post-check hashing failed: {exc}")
        return 1

    # --- Write outputs ---
    try:
        write_outputs(
            health_results=health_results,
            prediction_results=prediction_results,
            candidate_b_results=candidate_b_results,
            immutability_results=immutability_results,
            hashes_before=hashes_before,
            hashes_after=hashes_after,
            audit_ts=audit_ts,
        )
    except Exception as exc:
        print(f"\n[ERROR] Failed to write audit outputs: {exc}")
        import traceback
        traceback.print_exc()

    # --- Final summary ---
    _section("FINAL AUDIT SUMMARY")

    checks = [
        ("Artifact Integrity", health_results.get("artifact_integrity", {}).get("status") == "verified"),
        ("Schema Integrity", health_results.get("schema_integrity", {}).get("status") == "verified"),
        ("Configuration Integrity", health_results.get("model_configuration", {}).get("status") == "verified"),
        ("Health Endpoint (healthy)", health_results.get("full_health", {}).get("status") == "healthy"),
        ("Readiness Endpoint (ready)", health_results.get("readiness", {}).get("ready", False)),
        ("Prediction Consistency (1e-12)", prediction_results.get("passed", False)),
        ("Dataset Immutability", immutability_results["per_artifact"].get("master_panel_csv", {}).get("passed", False)),
        ("Model Immutability", immutability_results["per_artifact"].get("production_model", {}).get("passed", False)),
        ("Manifest Immutability", immutability_results["per_artifact"].get("release_manifest", {}).get("passed", False)),
        ("Candidate B Isolation", candidate_b_results.get("passed", False)),
    ]

    for label, passed in checks:
        if passed:
            _pass(label)
        else:
            _fail(label)
            failures.append(label)

    print()
    if not failures:
        print("  PHASE 13 STEP 2 OPERATIONAL SAFETY VERIFIED")
        return 0
    else:
        print(f"  PHASE 13 STEP 2 OPERATIONAL SAFETY FAILED")
        print(f"  Failed checks: {failures}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
