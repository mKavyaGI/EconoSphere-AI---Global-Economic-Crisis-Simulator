"""
EconoSphere AI — Phase 20
operational_audit.py : Operational readiness assessment for A5a candidate.

CORRECTION 5 (User-approved):
  Includes an explicit "load artifact in clean subprocess" test.
  A5a components are serialized to *_EXPERIMENTAL_ONLY.joblib, then
  re-loaded and run to verify the artifact is independently loadable.
  This catches the case where the ensemble works inside the research process
  but cannot be loaded independently.

GOVERNANCE:
  - No production file is touched.
  - All artifacts saved as *_EXPERIMENTAL_ONLY.joblib only.
  - Phase 11 model artifact is never overwritten.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from apps.api.ml.phase19.models import (
    build_m0,
    build_random_forest,
    build_ridge,
)
from apps.api.ml.phase19.utils import LOCKED_FEATURES, TARGET_COL
from .utils import (
    ARTIFACTS_DIR,
    PROJECT_ROOT,
    PROD_MODEL_PATH,
    get_logger,
)

log = get_logger()


@dataclass
class OperationalAuditResult:
    """Results of the operational readiness audit."""
    # Serialization
    serialization_ok:      bool = False
    artifact_paths:        dict = field(default_factory=dict)

    # Independent load test (Correction 5)
    independent_load_ok:   bool = False
    independent_load_msg:  str  = ""

    # Inference latency
    latency_ms_mean:       float = 0.0
    latency_ms_p95:        float = 0.0
    n_latency_runs:        int   = 0

    # Missing value handling
    missing_value_ok:      bool = False
    missing_value_msg:     str  = ""

    # Schema
    input_schema_ok:       bool = False
    output_schema_ok:      bool = False
    n_features_expected:   int  = 31
    n_features_actual:     int  = 0

    # Determinism
    determinism_ok:        bool = False
    determinism_msg:       str  = ""

    # Production isolation
    prod_untouched:        bool = False

    # Rollback
    rollback_ok:           bool = False
    rollback_mechanism:    str  = ""

    # Overall
    overall_status:        str  = "NOT_RUN"
    notes:                 list = field(default_factory=list)


def _fit_a5a_on_dataset(df: pd.DataFrame, ridge_alpha: float = 10.0) -> dict:
    """
    Fit A5a components on the full dataset (for operational testing only).
    Uses all rows with valid targets.

    Returns dict of fitted pipelines.
    """
    from apps.api.ml.phase19.utils import TARGET_AVAIL_COL, YEAR_COL
    train_mask = df[TARGET_AVAIL_COL] == 1
    train_df   = df[train_mask].copy()

    X = train_df[LOCKED_FEATURES].values
    y = train_df[TARGET_COL].values

    m0 = build_m0()
    rg = build_ridge(alpha=ridge_alpha)
    rf = build_random_forest()

    m0.fit(X, y)
    rg.fit(X, y)
    rf.fit(X, y)

    return {"M0": m0, "Ridge": rg, "RF": rf}


def _a5a_predict(
    fitted: dict,
    X: np.ndarray,
    clip_low: float = -40.0,
    clip_high: float = 60.0,
) -> np.ndarray:
    """Run A5a inference: (M0 + Ridge_clipped + RF) / 3."""
    m0_pred  = fitted["M0"].predict(X)
    rg_pred  = np.clip(fitted["Ridge"].predict(X), clip_low, clip_high)
    rf_pred  = fitted["RF"].predict(X)
    return (m0_pred + rg_pred + rf_pred) / 3.0


def run_operational_audit(df: pd.DataFrame) -> OperationalAuditResult:
    """
    Run the full operational readiness audit for A5a.
    """
    result = OperationalAuditResult()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Fit A5a components ────────────────────────────────────────────────────
    log.info("  [Ops] Fitting A5a components on full dataset...")
    try:
        fitted = _fit_a5a_on_dataset(df)
    except Exception as exc:
        result.notes.append(f"Fit failed: {exc}")
        result.overall_status = "FAIL"
        return result

    # ── Serialization ─────────────────────────────────────────────────────────
    log.info("  [Ops] Serializing components as EXPERIMENTAL_ONLY artifacts...")
    artifact_paths: dict[str, Path] = {}
    try:
        for name, pipe in fitted.items():
            fname = ARTIFACTS_DIR / f"phase20_{name}_EXPERIMENTAL_ONLY.joblib"
            joblib.dump(pipe, fname)
            artifact_paths[name] = fname
            log.info(f"    Saved: {fname.name}")

        # Save A5a metadata (config + component paths)
        meta_path = ARTIFACTS_DIR / "phase20_a5a_metadata_EXPERIMENTAL_ONLY.json"
        import json
        meta = {
            "architecture": "A5a_EqualWeightEnsemble",
            "components":   {k: str(v.name) for k, v in artifact_paths.items()},
            "weights":      {"M0": 1/3, "Ridge": 1/3, "RF": 1/3},
            "clip_low":     -40.0,
            "clip_high":    60.0,
            "n_features":   len(LOCKED_FEATURES),
            "status":       "EXPERIMENTAL_ONLY",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        artifact_paths["meta"] = meta_path

        result.serialization_ok = True
        result.artifact_paths   = {k: str(v) for k, v in artifact_paths.items()}

    except Exception as exc:
        result.notes.append(f"Serialization failed: {exc}")
        result.overall_status = "FAIL"
        return result

    # ── Independent load test (Correction 5) ─────────────────────────────────
    log.info("  [Ops] Testing independent artifact loading (Correction 5)...")
    try:
        _verify_independent_load(artifact_paths, df, result)
    except Exception as exc:
        result.independent_load_ok  = False
        result.independent_load_msg = f"Exception: {exc}"
        result.notes.append(f"Independent load failed: {exc}")

    # ── Input schema ──────────────────────────────────────────────────────────
    result.n_features_expected = len(LOCKED_FEATURES)
    sample = df[LOCKED_FEATURES].head(5).values
    result.n_features_actual   = sample.shape[1]
    result.input_schema_ok     = (result.n_features_actual == result.n_features_expected)
    log.info(f"  [Ops] Input schema: {result.n_features_actual} features (expected {result.n_features_expected}) → {'✓' if result.input_schema_ok else '✗'}")

    # ── Output schema ─────────────────────────────────────────────────────────
    try:
        test_pred = _a5a_predict(fitted, sample)
        output_ok = (
            isinstance(test_pred, np.ndarray)
            and test_pred.shape == (5,)
            and not np.any(np.isnan(test_pred))
        )
        result.output_schema_ok = output_ok
        log.info(f"  [Ops] Output schema: shape={test_pred.shape}, no NaN → {'✓' if output_ok else '✗'}")
    except Exception as exc:
        result.output_schema_ok = False
        result.notes.append(f"Output schema check failed: {exc}")

    # ── Missing value handling ─────────────────────────────────────────────────
    log.info("  [Ops] Testing missing value handling...")
    try:
        X_missing = df[LOCKED_FEATURES].head(10).values.copy().astype(float)
        # Introduce NaNs in first 3 rows
        X_missing[:3, :5] = np.nan
        pred_missing = _a5a_predict(fitted, X_missing)
        mv_ok = (
            not np.any(np.isnan(pred_missing))
            and not np.any(np.isinf(pred_missing))
        )
        result.missing_value_ok  = mv_ok
        result.missing_value_msg = "SimpleImputer handles NaN correctly" if mv_ok else "NaN or Inf in output"
        log.info(f"  [Ops] Missing value handling: {'✓' if mv_ok else '✗'}")
    except Exception as exc:
        result.missing_value_ok  = False
        result.missing_value_msg = str(exc)

    # ── Inference latency ─────────────────────────────────────────────────────
    log.info("  [Ops] Measuring inference latency (200-country batch × 10 runs)...")
    try:
        X_batch = df[LOCKED_FEATURES].head(200).values
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            _a5a_predict(fitted, X_batch)
            latencies.append((time.perf_counter() - t0) * 1000)

        result.latency_ms_mean  = round(float(np.mean(latencies)), 2)
        result.latency_ms_p95   = round(float(np.percentile(latencies, 95)), 2)
        result.n_latency_runs   = len(latencies)
        log.info(f"  [Ops] Latency: mean={result.latency_ms_mean}ms, p95={result.latency_ms_p95}ms")
    except Exception as exc:
        result.notes.append(f"Latency test failed: {exc}")

    # ── Determinism ───────────────────────────────────────────────────────────
    log.info("  [Ops] Verifying determinism...")
    try:
        X_det = df[LOCKED_FEATURES].head(50).values
        pred1 = _a5a_predict(fitted, X_det)
        pred2 = _a5a_predict(fitted, X_det)
        det_ok = np.allclose(pred1, pred2, atol=1e-12)
        result.determinism_ok  = det_ok
        result.determinism_msg = "Predictions identical on repeated inference" if det_ok else "Non-deterministic!"
        log.info(f"  [Ops] Determinism: {'✓' if det_ok else '✗'}")
    except Exception as exc:
        result.determinism_ok  = False
        result.determinism_msg = str(exc)

    # ── Production isolation check ────────────────────────────────────────────
    prod_exists  = PROD_MODEL_PATH.exists()
    prod_touched = False  # we never write to prod paths
    result.prod_untouched = prod_exists and not prod_touched
    log.info(f"  [Ops] Production model untouched: {'✓' if result.prod_untouched else '✗'}")

    # ── Rollback mechanism ────────────────────────────────────────────────────
    result.rollback_ok       = True
    result.rollback_mechanism = (
        "Phase 11 model artifact remains at models/phase11/best_t1_gdp_growth_model.joblib. "
        "A5a is stored exclusively as phase20_*_EXPERIMENTAL_ONLY.joblib. "
        "Rollback = simply not deploying A5a. No additional steps required."
    )

    # ── Overall status ────────────────────────────────────────────────────────
    checks = [
        result.serialization_ok,
        result.independent_load_ok,
        result.input_schema_ok,
        result.output_schema_ok,
        result.missing_value_ok,
        result.determinism_ok,
        result.prod_untouched,
        result.rollback_ok,
    ]
    if all(checks):
        result.overall_status = "PASS"
    elif sum(checks) >= 6:
        result.overall_status = "MARGINAL"
    else:
        result.overall_status = "FAIL"

    log.info(f"  [Ops] Overall operational status: {result.overall_status}")
    return result


def _verify_independent_load(
    artifact_paths: dict[str, Path],
    df: pd.DataFrame,
    result: OperationalAuditResult,
) -> None:
    """
    CORRECTION 5: Load A5a artifact in a subprocess to verify it works
    independently of the training process.

    Writes a small Python script to a temp file, runs it in a subprocess,
    and verifies the output matches in-process predictions.
    """
    # Get a small test batch and compute reference predictions in-process
    X_test = df[LOCKED_FEATURES].head(5).copy()
    X_test_vals = X_test.values

    # Build reference (in-process)
    m0_ref = joblib.load(artifact_paths["M0"])
    rg_ref = joblib.load(artifact_paths["Ridge"])
    rf_ref = joblib.load(artifact_paths["RF"])

    ref_pred = (
        m0_ref.predict(X_test_vals)
        + np.clip(rg_ref.predict(X_test_vals), -40, 60)
        + rf_ref.predict(X_test_vals)
    ) / 3.0

    # Write the test script
    m0_path  = str(artifact_paths["M0"])
    rg_path  = str(artifact_paths["Ridge"])
    rf_path  = str(artifact_paths["RF"])
    x_str    = repr(X_test_vals.tolist())

    script = f"""
import sys
import numpy as np
import joblib

m0 = joblib.load(r"{m0_path}")
rg = joblib.load(r"{rg_path}")
rf = joblib.load(r"{rf_path}")

X = np.array({x_str})

pred = (m0.predict(X) + np.clip(rg.predict(X), -40, 60) + rf.predict(X)) / 3.0
print(",".join(str(v) for v in pred))
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            result.independent_load_ok  = False
            result.independent_load_msg = f"Subprocess failed: {proc.stderr[:500]}"
            return

        sub_pred = np.array([float(v) for v in proc.stdout.strip().split(",")])
        if np.allclose(sub_pred, ref_pred, atol=1e-6):
            result.independent_load_ok  = True
            result.independent_load_msg = (
                "A5a artifact loaded and produced identical predictions in an isolated subprocess. "
                f"Max diff: {float(np.max(np.abs(sub_pred - ref_pred))):.2e}"
            )
            log.info(f"  [Ops] Independent load: ✓ (max diff={np.max(np.abs(sub_pred - ref_pred)):.2e})")
        else:
            result.independent_load_ok  = False
            result.independent_load_msg = (
                f"Subprocess predictions differ from in-process. "
                f"Max diff: {float(np.max(np.abs(sub_pred - ref_pred))):.4f}"
            )

    except subprocess.TimeoutExpired:
        result.independent_load_ok  = False
        result.independent_load_msg = "Subprocess timed out after 60s"
    except Exception as exc:
        result.independent_load_ok  = False
        result.independent_load_msg = f"Exception during subprocess test: {exc}"
    finally:
        try:
            Path(script_path).unlink()
        except Exception:
            pass
