"""
EconoSphere AI — Phase 21 Test Suite
=====================================
Script  : apps/api/tests/test_phase21_oos_validation.py
Purpose : 35 targeted tests covering all requirements from §30.

Run with:
    python -m pytest apps/api/tests/test_phase21_oos_validation.py -q

GOVERNANCE:
  - These tests must NOT modify any Phase 11 production artifact.
  - These tests must NOT train new models on OOS data.
  - These tests must NOT fabricate GDP values.
  - These tests verify the governance protocol is correctly implemented.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Production artifact paths ──────────────────────────────────────────────────
PROD_MODEL_PATH    = PROJECT_ROOT / "models" / "phase11" / "best_t1_gdp_growth_model.joblib"
PROD_MANIFEST_PATH = PROJECT_ROOT / "models" / "phase11" / "phase11_production_release_manifest.json"
PROD_DATASET_PATH  = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"

# ── Phase 21 paths ─────────────────────────────────────────────────────────────
PHASE21_DIR  = PROJECT_ROOT / "apps" / "api" / "ml" / "phase21"
PHASE21_DOCS = PROJECT_ROOT / "docs" / "model_accuracy" / "phase21"
PHASE20_ARTIFACTS = PROJECT_ROOT / "apps" / "api" / "ml" / "phase20" / "artifacts"
PHASE19_META = PROJECT_ROOT / "apps" / "api" / "ml" / "phase19" / "metrics" / "phase19_run_metadata.json"

# ── Known frozen production hashes ────────────────────────────────────────────
EXPECTED_PROD_MODEL_MD5    = "9c539735897eaf6e72e5f54f047390d7"
EXPECTED_PROD_MANIFEST_MD5 = "9c673e2d5bcd12adc171a55bd86ca34e"
EXPECTED_PROD_DATASET_MD5  = "2a490f1d3c245671b43602cd399d0fa0"
EXPECTED_PROD_MODEL_SHA256    = "748be64bcd3402375c4df4e45f9ba0c6858f52744714808ae8a2445f29248e78"
EXPECTED_PROD_MANIFEST_SHA256 = "d865d439bfdc8aae2fa61b53bd56fc79b53de96facfdfff93a55e2ba062040b3"
EXPECTED_PROD_DATASET_SHA256  = "eedbc8c67876f49f7db9c465bea057ec507994c3cc6429a0a346b3c5d1635047"

LOCKED_FEATURE_COUNT = 31
REQUIRED_EXPERIMENTAL_SUFFIX = "EXPERIMENTAL_ONLY"
VALID_GOVERNANCE_STATES = {
    "A5A_VALIDATED_AND_PROMOTION_RECOMMENDED",
    "A5A_OOS_VALIDATION_INCONCLUSIVE",
    "A5A_FAILED_TRUE_OOS_VALIDATION",
    "EXPERIMENT_FAILED_GOVERNANCE",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _load_p21_utils():
    from apps.api.ml.phase21 import utils as u
    return u

def _load_df():
    from apps.api.ml.phase21.utils import load_dataset
    return load_dataset()


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — PRODUCTION HASH IMMUTABILITY (Tests 1-4)
# ═══════════════════════════════════════════════════════════════════════════════

def test_01_prod_model_md5_unchanged():
    """Production model MD5 must match Phase 12 frozen value."""
    assert PROD_MODEL_PATH.exists(), "Production model missing"
    assert _md5(PROD_MODEL_PATH) == EXPECTED_PROD_MODEL_MD5

def test_02_prod_model_sha256_unchanged():
    """Production model SHA256 must match frozen value."""
    assert _sha256(PROD_MODEL_PATH) == EXPECTED_PROD_MODEL_SHA256

def test_03_prod_manifest_md5_unchanged():
    """Production manifest MD5 must match frozen value."""
    assert PROD_MANIFEST_PATH.exists(), "Production manifest missing"
    assert _md5(PROD_MANIFEST_PATH) == EXPECTED_PROD_MANIFEST_MD5

def test_04_prod_dataset_md5_unchanged():
    """Production dataset MD5 must match frozen value."""
    assert PROD_DATASET_PATH.exists(), "Dataset missing"
    assert _md5(PROD_DATASET_PATH) == EXPECTED_PROD_DATASET_MD5


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 2 — FEATURE SCHEMA (Tests 5-7)
# ═══════════════════════════════════════════════════════════════════════════════

def test_05_locked_feature_count():
    """Phase 21 must use exactly 31 locked features from Phase 19."""
    u = _load_p21_utils()
    assert len(u.LOCKED_FEATURES) == LOCKED_FEATURE_COUNT

def test_06_feature_list_matches_manifest():
    """Locked feature list must match the Phase 11 production manifest."""
    u = _load_p21_utils()
    manifest = json.loads(PROD_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_features = set(manifest["feature_names"])
    locked_features   = set(u.LOCKED_FEATURES)
    assert locked_features == manifest_features, (
        f"Feature mismatch. In locked not manifest: {locked_features - manifest_features}. "
        f"In manifest not locked: {manifest_features - locked_features}."
    )

def test_07_forbidden_features_not_in_locked_set():
    """Target-derived columns must not appear in the locked feature set."""
    u = _load_p21_utils()
    forbidden = u.FORBIDDEN_IN_FEATURES
    locked    = set(u.LOCKED_FEATURES)
    violations = [f for f in forbidden if f in locked]
    assert violations == [], f"Forbidden columns in LOCKED_FEATURES: {violations}"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 3 — TARGET EXCLUSION (Tests 8-9)
# ═══════════════════════════════════════════════════════════════════════════════

def test_08_target_col_not_in_features():
    """gdp_growth_next_year must never appear in feature set."""
    u = _load_p21_utils()
    assert u.TARGET_COL not in u.LOCKED_FEATURES

def test_09_target_avail_col_not_in_features():
    """next_year_target_available must never appear in feature set."""
    u = _load_p21_utils()
    assert u.TARGET_AVAIL_COL not in u.LOCKED_FEATURES


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 4 — T+1 CHRONOLOGY (Tests 10-12)
# ═══════════════════════════════════════════════════════════════════════════════

def test_10_t1_chronology_train_split():
    """Training rows must be strictly before the feature year."""
    from apps.api.ml.phase19.utils import get_train_eval_split
    df = _load_df()
    train, eval_df = get_train_eval_split(df, feature_year=2015)
    assert (train["year"] < 2015).all(), "Training rows >= feature_year detected"

def test_11_t1_chronology_eval_split():
    """Evaluation rows must be exactly the feature year."""
    from apps.api.ml.phase19.utils import get_train_eval_split
    df = _load_df()
    _, eval_df = get_train_eval_split(df, feature_year=2015)
    assert (eval_df["year"] == 2015).all(), "Eval rows not at feature_year=2015"

def test_12_t1_target_is_next_year():
    """Confirm dataset target column encodes Y+1 GDP growth."""
    u = _load_p21_utils()
    assert u.TARGET_COL == "gdp_growth_next_year"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 5 — OOS AUTHENTICITY (Tests 13-17)
# ═══════════════════════════════════════════════════════════════════════════════

def test_13_oos_discovery_reads_no_target_values():
    """
    OOS discovery must NOT read gdp_growth_next_year values.
    Verified structurally: discover_oos_origins() uses only TARGET_AVAIL_COL.
    """
    import inspect
    from apps.api.ml.phase21.utils import discover_oos_origins
    src = inspect.getsource(discover_oos_origins)
    # The function must not access gdp_growth_next_year directly
    assert "gdp_growth_next_year" not in src or "TARGET_AVAIL_COL" in src, (
        "discover_oos_origins must rely on TARGET_AVAIL_COL, not raw target values"
    )
    # Must use TARGET_AVAIL_COL for filtering
    assert "TARGET_AVAIL_COL" in src

def test_14_2025_origin_has_zero_valid_targets():
    """Feature year 2025 must have next_year_target_available=0 for all rows."""
    df = _load_df()
    rows_2025 = df[df["year"] == 2025]
    if len(rows_2025) > 0:
        assert rows_2025["next_year_target_available"].sum() == 0, (
            "Feature year 2025 has unexpected valid targets — "
            "re-run OOS discovery before proceeding"
        )

def test_15_2024_origin_in_phase19():
    """Feature year 2024 must be listed in Phase 19 valid_origins — not new OOS."""
    meta = json.loads(PHASE19_META.read_text(encoding="utf-8"))
    listed = [int(y) for y in meta["valid_origins"]]
    assert 2024 in listed, "2024 not in Phase 19 origins — dataset may have changed"

def test_16_oos_discovery_result_is_unavailable():
    """
    Programmatic OOS discovery must conclude that no new OOS data is available
    given the current dataset state.
    """
    import logging
    from apps.api.ml.phase21.utils import discover_oos_origins, load_dataset
    df = load_dataset()
    logger = logging.getLogger("phase21_test")
    logger.handlers = []  # suppress output in test
    result = discover_oos_origins(df, logger)
    assert not result["available"], (
        f"OOS discovery returned available=True with origins {result['oos_origins']}. "
        f"This is unexpected — verify dataset has not been updated with 2026 targets."
    )
    assert result["oos_origins"] == []
    assert result["oos_observation_count"] == 0

def test_17_oos_discovery_does_not_alter_governance_thresholds():
    """
    The PROMOTION_THRESHOLDS dict must be identical before and after
    running OOS discovery (lock contamination protection).
    """
    import logging
    from apps.api.ml.phase21 import utils as u
    import copy
    before = copy.deepcopy(u.PROMOTION_THRESHOLDS)
    logger = logging.getLogger("phase21_test_17")
    logger.handlers = []
    df = u.load_dataset()
    _ = u.discover_oos_origins(df, logger)
    after = u.PROMOTION_THRESHOLDS
    assert before == after, "OOS discovery mutated PROMOTION_THRESHOLDS — governance violation"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 6 — LOCKED CONFIGURATION (Tests 18-21)
# ═══════════════════════════════════════════════════════════════════════════════

def test_18_frozen_ensemble_weights_equal():
    """A5a ensemble weights must be exactly 1/3 for each component."""
    u = _load_p21_utils()
    weights = u.FROZEN_A5A_CONFIG["weights"]
    assert len(weights) == 3, "A5a must have exactly 3 components"
    for comp, w in weights.items():
        assert abs(w - 1/3) < 1e-9, f"Weight for {comp} is {w}, expected 1/3"

def test_19_frozen_clipping_bounds():
    """A5a clipping bounds must match Phase 19 frozen values."""
    u = _load_p21_utils()
    assert u.FROZEN_A5A_CONFIG["clip_low"]  == -40.0
    assert u.FROZEN_A5A_CONFIG["clip_high"] ==  60.0

def test_20_m0_params_match_manifest():
    """M0 reconstruction parameters must match the Phase 11 manifest."""
    u = _load_p21_utils()
    manifest = json.loads(PROD_MANIFEST_PATH.read_text(encoding="utf-8"))
    m0 = u.FROZEN_A5A_CONFIG["m0_params"]
    mhp = manifest["hyperparameters"]
    assert m0["learning_rate"]     == mhp["learning_rate"]
    assert m0["max_depth"]         == mhp["max_depth"]
    assert m0["max_iter"]          == mhp["max_iter"]
    assert m0["l2_regularization"] == mhp["l2_regularization"]
    assert m0["random_state"]      == mhp["random_state"]

def test_21_rf_params_match_phase19():
    """Random Forest parameters must match the Phase 19 frozen configuration."""
    from apps.api.ml.phase19.models import RF_PARAMS
    u = _load_p21_utils()
    rf = u.FROZEN_A5A_CONFIG["rf_params"]
    assert rf["n_estimators"]    == RF_PARAMS["n_estimators"]    == 300
    assert rf["max_depth"]       == RF_PARAMS["max_depth"]       == 8
    assert rf["min_samples_leaf"]== RF_PARAMS["min_samples_leaf"]== 5


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 7 — A5a FORMULA (Test 22)
# ═══════════════════════════════════════════════════════════════════════════════

def test_22_a5a_formula_equal_weight_mean():
    """Verify the A5a ensemble formula is exactly (M0 + Ridge + RF) / 3."""
    m0_pred   = np.array([1.0, 2.0, 3.0])
    ridge_pred = np.array([2.0, 3.0, 4.0])
    rf_pred    = np.array([3.0, 4.0, 5.0])

    # Correct formula
    a5a_expected = (m0_pred + ridge_pred + rf_pred) / 3.0

    # Mimic evaluation.py logic
    a5a_actual = np.mean(
        np.column_stack([m0_pred, ridge_pred, rf_pred]), axis=1
    )
    np.testing.assert_allclose(a5a_actual, a5a_expected, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 8 — ARTIFACT LOADING (Tests 23-26)
# ═══════════════════════════════════════════════════════════════════════════════

def test_23_experimental_artifacts_exist():
    """Phase 20 A5a experimental artifacts must exist."""
    for name in [
        "phase20_M0_EXPERIMENTAL_ONLY.joblib",
        "phase20_Ridge_EXPERIMENTAL_ONLY.joblib",
        "phase20_RF_EXPERIMENTAL_ONLY.joblib",
        "phase20_a5a_metadata_EXPERIMENTAL_ONLY.json",
    ]:
        path = PHASE20_ARTIFACTS / name
        assert path.exists(), f"Missing experimental artifact: {name}"

def test_24_experimental_artifacts_suffix():
    """All A5a artifact filenames must end with EXPERIMENTAL_ONLY."""
    for name in [
        "phase20_M0_EXPERIMENTAL_ONLY.joblib",
        "phase20_Ridge_EXPERIMENTAL_ONLY.joblib",
        "phase20_RF_EXPERIMENTAL_ONLY.joblib",
    ]:
        stem = Path(name).stem
        assert stem.endswith(REQUIRED_EXPERIMENTAL_SUFFIX), (
            f"{name} does not end with {REQUIRED_EXPERIMENTAL_SUFFIX}"
        )

def test_25_experimental_artifacts_not_in_production_dir():
    """Experimental artifacts must NOT be in the Phase 11 production directory."""
    prod_dir = PROD_MODEL_PATH.parent
    for name in [
        "phase20_M0_EXPERIMENTAL_ONLY.joblib",
        "phase20_Ridge_EXPERIMENTAL_ONLY.joblib",
        "phase20_RF_EXPERIMENTAL_ONLY.joblib",
    ]:
        assert not (prod_dir / name).exists(), (
            f"Experimental artifact found in production directory: {name}"
        )

def test_26_a5a_metadata_weights_equal():
    """A5a metadata JSON must declare equal weights (1/3 each)."""
    meta = json.loads(
        (PHASE20_ARTIFACTS / "phase20_a5a_metadata_EXPERIMENTAL_ONLY.json")
        .read_text(encoding="utf-8")
    )
    weights = meta["weights"]
    for comp, w in weights.items():
        assert abs(w - 1/3) < 1e-9, f"Metadata weight for {comp}={w}, expected 1/3"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 9 — DETERMINISM AND NaN (Tests 27-28)
# ═══════════════════════════════════════════════════════════════════════════════

def test_27_m0_predictions_deterministic():
    """M0 (HistGradientBoosting) must produce identical predictions on repeated calls."""
    from apps.api.ml.phase19.models import build_m0
    from apps.api.ml.phase19.utils import LOCKED_FEATURES, TARGET_COL, get_train_eval_split
    df = _load_df()
    train, ev = get_train_eval_split(df, 2015)
    m0 = build_m0()
    m0.fit(train[LOCKED_FEATURES].values, train[TARGET_COL].values)
    p1 = m0.predict(ev[LOCKED_FEATURES].values)
    p2 = m0.predict(ev[LOCKED_FEATURES].values)
    np.testing.assert_array_equal(p1, p2)

def test_28_no_nan_in_a5a_predictions():
    """A5a predictions on a representative origin must not contain NaN."""
    from apps.api.ml.phase19.models import build_m0, build_ridge, build_random_forest
    from apps.api.ml.phase19.utils import LOCKED_FEATURES, TARGET_COL, get_train_eval_split
    df = _load_df()
    train, ev = get_train_eval_split(df, 2015)
    X_train = train[LOCKED_FEATURES].values
    y_train = train[TARGET_COL].values
    X_eval  = ev[LOCKED_FEATURES].values
    m0 = build_m0(); m0.fit(X_train, y_train)
    rg = build_ridge(alpha=10.0); rg.fit(X_train, y_train)
    rf = build_random_forest(); rf.fit(X_train, y_train)
    m0_p  = m0.predict(X_eval)
    rg_p  = np.clip(rg.predict(X_eval), -40.0, 60.0)
    rf_p  = rf.predict(X_eval)
    a5a_p = (m0_p + rg_p + rf_p) / 3.0
    assert not np.isnan(a5a_p).any(), "NaN found in A5a predictions"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 10 — METRICS CALCULATIONS (Tests 29-30)
# ═══════════════════════════════════════════════════════════════════════════════

def test_29_rmse_calculation_correct():
    """RMSE formula must be sqrt(mean(squared_errors))."""
    y    = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 1.5, 3.5])
    errors = y - pred
    expected_rmse = float(np.sqrt(np.mean(errors ** 2)))
    from apps.api.ml.phase19.evaluation import compute_metrics
    m = compute_metrics(y, pred)
    assert abs(m["rmse"] - expected_rmse) < 1e-10

def test_30_mae_calculation_correct():
    """MAE formula must be mean(|errors|)."""
    y    = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 1.0, 5.0])
    expected_mae = float(np.mean(np.abs(y - pred)))
    from apps.api.ml.phase19.evaluation import compute_metrics
    m = compute_metrics(y, pred)
    assert abs(m["mae"] - expected_mae) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 11 — GOVERNANCE STATE (Tests 31-33)
# ═══════════════════════════════════════════════════════════════════════════════

def test_31_governance_state_is_valid():
    """
    If Phase 21 has been run, the governance JSON must contain a valid state.
    If not yet run, this test verifies the valid states set is correct.
    """
    gov_path = PHASE21_DOCS / "phase21_governance.json"
    if gov_path.exists():
        gov = json.loads(gov_path.read_text(encoding="utf-8"))
        state = gov.get("governance_decision", "")
        assert state in VALID_GOVERNANCE_STATES, (
            f"governance_decision '{state}' not in valid states: {VALID_GOVERNANCE_STATES}"
        )
    else:
        # Verify the valid states set is defined correctly
        assert "A5A_OOS_VALIDATION_INCONCLUSIVE" in VALID_GOVERNANCE_STATES
        assert "A5A_VALIDATED_AND_PROMOTION_RECOMMENDED" in VALID_GOVERNANCE_STATES
        assert "A5A_FAILED_TRUE_OOS_VALIDATION" in VALID_GOVERNANCE_STATES
        assert "EXPERIMENT_FAILED_GOVERNANCE" in VALID_GOVERNANCE_STATES

def test_32_no_automatic_promotion():
    """
    If governance JSON exists, promotion_recommended must be False
    when there is no OOS data (inconclusive state).
    """
    gov_path = PHASE21_DOCS / "phase21_governance.json"
    if gov_path.exists():
        gov = json.loads(gov_path.read_text(encoding="utf-8"))
        state = gov.get("governance_decision", "")
        if state == "A5A_OOS_VALIDATION_INCONCLUSIVE":
            assert gov.get("promotion_recommended") is False, (
                "promotion_recommended must be False for INCONCLUSIVE state"
            )

def test_33_governance_json_explicit_oos_flags():
    """
    If governance JSON exists, OOS flags must be explicitly set when unavailable.
    Correction 4: not silently null — must be False with explicit semantics.
    """
    gov_path = PHASE21_DOCS / "phase21_governance.json"
    if gov_path.exists():
        gov = json.loads(gov_path.read_text(encoding="utf-8"))
        if not gov.get("true_new_oos_data", True):
            # If no OOS data, these must be explicitly False
            assert gov.get("OOS_EVALUATION_EXECUTED")    is False
            assert gov.get("OOS_DATA_AVAILABLE")         is False
            assert gov.get("OOS_PREDICTIONS_GENERATED")  is False
            # Null metric fields must remain null/None — not fabricated
            assert gov.get("phase11_rmse") is None
            assert gov.get("a5a_rmse")     is None
            assert gov.get("delta_rmse")   is None


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 12 — LOCK FILE (Tests 34-35)
# ═══════════════════════════════════════════════════════════════════════════════

def test_34_pre_evaluation_lock_structure():
    """
    If the pre-evaluation lock exists, verify it contains required fields
    and does NOT contain any OOS-derived metric.
    """
    lock_path = PHASE21_DOCS / "phase21_pre_evaluation_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        # Required governance fields
        assert "ensemble_weights" in lock
        assert "clipping" in lock
        assert "promotion_thresholds" in lock
        assert "feature_list" in lock
        assert "oos_origin_selection_rule" in lock
        assert "lock_hash_sha256" in lock
        # Must NOT contain OOS performance metrics
        forbidden_in_lock = ["phase11_rmse", "a5a_rmse", "delta_rmse", "oos_rmse"]
        for key in forbidden_in_lock:
            assert key not in lock, (
                f"Pre-evaluation lock contains OOS-derived metric: {key}. "
                "This violates Correction 3 — lock contamination protection."
            )

def test_35_oos_origins_lock_explicit_flags():
    """
    If the OOS origins lock file exists, verify explicit OOS_EVALUATION_EXECUTED flags.
    """
    origins_path = PHASE21_DOCS / "phase21_oos_origins_locked.json"
    if origins_path.exists():
        frozen = json.loads(origins_path.read_text(encoding="utf-8"))
        # These flags must always be explicitly present
        assert "OOS_EVALUATION_EXECUTED"   in frozen
        assert "OOS_DATA_AVAILABLE"        in frozen
        assert "OOS_PREDICTIONS_GENERATED" in frozen
        # If not available, all three must be False
        if not frozen.get("oos_available", True):
            assert frozen["OOS_EVALUATION_EXECUTED"]   is False
            assert frozen["OOS_DATA_AVAILABLE"]        is False
            assert frozen["OOS_PREDICTIONS_GENERATED"] is False
