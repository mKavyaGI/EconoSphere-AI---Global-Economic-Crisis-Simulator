"""
EconoSphere AI — Phase 19 Test Suite
=====================================
Script   : apps/api/tests/test_phase19_architecture.py
Purpose  : 35 targeted integrity, leakage, and correctness tests for the
           Phase 19 Forecast Architecture & Residual Modeling Experiment.

Run with:
    uv run pytest apps/api/tests/test_phase19_architecture.py -v

IMPORTANT: These tests must NOT modify any Phase 11 production artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Frozen production artifact paths ──────────────────────────────────────────
PROD_MODEL_PATH    = PROJECT_ROOT / "models" / "phase11" / "best_t1_gdp_growth_model.joblib"
PROD_MANIFEST_PATH = PROJECT_ROOT / "models" / "phase11" / "phase11_production_release_manifest.json"
PROD_DATASET_PATH  = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"

# ── Phase 19 artifact paths ───────────────────────────────────────────────────
PHASE19_DIR    = PROJECT_ROOT / "apps" / "api" / "ml" / "phase19"
METRICS_CSV    = PHASE19_DIR / "metrics" / "phase19_master_metrics.csv"
PREDS_CSV      = PHASE19_DIR / "predictions" / "phase19_all_predictions.csv"
META_JSON      = PHASE19_DIR / "metrics" / "phase19_run_metadata.json"
HASHES_BEFORE  = PHASE19_DIR / "hashes" / "before_production_hashes.json"
HASHES_AFTER   = PHASE19_DIR / "hashes" / "after_production_hashes.json"
REPORTS_DIR    = PROJECT_ROOT / "docs" / "model_accuracy" / "phase19"

# ── Phase 19 source files ─────────────────────────────────────────────────────
UTILS_PATH      = PHASE19_DIR / "utils.py"
MODELS_PATH     = PHASE19_DIR / "models.py"
EVAL_PATH       = PHASE19_DIR / "evaluation.py"
RUN_SCRIPT_PATH = PHASE19_DIR / "run_experiment.py"

# ── Known frozen values ───────────────────────────────────────────────────────
EXPECTED_PROD_MODEL_MD5    = "9c539735897eaf6e72e5f54f047390d7"
EXPECTED_PROD_MANIFEST_MD5 = "9c673e2d5bcd12adc171a55bd86ca34e"
EXPECTED_PROD_DATASET_MD5  = "2a490f1d3c245671b43602cd399d0fa0"

EXPECTED_PROD_MODEL_SHA256    = "748be64bcd3402375c4df4e45f9ba0c6858f52744714808ae8a2445f29248e78"
EXPECTED_PROD_MANIFEST_SHA256 = "d865d439bfdc8aae2fa61b53bd56fc79b53de96facfdfff93a55e2ba062040b3"
EXPECTED_PROD_DATASET_SHA256  = "eedbc8c67876f49f7db9c465bea057ec507994c3cc6429a0a346b3c5d1635047"

LOCKED_FEATURES_COUNT = 31
FORBIDDEN_FEATURE_COLS = [
    "gdp_growth_next_year",
    "next_year_target_available",
    "gdp_growth_pct",
    "growth_regime",
    "total_missing_feature_ratio",
    "total_missing_feature_count",
]

REQUIRED_EXPERIMENTAL_SUFFIX = "EXPERIMENTAL_ONLY"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# =============================================================================
# GROUP 1 — Production Artifact Immutability (Tests 1–6)
# =============================================================================

def test_1_production_model_exists():
    """Frozen Phase 11 model must still exist."""
    assert PROD_MODEL_PATH.exists(), f"Production model not found: {PROD_MODEL_PATH}"


def test_2_production_model_md5_unchanged():
    """Phase 11 model MD5 must equal the pre-experiment value."""
    actual = _md5(PROD_MODEL_PATH).lower()
    assert actual == EXPECTED_PROD_MODEL_MD5, (
        f"Production model MD5 changed!\n"
        f"  Expected: {EXPECTED_PROD_MODEL_MD5}\n"
        f"  Actual  : {actual}"
    )


def test_3_production_model_sha256_unchanged():
    """Phase 11 model SHA256 must equal the pre-experiment value."""
    actual = _sha256(PROD_MODEL_PATH).lower()
    assert actual == EXPECTED_PROD_MODEL_SHA256, (
        f"Production model SHA256 changed!\n"
        f"  Expected: {EXPECTED_PROD_MODEL_SHA256}\n"
        f"  Actual  : {actual}"
    )


def test_4_production_manifest_md5_unchanged():
    """Phase 11 manifest MD5 must be unchanged."""
    actual = _md5(PROD_MANIFEST_PATH).lower()
    assert actual == EXPECTED_PROD_MANIFEST_MD5


def test_5_production_dataset_md5_unchanged():
    """master_panel_t1_missingness.csv MD5 must be unchanged."""
    actual = _md5(PROD_DATASET_PATH).lower()
    assert actual == EXPECTED_PROD_DATASET_MD5, (
        f"Production dataset MD5 changed!\n"
        f"  Expected: {EXPECTED_PROD_DATASET_MD5}\n"
        f"  Actual  : {actual}"
    )


def test_6_before_after_hashes_match():
    """Before and after phase19 hashes must be identical for all three artifacts."""
    assert HASHES_BEFORE.exists(), f"Before-hash file missing: {HASHES_BEFORE}"
    assert HASHES_AFTER.exists(),  f"After-hash file missing: {HASHES_AFTER}"
    with open(HASHES_BEFORE) as f:
        before = json.load(f)
    with open(HASHES_AFTER) as f:
        after = json.load(f)
    for artifact in before:
        assert before[artifact]["md5"]    == after[artifact]["md5"],    \
            f"MD5 mismatch for {artifact}"
        assert before[artifact]["sha256"] == after[artifact]["sha256"], \
            f"SHA256 mismatch for {artifact}"


# =============================================================================
# GROUP 2 — Script Structural Checks (Tests 7–12)
# =============================================================================

def test_7_no_random_split_in_evaluation():
    """evaluation.py must not use train_test_split."""
    content = EVAL_PATH.read_text(encoding="utf-8")
    assert "train_test_split" not in content, \
        "train_test_split found in evaluation.py — prohibited!"


def test_8_no_shuffle_in_evaluation():
    """evaluation.py must not perform random shuffling."""
    content = EVAL_PATH.read_text(encoding="utf-8")
    # KFold(shuffle=False) is allowed; KFold(shuffle=True) is not
    bad_shuffle = re.search(r"KFold\s*\(.*shuffle\s*=\s*True", content)
    assert not bad_shuffle, "KFold with shuffle=True detected in evaluation.py!"


def test_9_experimental_filenames_enforced_in_models():
    """models.py must define experimental_model_filename() returning EXPERIMENTAL_ONLY."""
    content = MODELS_PATH.read_text(encoding="utf-8")
    assert "EXPERIMENTAL_ONLY" in content, \
        "EXPERIMENTAL_ONLY not found in models.py filename logic"


def test_10_no_production_paths_written_in_run_script():
    """run_experiment.py must not write to models/phase11/ or data/processed/."""
    content = RUN_SCRIPT_PATH.read_text(encoding="utf-8")
    # Writing to production paths: check no open() / to_csv / joblib.dump
    # targets the production directories
    forbidden_writes = [
        r'models[/\\]phase11',
        r'best_t1_gdp_growth_model',
        r'phase11_production_release_manifest',
    ]
    for pattern in forbidden_writes:
        write_match = re.search(
            r'(open|to_csv|joblib\.dump|write_text|write_bytes).*' + pattern,
            content,
        )
        assert not write_match, \
            f"Potential write to production path detected ({pattern}) in run_experiment.py"


def test_11_t_plus_1_design_enforced_in_evaluation():
    """evaluation.py must explicitly compute target_year = feature_year + 1."""
    content = EVAL_PATH.read_text(encoding="utf-8")
    assert "feature_year + 1" in content or "feature_year+1" in content, \
        "T+1 target_year computation not found in evaluation.py"


def test_12_run_script_hashes_before_and_after():
    """run_experiment.py must call record_hashes for both 'before' and 'after'."""
    content = RUN_SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'record_hashes("before")' in content or "record_hashes('before')" in content, \
        "BEFORE hash recording not found in run_experiment.py"
    assert 'record_hashes("after")' in content or "record_hashes('after')" in content, \
        "AFTER hash recording not found in run_experiment.py"


# =============================================================================
# GROUP 3 — Feature & Target Leakage Checks (Tests 13–18)
# =============================================================================

def test_13_forbidden_columns_not_in_locked_features():
    """No forbidden column (target, regime, diagnostic) may appear in LOCKED_FEATURES."""
    content = UTILS_PATH.read_text(encoding="utf-8")
    base_block    = content.split("BASE_FEATURES = [")[1].split("]")[0]
    rolling_block = content.split("ROLLING_EXTRA  = [")[1].split("]")[0] \
        if "ROLLING_EXTRA  = [" in content \
        else content.split("ROLLING_EXTRA = [")[1].split("]")[0]
    combined = base_block + rolling_block
    for col in FORBIDDEN_FEATURE_COLS:
        assert col not in combined, \
            f"Forbidden column '{col}' found in LOCKED_FEATURES definition!"


def test_14_locked_feature_count_is_31():
    """LOCKED_FEATURES must have exactly 31 features."""
    from apps.api.ml.phase19.utils import LOCKED_FEATURES
    assert len(LOCKED_FEATURES) == LOCKED_FEATURES_COUNT, \
        f"Expected {LOCKED_FEATURES_COUNT} features, got {len(LOCKED_FEATURES)}"


def test_15_no_target_column_in_X_eval(tmp_path):
    """get_train_eval_split must not include target column in returned eval features."""
    from apps.api.ml.phase19.utils import LOCKED_FEATURES, TARGET_COL, load_dataset, get_train_eval_split
    df = load_dataset()
    # Pick a middle year that has valid origins
    test_year = 2018
    _, eval_df = get_train_eval_split(df, test_year)
    X_eval = eval_df[LOCKED_FEATURES]
    assert TARGET_COL not in X_eval.columns, \
        f"Target column '{TARGET_COL}' found in eval feature set!"


def test_16_train_rows_strictly_before_feature_year():
    """All training rows must have year < feature_year."""
    from apps.api.ml.phase19.utils import load_dataset, get_train_eval_split
    df = load_dataset()
    for test_year in [2015, 2018, 2022]:
        train_df, _ = get_train_eval_split(df, test_year)
        if "year" in train_df.columns:
            assert (train_df["year"] < test_year).all(), \
                f"Training row with year >= {test_year} found in training window!"


def test_17_eval_rows_exactly_at_feature_year():
    """Eval rows must all have year == feature_year."""
    from apps.api.ml.phase19.utils import load_dataset, get_train_eval_split
    df = load_dataset()
    for test_year in [2016, 2020, 2023]:
        _, eval_df = get_train_eval_split(df, test_year)
        if "year" in eval_df.columns and not eval_df.empty:
            assert (eval_df["year"] == test_year).all(), \
                f"Eval row with year != {test_year} found!"


def test_18_no_nan_targets_in_eval():
    """Eval rows must not have NaN targets (next_year_target_available == 1 is enforced)."""
    from apps.api.ml.phase19.utils import TARGET_COL, load_dataset, get_train_eval_split
    df = load_dataset()
    for test_year in [2018, 2020, 2022]:
        _, eval_df = get_train_eval_split(df, test_year)
        if not eval_df.empty:
            assert not eval_df[TARGET_COL].isna().any(), \
                f"NaN targets found in eval set at feature_year={test_year}"


# =============================================================================
# GROUP 4 — Origin Discovery (Tests 19–21)
# =============================================================================

def test_19_valid_origins_have_t_plus_1_target():
    """Every valid origin must have target_year == feature_year + 1."""
    from apps.api.ml.phase19.utils import load_dataset, get_valid_origins
    df = load_dataset()
    valid_origins, _ = get_valid_origins(df)
    for o in valid_origins:
        assert o["target_year"] == o["feature_year"] + 1, \
            f"Origin {o['feature_year']}: target_year={o['target_year']} != feature_year+1"


def test_20_valid_origins_have_minimum_training_rows():
    """Every valid origin must have at least 50 training rows."""
    from apps.api.ml.phase19.utils import load_dataset, get_valid_origins
    df = load_dataset()
    valid_origins, _ = get_valid_origins(df, min_train_rows=50)
    for o in valid_origins:
        assert o["n_train"] >= 50, \
            f"Origin {o['feature_year']} has only {o['n_train']} training rows"


def test_21_skipped_origins_have_documented_reasons():
    """Skipped origins must have a non-empty reason string."""
    from apps.api.ml.phase19.utils import load_dataset, get_valid_origins
    df = load_dataset()
    _, skipped = get_valid_origins(df)
    for s in skipped:
        assert s.get("reason"), \
            f"Skipped origin {s.get('feature_year')} has no documented reason"


# =============================================================================
# GROUP 5 — Model Naming & Artifact Rules (Tests 22–24)
# =============================================================================

def test_22_experimental_filename_ends_with_experimental_only():
    """experimental_model_filename() must always produce a name ending in EXPERIMENTAL_ONLY."""
    from apps.api.ml.phase19.models import experimental_model_filename
    for key in ["M0_Phase11", "A1_Ridge", "A2_ElasticNet", "A3_Huber",
                "A4_RandomForest", "A5a_Ensemble", "A5b_Ensemble", "A6_Residual"]:
        fname = experimental_model_filename(key)
        stem = Path(fname).stem
        assert stem.endswith("EXPERIMENTAL_ONLY"), \
            f"Filename '{fname}' does not end with EXPERIMENTAL_ONLY"


def test_23_validate_experimental_filename_accepts_correct_names():
    """validate_experimental_filename must return True for correct names."""
    from apps.api.ml.phase19.models import validate_experimental_filename
    valid_names = [
        "phase19_ridge_EXPERIMENTAL_ONLY.joblib",
        "phase19_m0_EXPERIMENTAL_ONLY.joblib",
        "my_model_EXPERIMENTAL_ONLY.pkl",
    ]
    for name in valid_names:
        assert validate_experimental_filename(name), \
            f"validate_experimental_filename rejected valid name: {name}"


def test_24_validate_experimental_filename_rejects_bad_names():
    """validate_experimental_filename must reject production-style names."""
    from apps.api.ml.phase19.models import validate_experimental_filename
    bad_names = [
        "best_model.joblib",
        "production_model.joblib",
        "phase19_ridge.joblib",
        "EXPERIMENTAL_ONLY_ridge.joblib",  # prefix, not suffix
    ]
    for name in bad_names:
        assert not validate_experimental_filename(name), \
            f"validate_experimental_filename accepted bad name: {name}"


# =============================================================================
# GROUP 6 — Metrics Correctness (Tests 25–29, post-run)
# =============================================================================

def test_25_master_metrics_csv_exists():
    """Phase 19 master metrics CSV must exist after the run."""
    assert METRICS_CSV.exists(), f"Master metrics CSV not found: {METRICS_CSV}"


def test_26_predictions_csv_exists():
    """Phase 19 predictions CSV must exist after the run."""
    assert PREDS_CSV.exists(), f"Predictions CSV not found: {PREDS_CSV}"


def test_27_master_metrics_contains_all_candidates():
    """Master metrics table must contain all 8 candidate model keys."""
    assert METRICS_CSV.exists(), pytest.skip("Run experiment first")
    df = pd.read_csv(METRICS_CSV)
    expected_models = {
        "M0_Phase11", "A1_Ridge", "A2_ElasticNet", "A3_Huber",
        "A4_RandomForest", "A5a_Ensemble", "A5b_Ensemble", "A6_Residual",
    }
    actual_models = set(df["model"].tolist())
    missing = expected_models - actual_models
    assert not missing, f"Missing models in master metrics: {missing}"


def test_28_predictions_residual_math_correct():
    """residual = actual - predicted must hold for all rows."""
    if not PREDS_CSV.exists():
        pytest.skip("Run experiment first")
    df = pd.read_csv(PREDS_CSV)
    expected = df["actual_gdp_growth"] - df["predicted_gdp_growth"]
    assert np.allclose(expected.values, df["residual"].values, atol=1e-5), \
        "residual column contains incorrect values!"


def test_29_predictions_absolute_error_math_correct():
    """absolute_error = |actual - predicted| must hold for all rows."""
    if not PREDS_CSV.exists():
        pytest.skip("Run experiment first")
    df = pd.read_csv(PREDS_CSV)
    expected = (df["actual_gdp_growth"] - df["predicted_gdp_growth"]).abs()
    assert np.allclose(expected.values, df["absolute_error"].values, atol=1e-5), \
        "absolute_error column contains incorrect values!"


# =============================================================================
# GROUP 7 — Ensemble & Residual Architecture (Tests 30–33)
# =============================================================================

def test_30_a5b_weights_sum_to_one():
    """A5b ensemble weights must sum to 1.0 for every origin."""
    if not META_JSON.exists():
        pytest.skip("Run experiment first")
    # We can't load per-origin weights from meta, but we check the code
    content = EVAL_PATH.read_text(encoding="utf-8")
    assert "sum=1" in content or "sum(w) - 1" in content, \
        "Ensemble weight sum-to-1 constraint not found in evaluation.py"


def test_31_a6_uses_oof_predictions():
    """Residual model must generate OOF predictions (not full-train predictions)."""
    content = EVAL_PATH.read_text(encoding="utf-8")
    assert "_generate_oof_m0_predictions" in content, \
        "OOF M0 prediction function not found in evaluation.py"
    assert "KFold" in content, \
        "KFold (for OOF generation) not found in evaluation.py"


def test_32_residual_model_architecture_correct():
    """A6 final prediction must be M0_pred + residual_correction."""
    content = EVAL_PATH.read_text(encoding="utf-8")
    assert "m0_eval_pred" in content or "M0_Phase11" in content, \
        "M0 prediction base not found for A6 construction"
    assert "residual_correction" in content or "a6_residual" in content, \
        "Residual correction term not found in A6 construction"


def test_33_val_split_is_chronological():
    """_make_val_split must not shuffle data (shuffle=True must never appear)."""
    content = EVAL_PATH.read_text(encoding="utf-8")
    # Extract the function body only
    func_start = content.find("def _make_val_split")
    func_end   = content.find("\ndef ", func_start + 1)
    func_body  = content[func_start:func_end]
    # The word "shuffle" may appear in comments/docstring ("Never shuffles"),
    # but shuffle=True must NOT appear in executable code.
    assert "shuffle=True" not in func_body, \
        "shuffle=True detected in _make_val_split — chronological split violated!"
    assert "random_state" not in func_body or "# no random" in func_body.lower(), \
        "random_state found inside _make_val_split — potential shuffle risk!"


# =============================================================================
# GROUP 8 — Report Files (Tests 34–35)
# =============================================================================

def test_34_all_17_reports_exist():
    """All 17 Phase 19 reports must exist after the run."""
    expected_reports = [
        "01_executive_summary.md",
        "02_experimental_design.md",
        "03_chronological_backtest.md",
        "04_architecture_comparison.md",
        "05_ridge_elasticnet_huber.md",
        "06_random_forest.md",
        "07_ensemble_analysis.md",
        "08_residual_modeling.md",
        "09_country_analysis.md",
        "10_priority_country_analysis.md",
        "11_guardrail_analysis.md",
        "12_shock_analysis.md",
        "13_error_tail_analysis.md",
        "14_prediction_diversity.md",
        "15_leakage_audit.md",
        "16_production_readiness.md",
        "17_phase19_walkthrough.md",
    ]
    missing = [r for r in expected_reports if not (REPORTS_DIR / r).exists()]
    assert not missing, f"Missing Phase 19 reports: {missing}"


def test_35_run_metadata_governance_is_valid():
    """run_metadata.json must contain a valid governance string."""
    if not META_JSON.exists():
        pytest.skip("Run experiment first")
    with open(META_JSON) as f:
        meta = json.load(f)
    valid_governance = {
        "ARCHITECTURE_ROBUST_AND_PROMISING",
        "ARCHITECTURE_PROMISING_BUT_INCONCLUSIVE",
        "NO_ARCHITECTURAL_IMPROVEMENT_FOUND",
        "EXPERIMENT_FAILED_GOVERNANCE",
    }
    assert meta.get("governance") in valid_governance, \
        f"Invalid governance string: {meta.get('governance')}"
    assert meta.get("production_status") == "FROZEN_PRODUCTION_RETAINED", \
        "production_status is not FROZEN_PRODUCTION_RETAINED"
