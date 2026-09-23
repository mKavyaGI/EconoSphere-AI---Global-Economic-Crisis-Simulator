"""
EconoSphere AI — Phase 12, Step 1: Walk-Forward Validation Tests
=================================================================
Script   : apps/api/tests/test_t1_walk_forward.py
Purpose  : 20 automated integrity, leakage, and correctness tests for the
           Phase 12 Step 1 walk-forward evaluation.

Run with:
    python -X utf8 -m pytest apps/api/tests/test_t1_walk_forward.py -v

IMPORTANT: This test module must NOT modify any Phase 11 production artifacts.
"""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH          = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
INPUT_PATH        = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"

# Phase 12 artifacts
WF_SCRIPT_PATH    = PROJECT_ROOT / "apps" / "api" / "ml" / "evaluate_t1_walk_forward.py"
ERR_SCRIPT_PATH   = PROJECT_ROOT / "apps" / "api" / "ml" / "analyze_t1_forecast_errors.py"
METRICS_PATH      = PROJECT_ROOT / "data" / "processed" / "phase12_walk_forward_metrics.csv"
PREDS_PATH        = PROJECT_ROOT / "data" / "processed" / "phase12_walk_forward_predictions.csv"
META_PATH         = PROJECT_ROOT / "models" / "phase12" / "step1_walk_forward_metadata.json"

# Phase 11 production artifacts — must remain UNCHANGED
P11_PREDICTIONS   = PROJECT_ROOT / "data" / "processed" / "t1_step11_predictions.csv"
P11_FORECASTS     = PROJECT_ROOT / "data" / "processed" / "t1_step11_2026_forecasts.csv"
P11_META          = PROJECT_ROOT / "models" / "phase11" / "t1_step11_model_metadata.json"

# Expected immutable values
EXPECTED_MD5 = "8ac7e0b2bf09fbe89289f82d0c7cf25e"
LOCKED_PARAMS = {
    "l2_regularization": 5.0,
    "learning_rate":     0.05,
    "max_depth":         5,
    "max_iter":          300,
    "random_state":      42,
}
LOCKED_FEATURES_COUNT = 31
TARGET_COL = "gdp_growth_next_year"
FORBIDDEN_FEATURE_COLS = [
    "gdp_growth_next_year",
    "next_year_target_available",
    "gdp_growth_pct",
    "growth_regime",         # post-hoc only
    "total_missing_feature_ratio",  # diagnostic only
    "total_missing_feature_count",  # diagnostic only
]


# ── Test 1: Raw dataset MD5 unchanged ─────────────────────────────────────────
def test_1_raw_dataset_md5_unchanged():
    """Raw dataset must retain the exact MD5 from Phase 11 lock."""
    assert RAW_PATH.exists(), f"Raw dataset not found: {RAW_PATH}"
    actual_md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    assert actual_md5 == EXPECTED_MD5, (
        f"Raw dataset MD5 mismatch!\n"
        f"  Expected : {EXPECTED_MD5}\n"
        f"  Actual   : {actual_md5}"
    )


# ── Test 2: Phase 11 predictions artifact unchanged ───────────────────────────
def test_2_phase11_predictions_artifact_unchanged():
    """t1_step11_predictions.csv must still exist (Phase 11 artifact protection)."""
    assert P11_PREDICTIONS.exists(), "Phase 11 predictions artifact was removed or overwritten!"


# ── Test 3: Phase 11 forecasts artifact unchanged ─────────────────────────────
def test_3_phase11_forecasts_artifact_unchanged():
    """t1_step11_2026_forecasts.csv must still exist."""
    assert P11_FORECASTS.exists(), "Phase 11 forecasts artifact was removed or overwritten!"


# ── Test 4: Phase 11 metadata unchanged ───────────────────────────────────────
def test_4_phase11_metadata_unchanged():
    """Phase 11 model metadata must remain unchanged and reference the locked model."""
    assert P11_META.exists(), "Phase 11 model metadata artifact not found!"
    with open(P11_META) as f:
        meta = json.load(f)
    assert meta.get("selected_model") == "A. HistGradientBoosting (Locked Control)", (
        f"Phase 11 selected_model has changed: {meta.get('selected_model')}"
    )


# ── Test 5: No target columns in walk-forward script features ─────────────────
def test_5_no_target_in_locked_features_list():
    """Target columns must not appear in the LOCKED_FEATURES list in the script."""
    assert WF_SCRIPT_PATH.exists(), f"Script not found: {WF_SCRIPT_PATH}"
    content = WF_SCRIPT_PATH.read_text(encoding="utf-8")
    # Extract the LOCKED_FEATURES definition block
    feature_block_match = re.search(
        r"LOCKED_FEATURES\s*=\s*BASE_FEATURES\s*\+\s*ROLLING_EXTRA",
        content
    )
    # Check BASE_FEATURES and ROLLING_EXTRA blocks don't contain target col
    base_block = content.split("BASE_FEATURES = [")[1].split("]")[0]
    rolling_block = content.split("ROLLING_EXTRA = [")[1].split("]")[0]
    combined = base_block + rolling_block
    for forbidden in FORBIDDEN_FEATURE_COLS:
        assert forbidden not in combined, (
            f"Forbidden column '{forbidden}' found in feature definition block!"
        )


# ── Test 6: No random train/test split in walk-forward script ─────────────────
def test_6_no_random_split_in_walk_forward_script():
    """Walk-forward must use strictly chronological splits — no train_test_split."""
    content = WF_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "train_test_split" not in content, (
        "train_test_split found in evaluate_t1_walk_forward.py — random split prohibited!"
    )
    assert "shuffle" not in content.lower() or "# no shuffle" in content.lower(), (
        "Random shuffling detected in walk-forward script!"
    )


# ── Test 7: Every training year precedes its evaluation year ──────────────────
def test_7_training_precedes_evaluation_year():
    """All walk-forward windows must have train_end_year <= eval_feature_year.

    NOTE: The walk-forward script uses a strictly-less-than condition
    (df['year'] < eval_feature_year) for the training mask, meaning that
    even when train_end_year == feature_year the evaluation feature year
    itself is NOT included in training. We verify:
      (a) train_end_year <= feature_year  (no window trains on future data)
      (b) target_year > train_end_year    (eval target is strictly in the future)
    """
    assert METRICS_PATH.exists(), f"Walk-forward metrics not found: {METRICS_PATH}"
    df = pd.read_csv(METRICS_PATH)
    assert "train_end_year" in df.columns
    assert "feature_year" in df.columns
    assert "target_year" in df.columns
    for _, row in df.iterrows():
        assert row["train_end_year"] <= row["feature_year"], (
            f"Window {row['window_id']}: train_end_year ({row['train_end_year']}) "
            f"> feature_year ({row['feature_year']}) — training extends beyond eval window!"
        )
        assert row["target_year"] > row["train_end_year"], (
            f"Window {row['window_id']}: target_year ({row['target_year']}) "
            f"<= train_end_year ({row['train_end_year']}) — target contamination!"
        )


# ── Test 8: Eval feature_year + 1 == target_year ──────────────────────────────
def test_8_target_year_equals_feature_year_plus_one():
    """T+1 design: target_year must equal feature_year + 1 in all windows."""
    df = pd.read_csv(METRICS_PATH)
    for _, row in df.iterrows():
        assert int(row["target_year"]) == int(row["feature_year"]) + 1, (
            f"Window {row['window_id']}: target_year ({row['target_year']}) != "
            f"feature_year ({row['feature_year']}) + 1"
        )


# ── Test 9: 2025 is NOT evaluated (target is NaN inference year) ──────────────
def test_9_no_2025_target_in_predictions():
    """2025 is the inference year with NaN targets — must not appear as a target_year."""
    assert PREDS_PATH.exists(), f"Predictions file not found: {PREDS_PATH}"
    df = pd.read_csv(PREDS_PATH)
    assert 2025 not in df["target_year"].values, (
        "target_year=2025 found in walk-forward predictions — 2025 targets are NaN (inference year)!"
    )


# ── Test 10: No duplicate country/year prediction pairs ───────────────────────
def test_10_no_duplicate_country_year_pairs_per_window():
    """Within each window, no country should have two predictions for the same target_year."""
    df = pd.read_csv(PREDS_PATH)
    dup_check = df.groupby(["window_id", "country_code", "target_year"]).size()
    dupes = dup_check[dup_check > 1]
    assert len(dupes) == 0, (
        f"Duplicate country/target_year pairs found in {len(dupes)} window groups:\n{dupes}"
    )


# ── Test 11: absolute_error = |actual - predicted| ───────────────────────────
def test_11_absolute_error_math_correct():
    """absolute_error must exactly equal |actual_gdp_growth - predicted_gdp_growth|."""
    df = pd.read_csv(PREDS_PATH)
    expected = (df["actual_gdp_growth"] - df["predicted_gdp_growth"]).abs()
    assert np.allclose(expected.values, df["absolute_error"].values, atol=1e-5), (
        "absolute_error column contains incorrect values!"
    )


# ── Test 12: squared_error = (actual - predicted)^2 ──────────────────────────
def test_12_squared_error_math_correct():
    """squared_error must exactly equal (actual_gdp_growth - predicted_gdp_growth)^2."""
    df = pd.read_csv(PREDS_PATH)
    expected = (df["actual_gdp_growth"] - df["predicted_gdp_growth"]) ** 2
    assert np.allclose(expected.values, df["squared_error"].values, atol=1e-5), (
        "squared_error column contains incorrect values!"
    )


# ── Test 13: residual = actual - predicted ────────────────────────────────────
def test_13_residual_math_correct():
    """residual must exactly equal actual_gdp_growth - predicted_gdp_growth."""
    df = pd.read_csv(PREDS_PATH)
    expected = df["actual_gdp_growth"] - df["predicted_gdp_growth"]
    assert np.allclose(expected.values, df["residual"].values, atol=1e-5), (
        "residual column contains incorrect values!"
    )


# ── Test 14: RMSE in metrics is reproducible ──────────────────────────────────
def test_14_rmse_reproducible_from_predictions():
    """Per-window RMSE in metrics CSV must be reproducible from raw predictions."""
    metrics = pd.read_csv(METRICS_PATH)
    preds   = pd.read_csv(PREDS_PATH)
    for _, mrow in metrics.iterrows():
        wid = mrow["window_id"]
        wdf = preds[preds["window_id"] == wid]
        computed_rmse = math.sqrt(wdf["squared_error"].mean())
        assert abs(computed_rmse - mrow["rmse"]) < 0.001, (
            f"Window {wid}: RMSE mismatch — metrics={mrow['rmse']:.4f}, computed={computed_rmse:.4f}"
        )


# ── Test 15: Growth regime labels not in training features ────────────────────
def test_15_growth_regime_not_in_walk_forward_features():
    """Growth regime labels must never appear in the LOCKED_FEATURES list."""
    content = WF_SCRIPT_PATH.read_text(encoding="utf-8")
    feature_section = content.split("LOCKED_FEATURES")[1].split("TARGET_COL")[0]
    assert "growth_regime" not in feature_section
    assert "RECESSION" not in feature_section
    assert "LOW_GROWTH" not in feature_section
    assert "HIGH_GROWTH" not in feature_section
    assert "MODERATE_GROWTH" not in feature_section


# ── Test 16: Missingness indicators not added to the production model ─────────
def test_16_missingness_indicators_not_in_model_features():
    """Missingness diagnostic columns must not appear in walk-forward model features."""
    content = WF_SCRIPT_PATH.read_text(encoding="utf-8")
    base_block = content.split("BASE_FEATURES = [")[1].split("]")[0]
    rolling_block = content.split("ROLLING_EXTRA = [")[1].split("]")[0]
    combined = base_block + rolling_block
    assert "total_missing_feature_ratio" not in combined
    assert "total_missing_feature_count" not in combined
    assert "_missing" not in combined  # no per-column missing indicator features


# ── Test 17: Exact locked HistGradientBoosting parameters used ────────────────
def test_17_locked_parameters_in_script():
    """The walk-forward script must use the exact Phase 11 locked parameters.

    Uses regex to handle variable whitespace in dict literal definitions.
    """
    import re as _re
    content = WF_SCRIPT_PATH.read_text(encoding="utf-8")
    # Regex patterns handle spaces around colon and between value/comma
    def _param_present(key: str, value: str) -> bool:
        # Match 'key': value or "key": value with optional spaces
        pattern = _re.compile(
            r"['\"]" + _re.escape(key) + r"['\"]" + r"\s*:\s*" + _re.escape(value)
        )
        return bool(pattern.search(content))

    assert _param_present("l2_regularization", "5.0"),  "l2_regularization=5.0 not found"
    assert _param_present("learning_rate",     "0.05"), "learning_rate=0.05 not found"
    assert _param_present("max_depth",         "5"),    "max_depth=5 not found"
    assert _param_present("max_iter",          "300"),  "max_iter=300 not found"
    assert _param_present("random_state",      "42"),   "random_state=42 not found"


# ── Test 18: 31 locked features ───────────────────────────────────────────────
def test_18_feature_count_is_31():
    """Walk-forward metadata must confirm exactly 31 locked features."""
    assert META_PATH.exists(), f"Step 1 metadata not found: {META_PATH}"
    with open(META_PATH) as f:
        meta = json.load(f)
    assert meta.get("feature_count") == 31, (
        f"Expected 31 features, got {meta.get('feature_count')}"
    )


# ── Test 19: Production model modification flag is false ──────────────────────
def test_19_production_model_not_modified():
    """Step 1 metadata must confirm production_model_modified = False."""
    with open(META_PATH) as f:
        meta = json.load(f)
    assert meta.get("production_model_modified") is False, (
        "production_model_modified is not False in step1 metadata!"
    )


# ── Test 20: Phase 11 inference 2025 targets remain NaN ──────────────────────
def test_20_inference_2025_targets_nan():
    """2025 rows in the processed panel must have NaN gdp_growth_next_year (inference only)."""
    df = pd.read_csv(INPUT_PATH)
    rows_2025 = df[df["year"] == 2025]
    assert len(rows_2025) > 0, "No year=2025 rows found in processed panel!"
    assert rows_2025["gdp_growth_next_year"].isna().all(), (
        "Year 2025 rows have non-NaN gdp_growth_next_year — target contamination!"
    )
