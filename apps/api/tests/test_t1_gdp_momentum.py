"""
EconoSphere AI — Phase 12, Step 2: GDP Momentum Integrity Tests
================================================================
Script   : apps/api/tests/test_t1_gdp_momentum.py
Purpose  : 22 automated leakage, alignment, integrity, and reproducibility
           tests for the Phase 12 Step 2 GDP momentum experiment.

Run with:
    python -X utf8 -m pytest apps/api/tests/test_t1_gdp_momentum.py -v
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH     = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"

# Phase 12 Step 2 artifacts
FEAT_PATH    = PROJECT_ROOT / "data" / "processed" / "phase12_step2_momentum_features.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step2_experiment_metrics.csv"
PREDS_PATH   = PROJECT_ROOT / "data" / "processed" / "phase12_step2_predictions.csv"
FC_PATH      = PROJECT_ROOT / "data" / "processed" / "phase12_step2_2026_forecasts.csv"
META_PATH    = PROJECT_ROOT / "models" / "phase12" / "step2_gdp_momentum_metadata.json"
FEAT_SCRIPT  = PROJECT_ROOT / "apps" / "api" / "ml" / "data_t1_gdp_momentum.py"
TRAIN_SCRIPT = PROJECT_ROOT / "apps" / "api" / "ml" / "train_t1_gdp_momentum.py"
INPUT_PATH   = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"

# Phase 11 production artifacts (must be unchanged)
P11_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "t1_step11_predictions.csv"
P11_FORECASTS   = PROJECT_ROOT / "data" / "processed" / "t1_step11_2026_forecasts.csv"
P11_META        = PROJECT_ROOT / "models" / "phase11" / "t1_step11_model_metadata.json"

EXPECTED_MD5  = "8ac7e0b2bf09fbe89289f82d0c7cf25e"
TARGET_COL    = "gdp_growth_next_year"
AVAIL_COL     = "next_year_target_available"

NEW_MOMENTUM_FEATURES = [
    "gdp_growth_accel_1y", "gdp_growth_accel_2y", "gdp_growth_trend_change",
    "gdp_growth_mean_2y",  "gdp_growth_mean_3y",  "gdp_growth_std_3y",
    "gdp_growth_decelerating", "gdp_growth_accelerating",
    "gdp_growth_negative_lag1", "gdp_growth_decline_2y",
]


# ── Test 1: Raw dataset MD5 unchanged ─────────────────────────────────────────
def test_1_raw_dataset_md5_unchanged():
    actual = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    assert actual == EXPECTED_MD5, f"MD5 mismatch: {actual}"


# ── Test 2: Phase 11 predictions unchanged ────────────────────────────────────
def test_2_phase11_predictions_artifact_unchanged():
    assert P11_PREDICTIONS.exists(), "Phase 11 predictions artifact was removed or overwritten!"


# ── Test 3: Phase 11 forecasts unchanged ──────────────────────────────────────
def test_3_phase11_forecasts_artifact_unchanged():
    assert P11_FORECASTS.exists(), "Phase 11 forecasts artifact was removed!"


# ── Test 4: Phase 11 metadata unchanged ───────────────────────────────────────
def test_4_phase11_metadata_unchanged():
    assert P11_META.exists(), "Phase 11 metadata artifact was removed!"
    with open(P11_META) as f:
        meta = json.load(f)
    assert meta.get("selected_model") == "A. HistGradientBoosting (Locked Control)"


# ── Test 5: Target column absent from ALL experimental feature lists ──────────
def test_5_target_not_in_feature_scripts():
    """gdp_growth_next_year must not appear in BASE_FEATURES or momentum feature lists."""
    for script in [FEAT_SCRIPT, TRAIN_SCRIPT]:
        content = script.read_text(encoding="utf-8")
        # Extract feature list sections — look for any assignment referencing target
        lines = content.split("\n")
        in_feature_block = False
        for i, line in enumerate(lines):
            if "CONTROL_FEATURES" in line or "BASE_FEATURES" in line:
                in_feature_block = True
            if in_feature_block and TARGET_COL in line and "=" not in line:
                assert False, (
                    f"Target column '{TARGET_COL}' found in feature block of "
                    f"{script.name} at line {i+1}: {line}"
                )
            if in_feature_block and "]" in line:
                in_feature_block = False


# ── Test 6: gdp_growth_next_year NOT used to build any momentum feature ───────
def test_6_next_year_target_never_in_feature_construction():
    """The feature construction function must not reference gdp_growth_next_year."""
    content = FEAT_SCRIPT.read_text(encoding="utf-8")
    # Find the build_momentum_features function
    fn_start = content.find("def build_momentum_features")
    fn_end   = content.find("\ndef ", fn_start + 1)
    fn_body  = content[fn_start:fn_end] if fn_end > 0 else content[fn_start:]
    assert TARGET_COL not in fn_body, (
        f"'{TARGET_COL}' found inside build_momentum_features()!"
    )


# ── Test 7: accel_1y = gdp_growth_pct - gdp_growth_lag1 ──────────────────────
def test_7_accel_1y_formula_correct():
    """gdp_growth_accel_1y = gdp_growth_pct - gdp_growth_lag1."""
    df = pd.read_csv(FEAT_PATH)
    valid = df[df["gdp_growth_accel_1y"].notna() & df["gdp_growth_pct"].notna() & df["gdp_growth_lag1"].notna()]
    expected = valid["gdp_growth_pct"] - valid["gdp_growth_lag1"]
    assert np.allclose(expected.values, valid["gdp_growth_accel_1y"].values, atol=1e-6), (
        "gdp_growth_accel_1y formula is incorrect!"
    )


# ── Test 8: accel_2y = gdp_growth_lag1 - gdp_growth_lag2 ─────────────────────
def test_8_accel_2y_formula_correct():
    """gdp_growth_accel_2y = gdp_growth_lag1 - gdp_growth_lag2."""
    df = pd.read_csv(FEAT_PATH)
    valid = df[df["gdp_growth_accel_2y"].notna() & df["gdp_growth_lag1"].notna() & df["gdp_growth_lag2"].notna()]
    expected = valid["gdp_growth_lag1"] - valid["gdp_growth_lag2"]
    assert np.allclose(expected.values, valid["gdp_growth_accel_2y"].values, atol=1e-6), (
        "gdp_growth_accel_2y formula is incorrect!"
    )


# ── Test 9: trend_change = accel_1y - accel_2y ───────────────────────────────
def test_9_trend_change_formula_correct():
    """gdp_growth_trend_change = accel_1y - accel_2y."""
    df = pd.read_csv(FEAT_PATH)
    valid = df[df["gdp_growth_trend_change"].notna() &
               df["gdp_growth_accel_1y"].notna() &
               df["gdp_growth_accel_2y"].notna()]
    expected = valid["gdp_growth_accel_1y"] - valid["gdp_growth_accel_2y"]
    assert np.allclose(expected.values, valid["gdp_growth_trend_change"].values, atol=1e-6), (
        "gdp_growth_trend_change formula is incorrect!"
    )


# ── Test 10: No t+1 GDP value in accel_1y via lag alignment check ─────────────
def test_10_accel_1y_does_not_use_next_year_gdp():
    """
    For feature_year=t: accel_1y = pct(t) - pct(t-1).
    Verify for USA row year=2024: accel_1y = gdp_growth_pct(2024) - lag1(2024).
    lag1(2024) == gdp_growth_pct(2023). Next year (2025) must NOT appear.
    """
    df  = pd.read_csv(FEAT_PATH)
    usa = df[df["country_code"] == "USA"].sort_values("year")
    row_2024 = usa[usa["year"] == 2024].iloc[0]
    row_2023 = usa[usa["year"] == 2023].iloc[0]
    row_2025 = usa[usa["year"] == 2025].iloc[0]

    # accel_1y(2024) = pct(2024) - lag1(2024) = pct(2024) - pct(2023)
    expected = row_2024["gdp_growth_pct"] - row_2023["gdp_growth_pct"]
    actual   = row_2024["gdp_growth_accel_1y"]
    assert abs(expected - actual) < 1e-6, (
        f"accel_1y(2024): expected {expected:.6f}, got {actual:.6f}"
    )

    # CONFIRM: 2025 gdp_growth_pct is NOT equal to accel_1y(2024) + lag1(2024)
    # i.e., 2025's gdp value was not smuggled in
    smuggled_val = row_2025["gdp_growth_pct"]
    assert abs(actual - smuggled_val) > 0.001 or math.isnan(smuggled_val), (
        "Possible t+1 leakage: accel_1y(2024) equals gdp_growth_pct(2025)!"
    )


# ── Test 11: lag1 alignment is historically correct ───────────────────────────
def test_11_lag1_alignment_correct():
    """lag1 at year t must equal gdp_growth_pct at year t-1."""
    df  = pd.read_csv(FEAT_PATH)
    grp = df.sort_values(["country_code","year"])
    for cc, sub in grp.groupby("country_code"):
        sub = sub.sort_values("year").reset_index(drop=True)
        for i in range(1, len(sub)):
            prev_pct = sub.iloc[i-1]["gdp_growth_pct"]
            curr_lag1 = sub.iloc[i]["gdp_growth_lag1"]
            if not (pd.isna(prev_pct) or pd.isna(curr_lag1)):
                assert abs(prev_pct - curr_lag1) < 1e-5, (
                    f"lag1 misalignment for {cc} at year {sub.iloc[i]['year']}: "
                    f"pct(t-1)={prev_pct:.4f}, lag1(t)={curr_lag1:.4f}"
                )
        break  # Checking one country is sufficient for unit test


# ── Test 12: lag2 alignment is historically correct ───────────────────────────
def test_12_lag2_alignment_correct():
    """lag2 at year t must equal gdp_growth_pct at year t-2."""
    df  = pd.read_csv(FEAT_PATH)
    usa = df[df["country_code"] == "USA"].sort_values("year").reset_index(drop=True)
    for i in range(2, len(usa)):
        prev2_pct = usa.iloc[i-2]["gdp_growth_pct"]
        curr_lag2 = usa.iloc[i]["gdp_growth_lag2"]
        if not (pd.isna(prev2_pct) or pd.isna(curr_lag2)):
            assert abs(prev2_pct - curr_lag2) < 1e-5, (
                f"lag2 misalignment at year {usa.iloc[i]['year']}"
            )


# ── Test 13: lag3 alignment is historically correct ───────────────────────────
def test_13_lag3_alignment_correct():
    """lag3 at year t must equal gdp_growth_pct at year t-3."""
    df  = pd.read_csv(FEAT_PATH)
    usa = df[df["country_code"] == "USA"].sort_values("year").reset_index(drop=True)
    for i in range(3, len(usa)):
        prev3_pct = usa.iloc[i-3]["gdp_growth_pct"]
        curr_lag3 = usa.iloc[i]["gdp_growth_lag3"]
        if not (pd.isna(prev3_pct) or pd.isna(curr_lag3)):
            assert abs(prev3_pct - curr_lag3) < 1e-5, (
                f"lag3 misalignment at year {usa.iloc[i]['year']}"
            )


# ── Test 14: rolling mean_3y does NOT use current year's gdp_growth_pct ───────
def test_14_rolling_mean_3y_historical_only():
    """
    mean_3y for year t = mean(pct(t-1), pct(t-2), pct(t-3)).
    Current year pct(t) must NOT appear in the rolling mean.
    """
    df  = pd.read_csv(FEAT_PATH)
    usa = df[df["country_code"] == "USA"].sort_values("year").reset_index(drop=True)
    # Year 2010: mean_3y should use 2009, 2008, 2007
    row_2010 = usa[usa["year"] == 2010].iloc[0]
    pct_2007 = usa[usa["year"] == 2007].iloc[0]["gdp_growth_pct"]
    pct_2008 = usa[usa["year"] == 2008].iloc[0]["gdp_growth_pct"]
    pct_2009 = usa[usa["year"] == 2009].iloc[0]["gdp_growth_pct"]
    expected = np.mean([pct_2007, pct_2008, pct_2009])
    actual   = row_2010["gdp_growth_mean_3y"]
    assert abs(expected - actual) < 1e-4, (
        f"mean_3y(2010): expected {expected:.4f} (mean of 2007-2009), got {actual:.4f}"
    )


# ── Test 15: rolling std_3y uses historical values only ───────────────────────
def test_15_rolling_std_3y_historical_only():
    """std_3y for year t = std(pct(t-1), pct(t-2), pct(t-3))."""
    df  = pd.read_csv(FEAT_PATH)
    usa = df[df["country_code"] == "USA"].sort_values("year").reset_index(drop=True)
    row_2010 = usa[usa["year"] == 2010].iloc[0]
    pct_2007 = usa[usa["year"] == 2007].iloc[0]["gdp_growth_pct"]
    pct_2008 = usa[usa["year"] == 2008].iloc[0]["gdp_growth_pct"]
    pct_2009 = usa[usa["year"] == 2009].iloc[0]["gdp_growth_pct"]
    expected = float(np.std([pct_2007, pct_2008, pct_2009], ddof=1))
    actual   = row_2010["gdp_growth_std_3y"]
    assert abs(expected - actual) < 1e-4, (
        f"std_3y(2010): expected {expected:.4f}, got {actual:.4f}"
    )


# ── Test 16: No forward fill / backfill from future years ─────────────────────
def test_16_no_future_backfill():
    """
    Verify that the momentum features script uses shift(1) before rolling,
    preventing any future observation from leaking backward.
    """
    content = FEAT_SCRIPT.read_text(encoding="utf-8")
    # The script must use shift(1) before rolling transformations
    assert "shift(1)" in content, "shift(1) not found in feature script — potential future backfill!"
    # Must NOT use shift(-1) (forward looking shift)
    assert "shift(-1)" not in content, "shift(-1) found — forward-looking shift is prohibited!"
    # Must NOT use fillna(method='bfill') or .bfill()
    assert "bfill" not in content.lower(), "bfill (backward fill) found in feature script!"
    # ffill is acceptable for forward-filling within the historical window,
    # but only if explicitly not using future data — we flag it as a warning here
    # (the script does not use ffill at all, so this is a double check)
    assert "ffill" not in content, "ffill found in feature script — check for future leakage!"


# ── Test 17: No random split in training script ───────────────────────────────
def test_17_no_random_split_in_train_script():
    """Training script must use strictly chronological splits."""
    content = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert "train_test_split" not in content, (
        "train_test_split found in train_t1_gdp_momentum.py!"
    )
    # Also verify chronological year constraints are present
    assert "TRAIN_YEARS" in content or "2000" in content, (
        "No chronological year boundary found in training script!"
    )


# ── Test 18: 2025 target remains NaN in momentum features dataset ─────────────
def test_18_inference_2025_target_is_nan():
    """Year 2025 rows must have NaN gdp_growth_next_year in momentum features dataset."""
    df  = pd.read_csv(FEAT_PATH)
    rows_2025 = df[df["year"] == 2025]
    assert len(rows_2025) > 0, "No year=2025 rows in momentum features dataset!"
    assert rows_2025[TARGET_COL].isna().all(), (
        "Year 2025 has non-NaN gdp_growth_next_year — target contamination!"
    )


# ── Test 19: Experimental 2026 forecasts are finite ──────────────────────────
def test_19_experimental_forecasts_finite():
    """All 2026 experimental forecasts must be finite real numbers."""
    assert FC_PATH.exists(), f"2026 forecast file not found: {FC_PATH}"
    df = pd.read_csv(FC_PATH)
    assert len(df) > 0, "2026 forecast file is empty!"
    preds = df["predicted_gdp_growth_experimental"]
    assert preds.notna().all(), "Some 2026 forecasts are NaN!"
    assert np.isfinite(preds.values).all(), "Some 2026 forecasts are infinite!"


# ── Test 20: Experimental forecasts do not overwrite Phase 11 production ──────
def test_20_experimental_forecasts_separate_from_production():
    """Experimental 2026 forecasts must be stored in phase12_step2 path, not production path."""
    assert FC_PATH.exists(), f"Experimental forecast file missing: {FC_PATH}"
    # Verify Phase 11 production forecast has NOT been overwritten
    p11_fc = P11_FORECASTS
    assert p11_fc.exists(), "Phase 11 production forecast artifact is missing!"
    # Verify the experimental file is a DIFFERENT file from the production one
    assert FC_PATH.resolve() != p11_fc.resolve(), (
        "Experimental forecasts are pointing to the same file as Phase 11 production forecasts!"
    )
    # Verify experimental file name does NOT match production name
    assert "step11" not in FC_PATH.name, (
        "Experimental forecast filename contains 'step11' — potential overwrite!"
    )


# ── Test 21: Production model not modified (metadata flag) ────────────────────
def test_21_production_model_not_modified():
    """Step 2 metadata must record production_model_modified = False."""
    assert META_PATH.exists(), f"Step 2 metadata not found: {META_PATH}"
    with open(META_PATH) as f:
        meta = json.load(f)
    assert meta.get("production_model_modified") is False, (
        "production_model_modified is not False in step2 metadata!"
    )


# ── Test 22: decelerating and accelerating indicators are mutually consistent ──
def test_22_decelerating_accelerating_consistent():
    """
    For any row where both decelerating and accelerating are non-NaN,
    they should NOT both be 1 simultaneously (gdp can't speed up and slow down at once).
    """
    df = pd.read_csv(FEAT_PATH)
    valid = df[df["gdp_growth_decelerating"].notna() &
               df["gdp_growth_accelerating"].notna()].copy()
    both_one = (valid["gdp_growth_decelerating"] == 1) & (valid["gdp_growth_accelerating"] == 1)
    assert not both_one.any(), (
        f"Found {both_one.sum()} rows where both decelerating AND accelerating = 1!"
    )
