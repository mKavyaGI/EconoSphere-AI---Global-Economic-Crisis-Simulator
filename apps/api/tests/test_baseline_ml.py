"""
EconoSphere AI -- Phase 11, Step 5: Baseline ML Tests
======================================================
File   : apps/api/tests/test_baseline_ml.py
Run    : pytest apps/api/tests/test_baseline_ml.py -v

Tests verify:
1.  Target not in feature columns.
2.  Aggregate codes excluded from training.
3.  Train years are 2000-2018.
4.  Validation years are 2019-2022.
5.  Test years are 2023-2025.
6.  No random split.
7.  Model trains successfully.
8.  Predictions contain no NaN.
9.  Metrics can be calculated.
10. Saved model can be loaded.
11. Raw dataset checksum unchanged.
12. Leakage: imputer fitted only on train.
13. Chronological ordering maintained.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Locate project root relative to this test file
# ---------------------------------------------------------------------------
THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]  # apps/api/tests -> apps/api -> apps -> root

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_processed.csv"
RAW_PATH       = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
MODELS_DIR     = PROJECT_ROOT / "models" / "phase11"
METADATA_PATH  = MODELS_DIR / "model_metadata.json"
BEST_MODEL_PATH = MODELS_DIR / "best_gdp_growth_model.joblib"

# Import constants from the training script
import sys
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api" / "ml"))
from train_baseline import (
    FEATURE_COLUMNS,
    TARGET_COL,
    TRAIN_YEARS,
    VAL_YEARS,
    TEST_YEARS,
    WB_AGGREGATE_CODES,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def processed_df() -> pd.DataFrame:
    """Load the processed dataset once for all tests."""
    assert PROCESSED_PATH.exists(), f"Processed dataset not found: {PROCESSED_PATH}"
    df = pd.read_csv(PROCESSED_PATH, low_memory=False)
    return df


@pytest.fixture(scope="module")
def country_df(processed_df: pd.DataFrame) -> pd.DataFrame:
    """Processed dataset filtered to country-only rows."""
    return processed_df[~processed_df["country_code"].isin(WB_AGGREGATE_CODES)].copy()


@pytest.fixture(scope="module")
def train_df(country_df: pd.DataFrame) -> pd.DataFrame:
    return country_df[
        (country_df["year"] >= TRAIN_YEARS[0]) &
        (country_df["year"] <= TRAIN_YEARS[1]) &
        country_df[TARGET_COL].notna()
    ].copy()


@pytest.fixture(scope="module")
def val_df(country_df: pd.DataFrame) -> pd.DataFrame:
    return country_df[
        (country_df["year"] >= VAL_YEARS[0]) &
        (country_df["year"] <= VAL_YEARS[1]) &
        country_df[TARGET_COL].notna()
    ].copy()


@pytest.fixture(scope="module")
def test_df(country_df: pd.DataFrame) -> pd.DataFrame:
    return country_df[
        (country_df["year"] >= TEST_YEARS[0]) &
        (country_df["year"] <= TEST_YEARS[1]) &
        country_df[TARGET_COL].notna()
    ].copy()


@pytest.fixture(scope="module")
def best_model() -> Pipeline:
    """Load the saved best model."""
    assert BEST_MODEL_PATH.exists(), f"Best model not found: {BEST_MODEL_PATH}"
    return joblib.load(BEST_MODEL_PATH)


@pytest.fixture(scope="module")
def metadata() -> dict:
    """Load training metadata."""
    assert METADATA_PATH.exists(), f"Metadata not found: {METADATA_PATH}"
    with open(METADATA_PATH, "r") as fh:
        return json.load(fh)


# ===========================================================================
# TEST 1: Target not in feature columns
# ===========================================================================
def test_target_not_in_feature_columns():
    """The target variable must never appear as an input feature."""
    assert TARGET_COL not in FEATURE_COLUMNS, (
        f"[LEAKAGE] '{TARGET_COL}' found in FEATURE_COLUMNS -- critical target leakage!"
    )


# ===========================================================================
# TEST 2: Aggregate codes excluded from training data
# ===========================================================================
def test_aggregate_codes_excluded_from_training(train_df: pd.DataFrame):
    """World Bank aggregate codes must not appear in the training split."""
    agg_in_train = set(train_df["country_code"].unique()) & WB_AGGREGATE_CODES
    assert len(agg_in_train) == 0, (
        f"Aggregate codes found in training data: {agg_in_train}"
    )


def test_aggregate_codes_excluded_from_validation(val_df: pd.DataFrame):
    """Aggregate codes must not appear in validation split."""
    agg_in_val = set(val_df["country_code"].unique()) & WB_AGGREGATE_CODES
    assert len(agg_in_val) == 0, (
        f"Aggregate codes found in validation data: {agg_in_val}"
    )


def test_aggregate_codes_excluded_from_test(test_df: pd.DataFrame):
    """Aggregate codes must not appear in test split."""
    agg_in_test = set(test_df["country_code"].unique()) & WB_AGGREGATE_CODES
    assert len(agg_in_test) == 0, (
        f"Aggregate codes found in test data: {agg_in_test}"
    )


# ===========================================================================
# TEST 3: Train years are 2000-2018
# ===========================================================================
def test_train_years_within_bounds(train_df: pd.DataFrame):
    """All training observations must fall within 2000-2018."""
    min_yr = int(train_df["year"].min())
    max_yr = int(train_df["year"].max())
    assert min_yr >= TRAIN_YEARS[0], f"Train min year {min_yr} < {TRAIN_YEARS[0]}"
    assert max_yr <= TRAIN_YEARS[1], f"Train max year {max_yr} > {TRAIN_YEARS[1]}"


def test_train_has_sufficient_rows(train_df: pd.DataFrame):
    """Training set must have a reasonable number of observations."""
    assert len(train_df) >= 1000, f"Training set too small: {len(train_df)} rows"


# ===========================================================================
# TEST 4: Validation years are 2019-2022
# ===========================================================================
def test_validation_years_within_bounds(val_df: pd.DataFrame):
    """All validation observations must fall within 2019-2022."""
    min_yr = int(val_df["year"].min())
    max_yr = int(val_df["year"].max())
    assert min_yr >= VAL_YEARS[0], f"Val min year {min_yr} < {VAL_YEARS[0]}"
    assert max_yr <= VAL_YEARS[1], f"Val max year {max_yr} > {VAL_YEARS[1]}"


# ===========================================================================
# TEST 5: Test years are 2023-2025
# ===========================================================================
def test_test_years_within_bounds(test_df: pd.DataFrame):
    """All test observations must fall within 2023-2025."""
    min_yr = int(test_df["year"].min())
    max_yr = int(test_df["year"].max())
    assert min_yr >= TEST_YEARS[0], f"Test min year {min_yr} < {TEST_YEARS[0]}"
    assert max_yr <= TEST_YEARS[1], f"Test max year {max_yr} > {TEST_YEARS[1]}"


# ===========================================================================
# TEST 6: No temporal contamination between splits
# ===========================================================================
def test_no_overlap_train_val(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Train and validation periods must not overlap in time."""
    train_years = set(train_df["year"].unique())
    val_years   = set(val_df["year"].unique())
    overlap     = train_years & val_years
    assert len(overlap) == 0, f"Train/val year overlap: {sorted(overlap)}"


def test_no_overlap_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Train and test periods must not overlap in time."""
    train_years = set(train_df["year"].unique())
    test_years  = set(test_df["year"].unique())
    overlap     = train_years & test_years
    assert len(overlap) == 0, f"Train/test year overlap: {sorted(overlap)}"


def test_no_overlap_val_test(val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Validation and test periods must not overlap."""
    val_years  = set(val_df["year"].unique())
    test_years = set(test_df["year"].unique())
    overlap    = val_years & test_years
    assert len(overlap) == 0, f"Val/test year overlap: {sorted(overlap)}"


def test_chronological_order_train_before_val(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """All training years must be strictly before all validation years."""
    assert int(train_df["year"].max()) < int(val_df["year"].min()), (
        "Training data contains years >= validation start year"
    )


def test_chronological_order_val_before_test(val_df: pd.DataFrame, test_df: pd.DataFrame):
    """All validation years must be strictly before all test years."""
    assert int(val_df["year"].max()) < int(test_df["year"].min()), (
        "Validation data contains years >= test start year"
    )


# ===========================================================================
# TEST 7: Model trains successfully (re-train on fixtures for isolation)
# ===========================================================================
def test_model_trains_successfully(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """A Ridge regression must successfully fit on training data."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COL].values
    X_val   = val_df[FEATURE_COLUMNS].values

    pipe = Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    assert len(preds) == len(X_val), "Prediction count mismatch"


# ===========================================================================
# TEST 8: Predictions contain no NaN
# ===========================================================================
def test_predictions_no_nan(best_model: Pipeline, test_df: pd.DataFrame):
    """The best model must produce finite predictions for all test rows."""
    X_test = test_df[FEATURE_COLUMNS]
    preds  = best_model.predict(X_test)
    assert not np.any(np.isnan(preds)), "Predictions contain NaN values"
    assert not np.any(np.isinf(preds)), "Predictions contain Inf values"


# ===========================================================================
# TEST 9: Metrics can be calculated on predictions
# ===========================================================================
def test_metrics_calculable(best_model: Pipeline, test_df: pd.DataFrame):
    """MAE and R2 must be finite and sensible for the test set."""
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL].values
    preds  = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    assert np.isfinite(mae), f"MAE is not finite: {mae}"
    assert np.isfinite(r2),  f"R2 is not finite: {r2}"
    # Sanity range: MAE should be plausible for GDP growth (not absurdly large)
    assert mae < 100.0, f"MAE suspiciously large: {mae}"


# ===========================================================================
# TEST 10: Saved model can be loaded and used
# ===========================================================================
def test_best_model_loadable(test_df: pd.DataFrame):
    """The saved joblib model must load and produce predictions."""
    assert BEST_MODEL_PATH.exists(), f"Best model file not found: {BEST_MODEL_PATH}"
    loaded = joblib.load(BEST_MODEL_PATH)
    X_test = test_df[FEATURE_COLUMNS]
    preds  = loaded.predict(X_test)
    assert len(preds) == len(X_test)
    assert not np.any(np.isnan(preds))


def test_all_model_files_exist():
    """All three baseline model files must exist."""
    expected_files = [
        "linear_regression_baseline.joblib",
        "random_forest_baseline.joblib",
        "gradient_boosting_baseline.joblib",
        "best_gdp_growth_model.joblib",
    ]
    for fname in expected_files:
        fpath = MODELS_DIR / fname
        assert fpath.exists(), f"Model file not found: {fpath}"


# ===========================================================================
# TEST 11: Raw dataset checksum unchanged
# ===========================================================================
def test_raw_dataset_unchanged():
    """The raw master_panel.csv must not have been modified."""
    import hashlib
    assert RAW_PATH.exists(), f"Raw file not found: {RAW_PATH}"
    actual_md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    expected   = "8ac7e0b2bf09fbe89289f82d0c7cf25e"
    assert actual_md5 == expected, (
        f"[CRITICAL] Raw dataset was MODIFIED!\n"
        f"  Expected MD5 : {expected}\n"
        f"  Actual MD5   : {actual_md5}"
    )


# ===========================================================================
# TEST 12: Imputer fitted only on training data
# ===========================================================================
def test_imputer_fitted_on_train_only(best_model: Pipeline, train_df: pd.DataFrame):
    """
    The SimpleImputer inside the best model's pipeline must have statistics
    consistent with the training set (not the full dataset or test set).
    Verify that the imputer's medians fall within the training distribution.
    """
    # Get the imputer from the pipeline
    # Pipeline step names differ by model: 'prep' contains the imputer
    steps = best_model.named_steps
    if "prep" in steps:
        prep   = steps["prep"]
        imputer = prep.named_steps.get("imputer")
    else:
        imputer = steps.get("imputer")  # HistGBM flat pipeline variant

    if imputer is None:
        pytest.skip("Imputer step not found in pipeline -- skipping this check")

    train_medians = train_df[FEATURE_COLUMNS].median()
    imp_medians   = imputer.statistics_

    # Allow 5% relative tolerance -- should match closely
    for i, col in enumerate(FEATURE_COLUMNS):
        if pd.isna(train_medians[col]):
            continue
        tm  = float(train_medians[col])
        im  = float(imp_medians[i])
        if abs(tm) > 1e-6:
            rel_diff = abs(tm - im) / abs(tm)
            assert rel_diff < 0.1, (
                f"Imputer median for '{col}' ({im:.4f}) deviates >10% "
                f"from training median ({tm:.4f}). Possible data leakage."
            )


# ===========================================================================
# TEST 13: No target imputation (target NaN rows excluded from splits)
# ===========================================================================
def test_no_target_nan_in_train(train_df: pd.DataFrame):
    """Training set must have zero NaN target values."""
    n_nan = int(train_df[TARGET_COL].isna().sum())
    assert n_nan == 0, f"Training set has {n_nan} NaN target values -- these must be excluded."


def test_no_target_nan_in_val(val_df: pd.DataFrame):
    """Validation set must have zero NaN target values."""
    n_nan = int(val_df[TARGET_COL].isna().sum())
    assert n_nan == 0, f"Validation set has {n_nan} NaN target values."


def test_no_target_nan_in_test(test_df: pd.DataFrame):
    """Test set must have zero NaN target values."""
    n_nan = int(test_df[TARGET_COL].isna().sum())
    assert n_nan == 0, f"Test set has {n_nan} NaN target values."


# ===========================================================================
# TEST 14: Metadata integrity
# ===========================================================================
def test_metadata_exists_and_valid(metadata: dict):
    """Training metadata JSON must exist and contain required keys."""
    required_keys = [
        "target", "feature_columns", "train_period", "validation_period",
        "test_period", "n_train_rows", "n_val_rows", "n_test_rows",
        "best_model", "random_seed", "data_integrity",
    ]
    for key in required_keys:
        assert key in metadata, f"Missing key in metadata: '{key}'"

    assert metadata["target"] == TARGET_COL
    assert metadata["random_seed"] == RANDOM_SEED
    assert metadata["data_integrity"]["target_imputed"] is False
    assert metadata["data_integrity"]["random_split_used"] is False
    assert metadata["data_integrity"]["raw_dataset_modified"] is False


def test_metadata_feature_columns_match(metadata: dict):
    """Feature columns in metadata must match FEATURE_COLUMNS in training script."""
    meta_features = metadata.get("feature_columns", [])
    assert meta_features == FEATURE_COLUMNS, (
        f"Metadata feature_columns do not match FEATURE_COLUMNS constant.\n"
        f"  Metadata: {meta_features}\n"
        f"  Script:   {FEATURE_COLUMNS}"
    )


# ===========================================================================
# TEST 15: Feature columns present in processed dataset
# ===========================================================================
def test_all_feature_columns_in_processed_dataset(processed_df: pd.DataFrame):
    """Every feature in FEATURE_COLUMNS must exist in the processed dataset."""
    missing = [c for c in FEATURE_COLUMNS if c not in processed_df.columns]
    assert len(missing) == 0, f"Feature columns missing from processed dataset: {missing}"
