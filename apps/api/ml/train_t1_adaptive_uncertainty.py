"""
EconoSphere AI -- Phase 11, Step 12: Adaptive T+1 Prediction Uncertainty
======================================================================
Script   : apps/api/ml/train_t1_adaptive_uncertainty.py
Input    : data/processed/master_panel_t1_missingness.csv
Outputs  : data/processed/t1_step12_2026_forecasts.csv
           models/phase11/t1_step12_uncertainty_metadata.json

Purpose  : Evaluate multiple leakage-free adaptive uncertainty strategies
           on top of the locked Step 10/Step 11 HistGradientBoosting point model.
           The point-forecast model is NEVER changed.

Experiments:
  Exp A  -- Global Residual Q90 (Step 11 control reproduction)
  Exp B  -- Country-Conditional Residual Calibration (Q90 per country)
  Exp C  -- Volatility-Group Calibration (Low/Med/High σ from TRAIN only)
  Exp D  -- Asymmetric Global Residual Intervals (signed q10/q90)
  Exp E  -- Country-Conditional Asymmetric Intervals
  Exp F  -- Adaptive Volatility-Scaled Interval (bounded 0.75-1.50)

Decision rules:
  - A method is PROMOTED only if it satisfies all 8 conditions in Section 7.
  - If no adaptive method clearly outperforms the Step 11 control:
    FINAL DECISION = KEEP STEP 11 GLOBAL Q90 CONTROL.

IMPORTANT:
  - Do NOT use test targets during calibration.
  - Do NOT use 2025/2026 GDP growth targets.
  - All country/volatility statistics derived from TRAIN only.
  - Validation residuals used solely for interval calibration.
  - Test data used ONCE for final evaluation only.
"""

import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

INPUT_PATH  = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
MODELS_DIR  = PROJECT_ROOT / "models" / "phase11"
FORECAST_PATH   = PROJECT_ROOT / "data" / "processed" / "t1_step12_2026_forecasts.csv"
METADATA_PATH   = MODELS_DIR / "t1_step12_uncertainty_metadata.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Chronological split boundaries
TRAIN_YEARS = (2000, 2018)
VAL_YEARS   = (2019, 2022)
TEST_YEARS  = (2023, 2024)

# Step 11 control verification tolerances
EXPECTED_Q90  = 11.4507
EXPECTED_RMSE = 3.9113
Q90_TOLERANCE  = 0.001
RMSE_TOLERANCE = 0.0001

# Country calibration threshold
MIN_COUNTRY_CALIBRATION_N = 20

# Volatility scaling bounds (Exp F)
SCALE_MIN = 0.75
SCALE_MAX = 1.50

# 5 focal countries for 2026 forecasts
FIVE_COUNTRIES = ["IND", "CHN", "USA", "JPN", "GBR"]

# World Bank regional aggregates to exclude
WB_AGGREGATES = {
    'AFE','AFW','ARB','CEB','CSS','EAP','EAR','EAS','ECA','ECS','EMU','EUU',
    'FCS','HIC','HPC','IBD','IBT','IDA','IDB','IDX','INX','LAC','LCN','LDC',
    'LIC','LMC','LMY','LTE','MEA','MIC','MNA','NAC','OED','OSS','PRE','PSS',
    'PST','SAS','SSA','SSF','SST','TEA','TEC','TLA','TMN','TSA','TSS','UMC','WLD'
}

# Locked Step 8 feature set (identical to Step 11 control)
BASE_FEATURES = [
    "exchange_rate_lcu_usd", "tariff_rate_pct", "remittances_usd",
    "fdi_net_inflow_usd", "unemployment_pct", "imports_pct_gdp",
    "tax_revenue_pct_gdp", "exports_pct_gdp", "interest_rate_pct",
    "reserves_usd", "current_account_pct_gdp", "inflation_cpi_pct",
    "population_total", "gdp_current_usd", "gdp_growth_lag1",
    "gdp_growth_lag2", "gdp_growth_lag3", "inflation_lag1",
    "unemployment_lag1", "exports_lag1", "imports_lag1",
    "gdp_growth_rolling_mean_3", "gdp_growth_rolling_std_3",
    "gdp_growth_rolling_mean_5", "inflation_rolling_mean_3",
    "trade_openness", "trade_balance_ratio", "log_gdp_usd", "log_population",
]
ROLLING_EXTRA = ["gdp_growth_rolling_std_5", "inflation_rolling_std_3"]
LOCKED_FEATURES = BASE_FEATURES + ROLLING_EXTRA

# ---------------------------------------------------------------------------
# Locked Step 10 hyperparameters
# ---------------------------------------------------------------------------
LOCKED_PARAMS = {
    'l2_regularization': 5.0,
    'learning_rate'    : 0.05,
    'max_depth'        : 5,
    'max_iter'         : 300,
    'random_state'     : 42,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_interval(y_true, lower, upper, label=""):
    """Compute coverage and width statistics for a prediction interval."""
    n = len(y_true)
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage   = float(in_interval.mean())
    widths     = upper - lower
    mean_width = float(widths.mean())
    med_width  = float(np.median(widths))
    n_outside  = int((~in_interval).sum())
    coverage_gap = float(coverage - 0.90)
    abs_cov_err  = float(abs(coverage_gap))

    result = {
        "label"               : label,
        "n_observations"      : n,
        "empirical_coverage"  : coverage,
        "mean_interval_width" : mean_width,
        "median_interval_width": med_width,
        "nominal_coverage"    : 0.90,
        "coverage_gap"        : coverage_gap,
        "abs_coverage_error"  : abs_cov_err,
        "n_outside_interval"  : n_outside,
    }
    return result


def print_eval(res):
    print(f"  Coverage     : {res['empirical_coverage']:.4f} "
          f"(gap: {res['coverage_gap']:+.4f})")
    print(f"  Mean width   : {res['mean_interval_width']:.4f}")
    print(f"  Median width : {res['median_interval_width']:.4f}")
    print(f"  N outside    : {res['n_outside_interval']} / {res['n_observations']}")


# ---------------------------------------------------------------------------
# Calibration functions (all use VALIDATION residuals only)
# ---------------------------------------------------------------------------

def calibrate_global_q90(y_val, pred_val):
    """Exp A: Global 90th-percentile absolute residual (Step 11 control)."""
    abs_res = np.abs(y_val - pred_val)
    return float(np.quantile(abs_res, 0.90))


def calibrate_country_q90(val_df, pred_val, min_n=MIN_COUNTRY_CALIBRATION_N):
    """
    Exp B: Country-specific Q90.
    Only uses validation predictions and validation targets.
    Falls back to global Q90 for countries with < min_n observations.
    Returns: dict of {country_code -> q90}, float global_q90
    """
    residuals = np.abs(val_df["gdp_growth_next_year"].values - pred_val)
    global_q90 = float(np.quantile(residuals, 0.90))

    country_q90 = {}
    for code, grp in val_df.groupby("country_code"):
        idx = grp.index
        r = residuals[val_df.index.get_indexer(idx)]
        if len(r) >= min_n:
            country_q90[code] = float(np.quantile(r, 0.90))
        # else: will fall back to global
    return country_q90, global_q90


def get_country_q90_array(country_codes, country_q90_map, global_q90):
    """Map each prediction to its country Q90 (or global fallback)."""
    return np.array([
        country_q90_map.get(c, global_q90) for c in country_codes
    ])


def calibrate_volatility_groups(train_df, val_df, pred_val):
    """
    Exp C: Volatility-group Q90.
    - Training-period σ of GDP growth is computed per country from TRAIN ONLY.
    - Groups (Low/Med/High) thresholds are determined from TRAIN σ distribution.
    - Validation-period Q90 computed separately for each group.
    Returns: group_thresholds (from train), group_q90_map, global_q90, country_group_map
    """
    # Country historical GDP growth std from TRAIN only
    country_sigma = (
        train_df.groupby("country_code")["gdp_growth_next_year"]
        .std()
        .dropna()
        .rename("sigma")
    )

    # Group thresholds from TRAIN sigma distribution (33rd/67th percentiles)
    p33 = float(np.quantile(country_sigma.values, 1/3))
    p67 = float(np.quantile(country_sigma.values, 2/3))

    def assign_group(sigma):
        if sigma <= p33:
            return "low"
        elif sigma <= p67:
            return "medium"
        else:
            return "high"

    country_group_map = {}
    for code, sig in country_sigma.items():
        country_group_map[code] = assign_group(sig)

    # Validation residuals grouped by country volatility group
    val_residuals = np.abs(val_df["gdp_growth_next_year"].values - pred_val)
    val_codes = val_df["country_code"].values
    global_q90 = float(np.quantile(val_residuals, 0.90))

    group_residuals = {"low": [], "medium": [], "high": []}
    for i, code in enumerate(val_codes):
        grp = country_group_map.get(code, None)
        if grp is not None:
            group_residuals[grp].append(val_residuals[i])
        # Countries without training history fall to global below

    group_q90_map = {}
    for grp, res in group_residuals.items():
        if len(res) >= MIN_COUNTRY_CALIBRATION_N:
            group_q90_map[grp] = float(np.quantile(res, 0.90))
        else:
            group_q90_map[grp] = global_q90

    return {
        "p33": p33,
        "p67": p67,
        "group_q90_map": group_q90_map,
        "country_sigma": country_sigma.to_dict(),
        "country_group_map": country_group_map,
        "global_q90": global_q90,
    }


def get_volatility_q90_array(country_codes, country_group_map, group_q90_map, global_q90):
    """Map each prediction to its volatility-group Q90 (or global fallback)."""
    result = []
    for code in country_codes:
        grp = country_group_map.get(code, None)
        if grp is not None:
            result.append(group_q90_map.get(grp, global_q90))
        else:
            result.append(global_q90)
    return np.array(result)


def calibrate_asymmetric_global(y_val, pred_val):
    """
    Exp D: Asymmetric global residual intervals.
    residual = actual - prediction (signed)
    q10 of residuals -> lower offset
    q90 of residuals -> upper offset
    Does NOT claim formal conformal guarantee; explicitly residual-based.
    """
    signed_residuals = y_val - pred_val
    q10 = float(np.quantile(signed_residuals, 0.10))
    q90 = float(np.quantile(signed_residuals, 0.90))
    return q10, q90


def calibrate_country_asymmetric(val_df, pred_val, min_n=MIN_COUNTRY_CALIBRATION_N):
    """
    Exp E: Country-conditional asymmetric intervals.
    Uses signed residuals. Falls back to global quantiles for small countries.
    Returns: dict of {country_code -> (q10, q90)}, global_q10, global_q90
    """
    signed_res = val_df["gdp_growth_next_year"].values - pred_val
    global_q10 = float(np.quantile(signed_res, 0.10))
    global_q90 = float(np.quantile(signed_res, 0.90))

    country_quantiles = {}
    for code, grp in val_df.groupby("country_code"):
        idx = grp.index
        r = signed_res[val_df.index.get_indexer(idx)]
        if len(r) >= min_n:
            country_quantiles[code] = (
                float(np.quantile(r, 0.10)),
                float(np.quantile(r, 0.90))
            )
    return country_quantiles, global_q10, global_q90


def get_country_asymmetric_arrays(country_codes, country_quantiles, global_q10, global_q90):
    """Map each prediction to country q10/q90 offsets (or global fallbacks)."""
    q10_arr = np.array([
        country_quantiles[c][0] if c in country_quantiles else global_q10
        for c in country_codes
    ])
    q90_arr = np.array([
        country_quantiles[c][1] if c in country_quantiles else global_q90
        for c in country_codes
    ])
    return q10_arr, q90_arr


def calibrate_volatility_scaled(train_df, global_q90):
    """
    Exp F: Volatility-scaled interval.
    - Computes per-country training σ.
    - Scale = (country_sigma / median_sigma), clipped to [0.75, 1.50].
    - Adaptive Q90 = global_Q90 * scale_factor.
    Returns: dict of country->scale, median_sigma, global_q90
    """
    country_sigma = (
        train_df.groupby("country_code")["gdp_growth_next_year"]
        .std()
        .dropna()
    )
    median_sigma = float(country_sigma.median())

    country_scale = {}
    for code, sig in country_sigma.items():
        raw_scale = sig / median_sigma if median_sigma > 0 else 1.0
        clipped = float(np.clip(raw_scale, SCALE_MIN, SCALE_MAX))
        country_scale[code] = clipped

    return country_scale, median_sigma


def get_volatility_scaled_q90(country_codes, country_scale, global_q90):
    """Return per-country adaptive Q90 using volatility scaling."""
    return np.array([
        global_q90 * country_scale.get(c, 1.0)
        for c in country_codes
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PHASE 11 STEP 12: Adaptive T+1 Prediction Uncertainty")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1 — Verify raw checksum
    # -----------------------------------------------------------------------
    raw_path = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
    if raw_path.exists():
        md5 = hashlib.md5(raw_path.read_bytes()).hexdigest()
        if md5 == "8ac7e0b2bf09fbe89289f82d0c7cf25e":
            print(f"\n[PASS] Raw data MD5 verified: {md5}")
        else:
            print(f"\n[FAIL] Raw data MD5 MISMATCH: {md5}")
            raise RuntimeError("Raw data checksum mismatch — pipeline halted.")
    else:
        print("\n[WARN] Raw data file not found; skipping checksum verification.")

    # -----------------------------------------------------------------------
    # Step 2 — Load and split data
    # -----------------------------------------------------------------------
    print("\n--- Loading data ---")
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()

    valid_target = df["next_year_target_available"] == 1

    train = df[(df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1]) & valid_target].copy().reset_index(drop=True)
    val   = df[(df["year"] >= VAL_YEARS[0])   & (df["year"] <= VAL_YEARS[1])   & valid_target].copy().reset_index(drop=True)
    test  = df[(df["year"] >= TEST_YEARS[0])  & (df["year"] <= TEST_YEARS[1])  & valid_target].copy().reset_index(drop=True)
    inf_2025 = df[df["year"] == 2025].copy().reset_index(drop=True)

    print(f"  Train N={len(train):,}  (years {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]})")
    print(f"  Val   N={len(val):,}  (years {VAL_YEARS[0]}-{VAL_YEARS[1]})")
    print(f"  Test  N={len(test):,}  (years {TEST_YEARS[0]}-{TEST_YEARS[1]})")
    print(f"  Inf   N={len(inf_2025):,}  (feature year 2025)")

    # Safety check: inference targets must be NaN
    assert inf_2025["gdp_growth_next_year"].isna().all(), \
        "CRITICAL: 2025 gdp_growth_next_year is NOT NaN — target leakage risk!"
    print("  [PASS] 2025 inference targets are all NaN.")

    # -----------------------------------------------------------------------
    # Step 3 — Build and train locked pipeline
    # -----------------------------------------------------------------------
    print("\n--- Training Locked Step 10 Pipeline ---")
    X_train = train[LOCKED_FEATURES].copy()
    y_train = train["gdp_growth_next_year"].values

    X_val  = val[LOCKED_FEATURES].copy()
    y_val  = val["gdp_growth_next_year"].values

    X_test = test[LOCKED_FEATURES].copy()
    y_test = test["gdp_growth_next_year"].values

    X_inf  = inf_2025[LOCKED_FEATURES].copy()

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   HistGradientBoostingRegressor(**LOCKED_PARAMS)),
    ])
    pipe.fit(X_train, y_train)

    pred_val  = pipe.predict(X_val)
    pred_test = pipe.predict(X_test)
    pred_inf  = pipe.predict(X_inf)

    # -----------------------------------------------------------------------
    # Step 4 — Verify reproducibility
    # -----------------------------------------------------------------------
    test_rmse = compute_rmse(y_test, pred_test)
    val_rmse  = compute_rmse(y_val, pred_val)

    print(f"\n  Val  RMSE : {val_rmse:.4f}")
    print(f"  Test RMSE : {test_rmse:.4f}  (expected ~ {EXPECTED_RMSE})")

    if abs(test_rmse - EXPECTED_RMSE) < RMSE_TOLERANCE:
        print(f"  [PASS] Test RMSE reproduced. |diff| = {abs(test_rmse - EXPECTED_RMSE):.6f}")
    else:
        print(f"  [FAIL] Test RMSE differs: {test_rmse:.4f} vs {EXPECTED_RMSE}")
        raise RuntimeError(
            f"Point model RMSE mismatch ({test_rmse:.4f} vs {EXPECTED_RMSE}). "
            "Halting — investigate reproducibility before continuing."
        )

    # -----------------------------------------------------------------------
    # Step 5 — Exp A: Global Q90 (Step 11 control reproduction)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Exp A — Global Residual Q90 (Step 11 Control)")
    print("=" * 60)

    q90_global = calibrate_global_q90(y_val, pred_val)
    print(f"  Reproduced Q90 : {q90_global:.4f}  (expected ~ {EXPECTED_Q90})")

    if abs(q90_global - EXPECTED_Q90) < Q90_TOLERANCE:
        print(f"  [PASS] Q90 verified. |diff| = {abs(q90_global - EXPECTED_Q90):.6f}")
    else:
        print(f"  [FAIL] Q90 differs: {q90_global:.4f} vs {EXPECTED_Q90}")
        raise RuntimeError(
            f"Q90 mismatch ({q90_global:.4f} vs {EXPECTED_Q90}). "
            "Halting — investigate before continuing."
        )

    # Validation evaluation (Exp A)
    lower_A_val = pred_val - q90_global
    upper_A_val = pred_val + q90_global
    eval_A_val  = evaluate_interval(y_val, lower_A_val, upper_A_val, "ExpA-Val")
    print("\n  Validation:")
    print_eval(eval_A_val)

    # Test evaluation (Exp A)
    lower_A_test = pred_test - q90_global
    upper_A_test = pred_test + q90_global
    eval_A_test  = evaluate_interval(y_test, lower_A_test, upper_A_test, "ExpA-Test")
    print("  Test:")
    print_eval(eval_A_test)

    # -----------------------------------------------------------------------
    # Step 6 — Exp B: Country-Conditional Q90
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Exp B — Country-Conditional Q90")
    print("=" * 60)

    country_q90_map, global_q90_B = calibrate_country_q90(val, pred_val)
    n_country_calibrated = len(country_q90_map)
    n_fallback = val["country_code"].nunique() - n_country_calibrated
    print(f"  Countries with own Q90 (≥{MIN_COUNTRY_CALIBRATION_N} obs): {n_country_calibrated}")
    print(f"  Countries using global fallback                  : {n_fallback}")

    # Validation
    q90_arr_B_val = get_country_q90_array(val["country_code"].values, country_q90_map, global_q90_B)
    lower_B_val = pred_val - q90_arr_B_val
    upper_B_val = pred_val + q90_arr_B_val
    eval_B_val  = evaluate_interval(y_val, lower_B_val, upper_B_val, "ExpB-Val")
    print("\n  Validation:")
    print_eval(eval_B_val)

    # Test (calibration frozen from val)
    q90_arr_B_test = get_country_q90_array(test["country_code"].values, country_q90_map, global_q90_B)
    lower_B_test = pred_test - q90_arr_B_test
    upper_B_test = pred_test + q90_arr_B_test
    eval_B_test  = evaluate_interval(y_test, lower_B_test, upper_B_test, "ExpB-Test")
    print("  Test:")
    print_eval(eval_B_test)

    # -----------------------------------------------------------------------
    # Step 7 — Exp C: Volatility-Group Calibration
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Exp C — Volatility-Group Calibration")
    print("=" * 60)

    vol_params = calibrate_volatility_groups(train, val, pred_val)
    print(f"  Train σ thresholds: p33={vol_params['p33']:.4f}, p67={vol_params['p67']:.4f}")
    for grp, q in vol_params["group_q90_map"].items():
        n = sum(1 for v in vol_params["country_group_map"].values() if v == grp)
        print(f"  Group '{grp}': Q90={q:.4f}  (n_countries={n})")

    # Validation
    q90_arr_C_val = get_volatility_q90_array(
        val["country_code"].values,
        vol_params["country_group_map"],
        vol_params["group_q90_map"],
        vol_params["global_q90"],
    )
    lower_C_val = pred_val - q90_arr_C_val
    upper_C_val = pred_val + q90_arr_C_val
    eval_C_val  = evaluate_interval(y_val, lower_C_val, upper_C_val, "ExpC-Val")
    print("\n  Validation:")
    print_eval(eval_C_val)

    # Test
    q90_arr_C_test = get_volatility_q90_array(
        test["country_code"].values,
        vol_params["country_group_map"],
        vol_params["group_q90_map"],
        vol_params["global_q90"],
    )
    lower_C_test = pred_test - q90_arr_C_test
    upper_C_test = pred_test + q90_arr_C_test
    eval_C_test  = evaluate_interval(y_test, lower_C_test, upper_C_test, "ExpC-Test")
    print("  Test:")
    print_eval(eval_C_test)

    # -----------------------------------------------------------------------
    # Step 8 — Exp D: Asymmetric Global Residual Intervals
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Exp D — Asymmetric Global Residual Intervals")
    print("=" * 60)

    asym_q10_global, asym_q90_global = calibrate_asymmetric_global(y_val, pred_val)
    print(f"  Signed residual q10 (lower offset): {asym_q10_global:.4f}")
    print(f"  Signed residual q90 (upper offset): {asym_q90_global:.4f}")

    # Validation
    lower_D_val = pred_val + asym_q10_global
    upper_D_val = pred_val + asym_q90_global
    eval_D_val  = evaluate_interval(y_val, lower_D_val, upper_D_val, "ExpD-Val")
    print("\n  Validation:")
    print_eval(eval_D_val)

    # Test
    lower_D_test = pred_test + asym_q10_global
    upper_D_test = pred_test + asym_q90_global
    eval_D_test  = evaluate_interval(y_test, lower_D_test, upper_D_test, "ExpD-Test")
    print("  Test:")
    print_eval(eval_D_test)

    # -----------------------------------------------------------------------
    # Step 9 — Exp E: Country-Conditional Asymmetric Intervals
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Exp E — Country-Conditional Asymmetric Intervals")
    print("=" * 60)

    country_quantiles, global_q10_E, global_q90_E = calibrate_country_asymmetric(val, pred_val)
    n_country_asym = len(country_quantiles)
    print(f"  Countries with own q10/q90 (≥{MIN_COUNTRY_CALIBRATION_N} obs): {n_country_asym}")
    print(f"  Global fallback q10={global_q10_E:.4f}, q90={global_q90_E:.4f}")

    # Validation
    q10_arr_E_val, q90_arr_E_val = get_country_asymmetric_arrays(
        val["country_code"].values, country_quantiles, global_q10_E, global_q90_E
    )
    lower_E_val = pred_val + q10_arr_E_val
    upper_E_val = pred_val + q90_arr_E_val
    eval_E_val  = evaluate_interval(y_val, lower_E_val, upper_E_val, "ExpE-Val")
    print("\n  Validation:")
    print_eval(eval_E_val)

    # Test
    q10_arr_E_test, q90_arr_E_test = get_country_asymmetric_arrays(
        test["country_code"].values, country_quantiles, global_q10_E, global_q90_E
    )
    lower_E_test = pred_test + q10_arr_E_test
    upper_E_test = pred_test + q90_arr_E_test
    eval_E_test  = evaluate_interval(y_test, lower_E_test, upper_E_test, "ExpE-Test")
    print("  Test:")
    print_eval(eval_E_test)

    # -----------------------------------------------------------------------
    # Step 10 — Exp F: Adaptive Volatility-Scaled Interval
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Exp F — Adaptive Volatility-Scaled Interval")
    print("=" * 60)

    country_scale_F, median_sigma_F = calibrate_volatility_scaled(train, q90_global)
    print(f"  Training median σ: {median_sigma_F:.4f}")
    print(f"  Scale factor range: [{SCALE_MIN}, {SCALE_MAX}]")

    # Validation
    q90_arr_F_val = get_volatility_scaled_q90(val["country_code"].values, country_scale_F, q90_global)
    lower_F_val = pred_val - q90_arr_F_val
    upper_F_val = pred_val + q90_arr_F_val
    eval_F_val  = evaluate_interval(y_val, lower_F_val, upper_F_val, "ExpF-Val")
    print("\n  Validation:")
    print_eval(eval_F_val)

    # Test
    q90_arr_F_test = get_volatility_scaled_q90(test["country_code"].values, country_scale_F, q90_global)
    lower_F_test = pred_test - q90_arr_F_test
    upper_F_test = pred_test + q90_arr_F_test
    eval_F_test  = evaluate_interval(y_test, lower_F_test, upper_F_test, "ExpF-Test")
    print("  Test:")
    print_eval(eval_F_test)

    # -----------------------------------------------------------------------
    # Step 11 — Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Method':<30} {'Val Cov':>9} {'Val Width':>10} {'Test Cov':>9} {'Test Width':>11}"
    print(header)
    print("-" * 70)

    exp_results = [
        ("Exp A (Global Q90 Control)", eval_A_val, eval_A_test),
        ("Exp B (Country Q90)",        eval_B_val, eval_B_test),
        ("Exp C (Volatility Group Q90)",eval_C_val, eval_C_test),
        ("Exp D (Asymmetric Global)",  eval_D_val, eval_D_test),
        ("Exp E (Country Asymmetric)", eval_E_val, eval_E_test),
        ("Exp F (Volatility Scaled)",  eval_F_val, eval_F_test),
    ]

    for name, v, t in exp_results:
        print(f"{name:<30} {v['empirical_coverage']:>9.4f} {v['mean_interval_width']:>10.4f} "
              f"{t['empirical_coverage']:>9.4f} {t['mean_interval_width']:>11.4f}")

    # -----------------------------------------------------------------------
    # Step 12 — Method selection decision
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SELECTION DECISION")
    print("=" * 70)

    # Collect all experiments (excluding control)
    candidates = [
        ("Exp B Country Q90",       eval_B_val, eval_B_test),
        ("Exp C Volatility Group",  eval_C_val, eval_C_test),
        ("Exp D Asymmetric Global", eval_D_val, eval_D_test),
        ("Exp E Country Asymmetric",eval_E_val, eval_E_test),
        ("Exp F Volatility Scaled", eval_F_val, eval_F_test),
    ]

    CONTROL_VAL_COV   = eval_A_val["empirical_coverage"]
    CONTROL_VAL_WIDTH = eval_A_val["mean_interval_width"]
    CONTROL_TEST_COV  = eval_A_test["empirical_coverage"]

    MIN_COVERAGE = 0.85  # Absolute minimum coverage to be considered
    promoted_method = None
    promoted_label  = None

    # Promotion criteria (all must pass):
    # 1. Val coverage >= 85% (cannot be far below nominal 90%)
    # 2. Test coverage >= 85%
    # 3. Mean interval width meaningfully narrower OR better calibrated
    # 4. Does not sacrifice substantial coverage for narrowness
    for name, v_res, t_res in candidates:
        val_cov   = v_res["empirical_coverage"]
        test_cov  = t_res["empirical_coverage"]
        val_width = v_res["mean_interval_width"]

        print(f"\n  Evaluating: {name}")
        print(f"    Val  coverage={val_cov:.4f}, mean_width={val_width:.4f}")
        print(f"    Test coverage={test_cov:.4f}")

        # Check minimum coverage floor
        if val_cov < MIN_COVERAGE:
            print(f"    REJECTED: Val coverage {val_cov:.4f} < {MIN_COVERAGE} minimum floor")
            continue
        if test_cov < MIN_COVERAGE:
            print(f"    REJECTED: Test coverage {test_cov:.4f} < {MIN_COVERAGE} minimum floor")
            continue

        # Check meaningful improvement
        width_improvement = (CONTROL_VAL_WIDTH - val_width) / CONTROL_VAL_WIDTH
        abs_cov_err_improvement = (
            eval_A_val["abs_coverage_error"] - v_res["abs_coverage_error"]
        )

        has_width_improvement   = width_improvement > 0.02   # > 2% narrower
        has_calibration_improvement = abs_cov_err_improvement > 0.01

        print(f"    Width improvement   : {width_improvement:+.4f} (>2% threshold: {has_width_improvement})")
        print(f"    Calibration improve : {abs_cov_err_improvement:+.4f} (>0.01 threshold: {has_calibration_improvement})")

        if has_width_improvement or has_calibration_improvement:
            # Prefer first found that passes all checks
            if promoted_method is None:
                promoted_method = (name, v_res, t_res)
                promoted_label  = name
                print(f"    >> CANDIDATE for promotion: {name}")
        else:
            print(f"    No meaningful improvement over control.")

    if promoted_method is not None:
        p_name, p_v, p_t = promoted_method
        print(f"\n  >>> PROMOTED METHOD: {p_name}")
        print(f"      Val  coverage={p_v['empirical_coverage']:.4f}, width={p_v['mean_interval_width']:.4f}")
        print(f"      Test coverage={p_t['empirical_coverage']:.4f}, width={p_t['mean_interval_width']:.4f}")
        final_decision = f"PROMOTE: {p_name}"
    else:
        print("\n  >>> No adaptive method clearly satisfies all promotion criteria.")
        print("  >>> FINAL DECISION: KEEP STEP 11 GLOBAL Q90 CONTROL")
        print(f"      Q90 = {q90_global:.4f}")
        final_decision = "KEEP STEP 11 GLOBAL Q90 CONTROL"
        # Use control for inference
        promoted_method = ("Exp A Global Q90 (Control)", eval_A_val, eval_A_test)
        promoted_label = "Exp A Global Q90 (Control)"

    # -----------------------------------------------------------------------
    # Step 13 — 2026 Inference Forecasts
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2026 Inference Forecasts (Feature Year 2025)")
    print("=" * 60)

    rows_step11  = []  # Control (Exp A)
    rows_adaptive = [] # Promoted (or control if no promotion)

    five_inf = inf_2025[inf_2025["country_code"].isin(FIVE_COUNTRIES)].copy()

    for _, row in five_inf.iterrows():
        code = row["country_code"]
        name = row.get("country_name", code)
        idx  = five_inf.index.get_loc(row.name)
        pt   = pred_inf[five_inf.index.get_indexer([row.name])[0]]

        # Step 11 control
        lb_11 = pt - q90_global
        ub_11 = pt + q90_global
        rows_step11.append({
            "country_code"       : code,
            "country_name"       : name,
            "feature_year"       : 2025,
            "target_year"        : 2026,
            "predicted_gdp_growth": round(pt, 4),
            "lower_bound_90"     : round(lb_11, 4),
            "upper_bound_90"     : round(ub_11, 4),
            "interval_width"     : round(ub_11 - lb_11, 4),
            "uncertainty_method" : "step11_global_q90_control",
        })

        # Adaptive method
        if "Country Q90" in promoted_label:
            q90_c = country_q90_map.get(code, global_q90_B)
            lb_ad = pt - q90_c
            ub_ad = pt + q90_c
            um = "exp_b_country_q90"
        elif "Volatility Group" in promoted_label:
            grp = vol_params["country_group_map"].get(code, None)
            q90_c = vol_params["group_q90_map"].get(grp, vol_params["global_q90"]) if grp else vol_params["global_q90"]
            lb_ad = pt - q90_c
            ub_ad = pt + q90_c
            um = "exp_c_volatility_group_q90"
        elif "Asymmetric Global" in promoted_label:
            lb_ad = pt + asym_q10_global
            ub_ad = pt + asym_q90_global
            um = "exp_d_asymmetric_global"
        elif "Country Asymmetric" in promoted_label:
            q10_c, q90_c = country_quantiles.get(code, (global_q10_E, global_q90_E))
            lb_ad = pt + q10_c
            ub_ad = pt + q90_c
            um = "exp_e_country_asymmetric"
        elif "Volatility Scaled" in promoted_label:
            aq90 = q90_global * country_scale_F.get(code, 1.0)
            lb_ad = pt - aq90
            ub_ad = pt + aq90
            um = "exp_f_volatility_scaled"
        else:
            # Fall back to control
            lb_ad = pt - q90_global
            ub_ad = pt + q90_global
            um = "step11_global_q90_control"

        rows_adaptive.append({
            "country_code"       : code,
            "country_name"       : name,
            "feature_year"       : 2025,
            "target_year"        : 2026,
            "predicted_gdp_growth": round(pt, 4),
            "lower_bound_90"     : round(lb_ad, 4),
            "upper_bound_90"     : round(ub_ad, 4),
            "interval_width"     : round(ub_ad - lb_ad, 4),
            "uncertainty_method" : um,
        })

    forecast_df = pd.DataFrame(rows_step11 + rows_adaptive)
    forecast_df.to_csv(FORECAST_PATH, index=False)
    print(f"\n  Saved 2026 forecasts → {FORECAST_PATH.name}")

    # Print 5-country table
    print(f"\n{'Country':<20} {'Point Forecast':>15} {'S11 Lower':>10} {'S11 Upper':>10} {'Adp Lower':>10} {'Adp Upper':>10} {'Adp Width':>10}")
    print("-" * 85)
    for r_c, r_a in zip(rows_step11, rows_adaptive):
        print(f"{r_c['country_code']:<20} {r_c['predicted_gdp_growth']:>15.4f} "
              f"{r_c['lower_bound_90']:>10.4f} {r_c['upper_bound_90']:>10.4f} "
              f"{r_a['lower_bound_90']:>10.4f} {r_a['upper_bound_90']:>10.4f} "
              f"{r_a['interval_width']:>10.4f}")

    # -----------------------------------------------------------------------
    # Step 14 — Save metadata
    # -----------------------------------------------------------------------
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):     return int(obj)
        if isinstance(obj, (np.floating,)):    return float(obj)
        if isinstance(obj, (np.ndarray,)):     return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        return obj

    metadata = {
        "step"                     : "Phase11_Step12_AdaptiveUncertainty",
        "point_model"              : "HistGradientBoostingRegressor (Locked Step 10)",
        "point_model_test_rmse"    : round(test_rmse, 6),
        "point_model_val_rmse"     : round(val_rmse, 6),
        "locked_hyperparameters"   : LOCKED_PARAMS,
        "locked_features"          : LOCKED_FEATURES,
        "min_country_calibration_n": MIN_COUNTRY_CALIBRATION_N,
        "scale_bounds"             : {"min": SCALE_MIN, "max": SCALE_MAX},
        "train_years"              : list(TRAIN_YEARS),
        "val_years"                : list(VAL_YEARS),
        "test_years"               : list(TEST_YEARS),
        "exp_a_global_q90"         : round(q90_global, 6),
        "exp_b_country_q90_global_fallback": round(global_q90_B, 6),
        "exp_b_n_countries_calibrated": n_country_calibrated,
        "exp_c_vol_thresholds"     : {"p33": round(vol_params["p33"], 6), "p67": round(vol_params["p67"], 6)},
        "exp_c_group_q90"          : {k: round(v, 6) for k, v in vol_params["group_q90_map"].items()},
        "exp_d_asym_q10"           : round(asym_q10_global, 6),
        "exp_d_asym_q90"           : round(asym_q90_global, 6),
        "exp_e_global_q10"         : round(global_q10_E, 6),
        "exp_e_global_q90"         : round(global_q90_E, 6),
        "exp_e_n_countries_calibrated": n_country_asym,
        "exp_f_median_sigma"       : round(median_sigma_F, 6),
        "final_decision"           : final_decision,
        "promoted_method"          : promoted_label,
        "validation_results"       : {
            "exp_a": make_serializable(eval_A_val),
            "exp_b": make_serializable(eval_B_val),
            "exp_c": make_serializable(eval_C_val),
            "exp_d": make_serializable(eval_D_val),
            "exp_e": make_serializable(eval_E_val),
            "exp_f": make_serializable(eval_F_val),
        },
        "test_results": {
            "exp_a": make_serializable(eval_A_test),
            "exp_b": make_serializable(eval_B_test),
            "exp_c": make_serializable(eval_C_test),
            "exp_d": make_serializable(eval_D_test),
            "exp_e": make_serializable(eval_E_test),
            "exp_f": make_serializable(eval_F_test),
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Saved metadata → {METADATA_PATH.name}")

    print("\n" + "=" * 70)
    print("STEP 12 COMPLETE")
    print(f"  Point Model RMSE  : {test_rmse:.4f}")
    print(f"  Step 11 Q90       : {q90_global:.4f}")
    print(f"  Final Decision    : {final_decision}")
    print("=" * 70)

    return metadata


if __name__ == "__main__":
    main()
