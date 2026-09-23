"""
EconoSphere AI — Phase 12, Step 1: Forecast Error Diagnosis
=============================================================
Script   : apps/api/ml/analyze_t1_forecast_errors.py
Purpose  : Comprehensive post-hoc diagnostic analysis of walk-forward prediction
           errors. All analysis is strictly post-hoc; regime labels and
           missingness indicators are NEVER introduced as model features.
           This script is DIAGNOSTIC ONLY. Phase 11 production is unchanged.

Inputs   : data/processed/phase12_walk_forward_predictions.csv
           data/processed/master_panel_t1_missingness.csv  (for missingness data)

Outputs  : data/processed/phase12_country_error_analysis.csv
           data/processed/phase12_growth_regime_errors.csv
           data/processed/phase12_largest_forecast_errors.csv
           data/processed/phase12_period_robustness.csv
           data/processed/phase12_missingness_error_analysis.csv
           models/phase12/step1_walk_forward_metadata.json
"""
import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PREDS_PATH   = PROJECT_ROOT / "data" / "processed" / "phase12_walk_forward_predictions.csv"
PANEL_PATH   = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"
META_DIR     = PROJECT_ROOT / "models" / "phase12"
META_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH     = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"

EXPECTED_MD5 = "8ac7e0b2bf09fbe89289f82d0c7cf25e"
MIN_COUNTRY_OBS = 5          # minimum predictions for country-level reporting

# Locked model info (for metadata output)
LOCKED_PARAMS = {
    "l2_regularization": 5.0,
    "learning_rate":     0.05,
    "max_depth":         5,
    "max_iter":          300,
    "random_state":      42,
}
LOCKED_FEATURES = [
    "exchange_rate_lcu_usd", "tariff_rate_pct", "remittances_usd",
    "fdi_net_inflow_usd", "unemployment_pct", "imports_pct_gdp",
    "tax_revenue_pct_gdp", "exports_pct_gdp", "interest_rate_pct",
    "reserves_usd", "current_account_pct_gdp", "inflation_cpi_pct",
    "population_total", "gdp_current_usd", "gdp_growth_lag1",
    "gdp_growth_lag2", "gdp_growth_lag3", "inflation_lag1",
    "unemployment_lag1", "exports_lag1", "imports_lag1",
    "gdp_growth_rolling_mean_3", "gdp_growth_rolling_std_3",
    "gdp_growth_rolling_mean_5", "inflation_rolling_mean_3",
    "trade_openness", "trade_balance_ratio", "log_gdp_usd",
    "log_population", "gdp_growth_rolling_std_5", "inflation_rolling_std_3",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def verify_md5():
    actual = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    assert actual == EXPECTED_MD5, f"MD5 mismatch: {actual}"
    print(f"[OK] Raw dataset MD5 verified: {actual}")


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> dict:
    if len(y_true) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "bias": np.nan}
    return {
        "n":    n,
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "bias": float(np.mean(y_pred - y_true)),
    }


def growth_regime(val: float) -> str:
    if val < 0:    return "RECESSION"
    if val < 2.0:  return "LOW_GROWTH"
    if val < 5.0:  return "MODERATE_GROWTH"
    return "HIGH_GROWTH"


# ── Analysis Parts ─────────────────────────────────────────────────────────────
def analyze_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Part A — Error by target year."""
    rows = []
    for yr, grp in df.groupby("target_year"):
        m = metrics_dict(grp["actual_gdp_growth"].values, grp["predicted_gdp_growth"].values, len(grp))
        rows.append({"target_year": int(yr), **m})
    return pd.DataFrame(rows).sort_values("target_year")


def analyze_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Part B — Error by country (minimum MIN_COUNTRY_OBS predictions)."""
    rows = []
    for cc, grp in df.groupby("country_code"):
        if len(grp) < MIN_COUNTRY_OBS:
            continue
        m = metrics_dict(grp["actual_gdp_growth"].values, grp["predicted_gdp_growth"].values, len(grp))
        rows.append({
            "country_code":        cc,
            "country_name":        grp["country_name"].iloc[0] if "country_name" in grp.columns else "",
            "n_predictions":       len(grp),
            "mae":                 m["mae"],
            "rmse":                m["rmse"],
            "bias":                m["bias"],
            "mean_actual_growth":  float(grp["actual_gdp_growth"].mean()),
            "mean_predicted_growth": float(grp["predicted_gdp_growth"].mean()),
        })
    return pd.DataFrame(rows).sort_values("rmse", ascending=False)


def analyze_growth_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Part C — Error by economic growth regime (post-hoc only)."""
    df = df.copy()
    df["growth_regime"] = df["actual_gdp_growth"].apply(growth_regime)
    rows = []
    regime_order = ["RECESSION", "LOW_GROWTH", "MODERATE_GROWTH", "HIGH_GROWTH"]
    for regime in regime_order:
        grp = df[df["growth_regime"] == regime]
        m = metrics_dict(grp["actual_gdp_growth"].values, grp["predicted_gdp_growth"].values, len(grp))
        rows.append({"growth_regime": regime, **m})
    return pd.DataFrame(rows)


def analyze_period_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Part 5 — COVID/shock period robustness."""
    # Pre-shock: 2013–2019 | Shock: 2020–2021 | Post-shock: 2022–2024
    periods = {
        "PRE_SHOCK":  (df["target_year"].between(2013, 2019)),
        "SHOCK":      (df["target_year"].between(2020, 2021)),
        "POST_SHOCK": (df["target_year"].between(2022, 2024)),
    }
    rows = []
    for label, mask in periods.items():
        grp = df[mask]
        if len(grp) == 0:
            continue
        m = metrics_dict(grp["actual_gdp_growth"].values, grp["predicted_gdp_growth"].values, len(grp))
        rows.append({"period": label, **m})
    return pd.DataFrame(rows)


def analyze_missingness_vs_error(df: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Part 6 — Feature availability vs forecast error."""
    # Merge missingness ratio from panel on (country_code, feature_year)
    miss_cols = panel[["country_code", "year", "total_missing_feature_ratio", "total_missing_feature_count"]].copy()
    miss_cols = miss_cols.rename(columns={"year": "feature_year"})
    merged = df.merge(miss_cols, on=["country_code", "feature_year"], how="left")

    # Correlation between missingness ratio and absolute error
    valid = merged.dropna(subset=["total_missing_feature_ratio", "absolute_error"])
    if len(valid) > 2:
        corr, pval = pearsonr(valid["total_missing_feature_ratio"], valid["absolute_error"])
    else:
        corr, pval = np.nan, np.nan

    # Quantile-based grouping
    merged["miss_group"] = pd.qcut(
        merged["total_missing_feature_ratio"].fillna(merged["total_missing_feature_ratio"].median()),
        q=3,
        labels=["LOW_MISSINGNESS", "MEDIUM_MISSINGNESS", "HIGH_MISSINGNESS"],
        duplicates="drop",
    )

    rows = []
    for group, grp in merged.groupby("miss_group"):
        m = metrics_dict(grp["actual_gdp_growth"].values, grp["predicted_gdp_growth"].values, len(grp))
        rows.append({"missingness_group": str(group), **m})

    result_df = pd.DataFrame(rows)
    result_df["pearson_corr_missingness_abs_error"] = corr
    result_df["pearson_pvalue"] = pval
    return result_df, float(corr) if not np.isnan(corr) else None


def get_largest_errors(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Part D — Top N largest absolute errors."""
    return (
        df.nlargest(n, "absolute_error")
        [["country_code", "country_name", "feature_year", "target_year",
          "actual_gdp_growth", "predicted_gdp_growth", "residual", "absolute_error"]]
        .reset_index(drop=True)
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PHASE 12 STEP 1 — Forecast Error Diagnosis")
    print("=" * 70)

    verify_md5()

    # Load walk-forward predictions
    df = pd.read_csv(PREDS_PATH)
    panel = pd.read_csv(PANEL_PATH)
    print(f"[INFO] Walk-forward predictions: {len(df):,} rows")
    print(f"[INFO] Target years available: {sorted(df['target_year'].unique())}")

    # Sanity checks
    assert "absolute_error" in df.columns, "absolute_error column missing from predictions"
    assert "residual" in df.columns, "residual column missing"
    # Verify math
    check_abs = np.abs(df["actual_gdp_growth"] - df["predicted_gdp_growth"])
    assert np.allclose(check_abs, df["absolute_error"], atol=1e-6), "absolute_error math mismatch"
    check_sq = (df["actual_gdp_growth"] - df["predicted_gdp_growth"]) ** 2
    assert np.allclose(check_sq, df["squared_error"], atol=1e-6), "squared_error math mismatch"
    print("[OK] absolute_error and squared_error math verified")

    # ── A. Error by year
    print("\n--- A. Error by Target Year ---")
    year_df = analyze_by_year(df)
    print(year_df.to_string(index=False))
    best_yr_row  = year_df.loc[year_df["rmse"].idxmin()]
    worst_yr_row = year_df.loc[year_df["rmse"].idxmax()]
    print(f"\n  Best year  (lowest RMSE): {int(best_yr_row['target_year'])} — RMSE={best_yr_row['rmse']:.4f}")
    print(f"  Worst year (highest RMSE): {int(worst_yr_row['target_year'])} — RMSE={worst_yr_row['rmse']:.4f}")

    # ── B. Error by country
    print(f"\n--- B. Error by Country (min {MIN_COUNTRY_OBS} predictions) ---")
    country_df = analyze_by_country(df)
    print(f"  Qualifying countries: {len(country_df)}")
    print("\n  Top 10 HIGHEST-error countries:")
    print(country_df.head(10)[["country_code","country_name","n_predictions","rmse","mae","bias"]].to_string(index=False))
    print("\n  Top 10 LOWEST-error countries:")
    print(country_df.tail(10)[["country_code","country_name","n_predictions","rmse","mae","bias"]].to_string(index=False))

    # Over/under-prediction
    over  = country_df.nsmallest(5, "bias")   # most negative bias = over-prediction (pred > actual)
    under = country_df.nlargest(5, "bias")    # most positive bias = under-prediction (pred < actual)
    print("\n  Strongest OVER-prediction countries (bias = pred - actual, most negative):")
    print(over[["country_code","country_name","n_predictions","bias"]].to_string(index=False))
    print("\n  Strongest UNDER-prediction countries:")
    print(under[["country_code","country_name","n_predictions","bias"]].to_string(index=False))

    country_df.to_csv(OUT_DIR / "phase12_country_error_analysis.csv", index=False)
    print(f"[SAVED] phase12_country_error_analysis.csv")

    # ── C. Growth regime
    print("\n--- C. Error by Growth Regime (POST-HOC ONLY) ---")
    regime_df = analyze_growth_regime(df)
    print(regime_df.to_string(index=False))
    regime_df.to_csv(OUT_DIR / "phase12_growth_regime_errors.csv", index=False)
    print(f"[SAVED] phase12_growth_regime_errors.csv")

    # ── D. Error distribution
    print("\n--- D. Residual Distribution ---")
    residuals = df["residual"].values
    print(f"  Mean     : {residuals.mean():+.4f}")
    print(f"  Median   : {np.median(residuals):+.4f}")
    print(f"  Std      : {residuals.std():.4f}")
    print(f"  Min      : {residuals.min():+.4f}")
    print(f"  Max      : {residuals.max():+.4f}")
    print(f"  P5       : {np.percentile(residuals, 5):+.4f}")
    print(f"  P25      : {np.percentile(residuals, 25):+.4f}")
    print(f"  P75      : {np.percentile(residuals, 75):+.4f}")
    print(f"  P95      : {np.percentile(residuals, 95):+.4f}")

    print("\n  Top 20 Largest Absolute Errors:")
    top20 = get_largest_errors(df, n=20)
    print(top20.to_string(index=False))
    top20.to_csv(OUT_DIR / "phase12_largest_forecast_errors.csv", index=False)
    print(f"[SAVED] phase12_largest_forecast_errors.csv")

    # ── Part 5. Shock period robustness
    print("\n--- E. COVID / Shock Period Robustness ---")
    period_df = analyze_period_robustness(df)
    print(period_df.to_string(index=False))
    period_df.to_csv(OUT_DIR / "phase12_period_robustness.csv", index=False)
    print(f"[SAVED] phase12_period_robustness.csv")

    # ── Part 6. Missingness vs error
    print("\n--- F. Feature Missingness vs Forecast Error ---")
    miss_df, miss_corr = analyze_missingness_vs_error(df, panel)
    print(miss_df.to_string(index=False))
    print(f"\n  Pearson correlation (missingness ratio vs absolute error): {miss_corr}")
    miss_df.to_csv(OUT_DIR / "phase12_missingness_error_analysis.csv", index=False)
    print(f"[SAVED] phase12_missingness_error_analysis.csv")

    # ── Part 8. Step 1 metadata JSON
    all_sq_errors = df["squared_error"].values
    overall_rmse  = float(np.sqrt(np.mean(all_sq_errors)))
    overall_mae   = float(df["absolute_error"].mean())

    # Determine shock period RMSEs for metadata
    def period_rmse(target_min, target_max):
        sub = df[df["target_year"].between(target_min, target_max)]
        if len(sub) == 0:
            return None
        return float(np.sqrt(np.mean(sub["squared_error"].values)))

    # Highest/lowest error country (only if statistically supported — >=5 obs)
    if len(country_df) > 0:
        highest_error_cc = country_df.iloc[0]["country_code"]
        over_pred_cc     = over.iloc[0]["country_code"] if len(over) > 0 else None
        under_pred_cc    = under.iloc[0]["country_code"] if len(under) > 0 else None
    else:
        highest_error_cc = over_pred_cc = under_pred_cc = None

    miss_rel = "positive" if (miss_corr is not None and miss_corr > 0.05) else (
               "negative" if (miss_corr is not None and miss_corr < -0.05) else "negligible")

    meta = {
        "experiment_name":              "Phase 12 Step 1 — Walk-Forward Validation",
        "locked_model_name":            "HistGradientBoostingRegressor",
        "locked_model_parameters":      LOCKED_PARAMS,
        "feature_count":                len(LOCKED_FEATURES),
        "feature_names":                LOCKED_FEATURES,
        "number_of_walk_forward_windows": int(df["window_id"].nunique()),
        "total_predictions":            int(len(df)),
        "overall_walk_forward_rmse":    overall_rmse,
        "overall_walk_forward_mae":     overall_mae,
        "best_target_year":             int(best_yr_row["target_year"]),
        "worst_target_year":            int(worst_yr_row["target_year"]),
        "best_year_rmse":               float(best_yr_row["rmse"]),
        "worst_year_rmse":              float(worst_yr_row["rmse"]),
        "highest_error_country":        highest_error_cc,
        "strongest_overprediction_country":  over_pred_cc,
        "strongest_underprediction_country": under_pred_cc,
        "missingness_error_relationship":    miss_rel,
        "missingness_pearson_corr":          miss_corr,
        "pre_shock_rmse":               period_rmse(2013, 2019),
        "shock_rmse":                   period_rmse(2020, 2021),
        "post_shock_rmse":              period_rmse(2022, 2024),
        "raw_dataset_md5":              EXPECTED_MD5,
        "leakage_status":               "clean — target excluded from features; imputer fitted on train only; no random splits",
        "production_model_modified":    False,
        "official_phase11_test_rmse":   3.9113,
        "phase12_step1_status":         "DIAGNOSTIC COMPLETE",
    }

    meta_path = META_DIR / "step1_walk_forward_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n[SAVED] {meta_path}")

    print("\n" + "=" * 70)
    print("PHASE 12 STEP 1 DIAGNOSIS COMPLETE")
    print(f"  Overall Walk-Forward RMSE : {overall_rmse:.4f}")
    print(f"  Overall Walk-Forward MAE  : {overall_mae:.4f}")
    print(f"  Official Phase 11 RMSE    : 3.9113  (LOCKED — UNCHANGED)")
    print(f"  Status                    : DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
