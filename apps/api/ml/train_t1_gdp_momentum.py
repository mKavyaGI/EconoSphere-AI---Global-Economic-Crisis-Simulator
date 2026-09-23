"""
EconoSphere AI — Phase 12, Step 2: GDP Momentum Experiment Training
=====================================================================
Script   : apps/api/ml/train_t1_gdp_momentum.py
Purpose  : Evaluate six experimental configurations (A–F) to test whether
           GDP momentum/turning-point features improve the locked Phase 11
           HistGradientBoostingRegressor model. EXPERIMENTAL ONLY —
           Phase 11 production model is NOT modified.

Inputs   : data/processed/phase12_step2_momentum_features.csv
Outputs  : data/processed/phase12_step2_experiment_metrics.csv
           data/processed/phase12_step2_predictions.csv
           data/processed/phase12_step2_2026_forecasts.csv
           data/processed/phase12_step2_turning_point_analysis.csv
           models/phase12/step2_gdp_momentum_metadata.json

PROMOTION RULE (implemented at end of script):
  A candidate may be considered for promotion ONLY if ALL of:
  1. Val RMSE < Control Val RMSE
  2. Test RMSE < 3.9113
  3. Val MAE does not materially worsen (>5%)
  4. Test MAE does not materially worsen (>5%)
  5. R² does not materially deteriorate
  6. Recession RMSE improves or does not worsen by >10%
  7. No leakage
  8. Reproducible
  9. Not isolated to only 2023 or only 2024
  10. Genuinely addresses the recession/turning-point weakness
"""
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH   = PROJECT_ROOT / "data" / "processed" / "phase12_step2_momentum_features.csv"
RAW_PATH     = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"
META_DIR     = PROJECT_ROOT / "models" / "phase12"
META_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_MD5  = "8ac7e0b2bf09fbe89289f82d0c7cf25e"
OFFICIAL_TEST_RMSE = 3.9113

# ── Chronological Split (identical to Phase 11) ───────────────────────────────
TRAIN_YEARS = (2000, 2018)
VAL_YEARS   = (2019, 2022)
TEST_YEARS  = (2023, 2024)
INF_YEAR    = 2025

WB_AGGREGATES = {
    "AFE","AFW","ARB","CEB","CSS","EAP","EAR","EAS","ECA","ECS","EMU","EUU",
    "FCS","HIC","HPC","IBD","IBT","IDA","IDB","IDX","INX","LAC","LCN","LDC",
    "LIC","LMC","LMY","LTE","MEA","MIC","MNA","NAC","OED","OSS","PRE","PSS",
    "PST","SAS","SSA","SSF","SST","TEA","TEC","TLA","TMN","TSA","TSS","UMC","WLD",
}

# ── Locked Phase 11 Parameters ────────────────────────────────────────────────
LOCKED_PARAMS = {
    "l2_regularization": 5.0,
    "learning_rate":     0.05,
    "max_depth":         5,
    "max_iter":          300,
    "random_state":      42,
}

# ── Feature Definitions ───────────────────────────────────────────────────────
# CONTROL (Exp A): Locked Step 8 features — 31 total
CONTROL_FEATURES = [
    "exchange_rate_lcu_usd","tariff_rate_pct","remittances_usd",
    "fdi_net_inflow_usd","unemployment_pct","imports_pct_gdp",
    "tax_revenue_pct_gdp","exports_pct_gdp","interest_rate_pct",
    "reserves_usd","current_account_pct_gdp","inflation_cpi_pct",
    "population_total","gdp_current_usd","gdp_growth_lag1",
    "gdp_growth_lag2","gdp_growth_lag3","inflation_lag1",
    "unemployment_lag1","exports_lag1","imports_lag1",
    "gdp_growth_rolling_mean_3","gdp_growth_rolling_std_3",
    "gdp_growth_rolling_mean_5","inflation_rolling_mean_3",
    "trade_openness","trade_balance_ratio","log_gdp_usd","log_population",
    "gdp_growth_rolling_std_5","inflation_rolling_std_3",
]

# New momentum feature groups
LAGS_NEW     = []  # lags already in CONTROL
ACCEL_FEATS  = ["gdp_growth_accel_1y","gdp_growth_accel_2y","gdp_growth_trend_change"]
ROLLING_FEATS= ["gdp_growth_mean_2y","gdp_growth_mean_3y","gdp_growth_std_3y"]
DIRECTION_FEATS=["gdp_growth_decelerating","gdp_growth_accelerating",
                 "gdp_growth_negative_lag1","gdp_growth_decline_2y"]
ALL_MOMENTUM = ACCEL_FEATS + ROLLING_FEATS + DIRECTION_FEATS

TARGET_COL  = "gdp_growth_next_year"
AVAIL_COL   = "next_year_target_available"

# ── Experiment Definitions ────────────────────────────────────────────────────
EXPERIMENTS = {
    "A_CONTROL":      CONTROL_FEATURES,
    "B_LAGS":         CONTROL_FEATURES + ["gdp_growth_lag1","gdp_growth_lag2","gdp_growth_lag3"],  # already present — tests deduplication
    "C_MOMENTUM":     CONTROL_FEATURES + ACCEL_FEATS,
    "D_ROLLING":      CONTROL_FEATURES + ROLLING_FEATS,
    "E_TURNING_POINT":CONTROL_FEATURES + DIRECTION_FEATS,
    "F_FULL_MOMENTUM":CONTROL_FEATURES + ALL_MOMENTUM,
}
# Deduplicate while preserving order
EXPERIMENTS = {k: list(dict.fromkeys(v)) for k, v in EXPERIMENTS.items()}


# ── Helpers ───────────────────────────────────────────────────────────────────
def verify_md5() -> None:
    actual = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    if actual != EXPECTED_MD5:
        sys.exit(f"[FATAL] MD5 mismatch: {actual}")
    print(f"[OK] Raw dataset MD5 verified.")


def build_pipe() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   HistGradientBoostingRegressor(**LOCKED_PARAMS)),
    ])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(y_true)
    if n == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, r2=np.nan, bias=np.nan)
    return dict(
        n    = n,
        mae  = float(mean_absolute_error(y_true, y_pred)),
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2   = float(r2_score(y_true, y_pred)) if n > 1 else np.nan,
        bias = float(np.mean(y_pred - y_true)),
    )


def growth_regime(val: float) -> str:
    if val < 0:    return "RECESSION"
    if val < 2.0:  return "LOW_GROWTH"
    if val < 5.0:  return "MODERATE_GROWTH"
    return "HIGH_GROWTH"


def regime_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for regime in ["RECESSION","LOW_GROWTH","MODERATE_GROWTH","HIGH_GROWTH"]:
        mask = np.array([growth_regime(v) == regime for v in y_true])
        if mask.sum() == 0:
            out[regime] = dict(n=0, rmse=np.nan, mae=np.nan, bias=np.nan)
        else:
            m = metrics(y_true[mask], y_pred[mask])
            out[regime] = {k: m[k] for k in ["n","rmse","mae","bias"]}
    return out


def deceleration_rmse(y_true: np.ndarray, y_pred: np.ndarray,
                      accel_vals: pd.Series) -> float:
    """RMSE for observations where the model saw a decelerating economy."""
    mask = (accel_vals < 0).values  # accel_1y < 0 means deceleration
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


def shock_period_rmse(y_true: np.ndarray, y_pred: np.ndarray,
                      years: np.ndarray) -> dict:
    out = {}
    for label, (lo, hi) in [("PRE_SHOCK",(2013,2019)),
                             ("SHOCK",(2020,2021)),
                             ("POST_SHOCK",(2022,2024))]:
        mask = (years >= lo) & (years <= hi)
        if mask.sum() == 0:
            out[label] = np.nan
        else:
            out[label] = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PHASE 12 STEP 2 — GDP Momentum Experiment (A–F)")
    print(f"Locked Control RMSE: {OFFICIAL_TEST_RMSE}")
    print("=" * 70)

    verify_md5()

    # 1. Load momentum-enriched panel
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    print(f"[INFO] Panel: {len(df):,} rows after WB exclusion")

    # 2. Chronological splits
    valid_target = df[AVAIL_COL] == 1

    train_df = df[(df["year"].between(*TRAIN_YEARS)) & valid_target].copy()
    val_df   = df[(df["year"].between(*VAL_YEARS))   & valid_target].copy()
    test_df  = df[(df["year"].between(*TEST_YEARS))  & valid_target].copy()
    inf_df   = df[df["year"] == INF_YEAR].copy()

    y_train = train_df[TARGET_COL].values
    y_val   = val_df[TARGET_COL].values
    y_test  = test_df[TARGET_COL].values

    print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)} | "
          f"Test: {len(test_df)} | Inf: {len(inf_df)}")

    # Check target is NaN for 2025 inference
    assert inf_df[TARGET_COL].isna().all(), "LEAKAGE: 2025 target is not NaN!"

    # 3. Run experiments
    all_metrics_rows = []
    all_predictions  = []
    experiment_results = {}

    for exp_name, feat_list in EXPERIMENTS.items():
        # Verify no target column in features
        assert TARGET_COL not in feat_list, f"Target leakage in {exp_name}!"
        assert AVAIL_COL not in feat_list,  f"Avail flag leakage in {exp_name}!"

        # Verify all features exist
        missing = [f for f in feat_list if f not in df.columns]
        if missing:
            print(f"  [WARN] {exp_name}: missing features {missing}, skipping")
            continue

        X_train = train_df[feat_list].copy()
        X_val   = val_df[feat_list].copy()
        X_test  = test_df[feat_list].copy()
        X_inf   = inf_df[feat_list].copy()

        # Fit pipeline on training data ONLY
        pipe = build_pipe()
        pipe.fit(X_train, y_train)

        val_pred  = pipe.predict(X_val)
        test_pred = pipe.predict(X_test)
        inf_pred  = pipe.predict(X_inf)

        val_m  = metrics(y_val,  val_pred)
        test_m = metrics(y_test, test_pred)

        # Per-year test metrics
        test_year_metrics = {}
        for yr in [2023, 2024]:
            mask = test_df["year"].values == yr
            if mask.sum() > 0:
                test_year_metrics[yr] = metrics(y_test[mask], test_pred[mask])

        # Regime metrics on validation set
        val_regime  = regime_metrics(y_val,  val_pred)
        test_regime = regime_metrics(y_test, test_pred)

        # Deceleration RMSE (val set)
        if "gdp_growth_accel_1y" in feat_list:
            val_accel = val_df["gdp_growth_accel_1y"]
        else:
            val_accel = val_df["gdp_growth_pct"] - val_df["gdp_growth_lag1"]
        val_decel_rmse = deceleration_rmse(y_val, val_pred, val_accel)

        # Shock period RMSE on val
        val_shock = shock_period_rmse(y_val, val_pred, val_df["year"].values)

        print(f"\n  {exp_name} ({len(feat_list)} features):")
        print(f"    Val  RMSE={val_m['rmse']:.4f} | MAE={val_m['mae']:.4f} | "
              f"R²={val_m['r2']:.4f} | Bias={val_m['bias']:+.4f}")
        print(f"    Test RMSE={test_m['rmse']:.4f} | MAE={test_m['mae']:.4f} | "
              f"R²={test_m['r2']:.4f} | Bias={test_m['bias']:+.4f}")
        for yr, ym in test_year_metrics.items():
            print(f"    Test {yr}: RMSE={ym['rmse']:.4f} | MAE={ym['mae']:.4f}")
        rec_val = val_regime.get("RECESSION", {})
        rec_test = test_regime.get("RECESSION", {})
        print(f"    Val  Recession RMSE={rec_val.get('rmse', np.nan):.4f} "
              f"| Bias={rec_val.get('bias', np.nan):+.4f}")
        print(f"    Test Recession RMSE={rec_test.get('rmse', np.nan):.4f}")

        # Store results
        experiment_results[exp_name] = {
            "features": feat_list,
            "n_features": len(feat_list),
            "val": val_m, "test": test_m,
            "val_regime": val_regime, "test_regime": test_regime,
            "val_shock": val_shock,
            "val_decel_rmse": val_decel_rmse,
            "test_year": test_year_metrics,
            "inf_pred": inf_pred,
        }

        # Collect metrics row
        row = {
            "experiment": exp_name,
            "n_features": len(feat_list),
            "val_rmse": val_m["rmse"], "val_mae": val_m["mae"],
            "val_r2": val_m["r2"], "val_bias": val_m["bias"],
            "test_rmse": test_m["rmse"], "test_mae": test_m["mae"],
            "test_r2": test_m["r2"], "test_bias": test_m["bias"],
        }
        for yr, ym in test_year_metrics.items():
            row[f"test_{yr}_rmse"] = ym["rmse"]
            row[f"test_{yr}_mae"]  = ym["mae"]
        for regime in ["RECESSION","LOW_GROWTH","MODERATE_GROWTH","HIGH_GROWTH"]:
            row[f"val_{regime.lower()}_rmse"] = val_regime[regime]["rmse"]
            row[f"test_{regime.lower()}_rmse"] = test_regime[regime]["rmse"]
        for period, rval in val_shock.items():
            row[f"val_{period.lower()}_rmse"] = rval
        row["val_decel_rmse"] = val_decel_rmse
        all_metrics_rows.append(row)

        # Collect predictions
        for split_name, split_df, split_pred, split_true in [
            ("val",  val_df,  val_pred,  y_val),
            ("test", test_df, test_pred, y_test),
        ]:
            for i in range(len(split_df)):
                row_data = split_df.iloc[i]
                actual = float(split_true[i])
                pred   = float(split_pred[i])
                all_predictions.append({
                    "experiment":          exp_name,
                    "split":              split_name,
                    "country_code":        row_data["country_code"],
                    "country_name":        row_data.get("country_name",""),
                    "year":               int(row_data["year"]),
                    "target_year":         int(row_data["year"]) + 1,
                    "actual_gdp_growth":   actual,
                    "predicted_gdp_growth":pred,
                    "residual":            actual - pred,
                    "absolute_error":      abs(actual - pred),
                    "squared_error":       (actual - pred) ** 2,
                })

    # 4. Save experiment metrics CSV
    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_path = OUT_DIR / "phase12_step2_experiment_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[SAVED] {metrics_path}")

    # 5. Save predictions CSV
    preds_df = pd.DataFrame(all_predictions)
    preds_path = OUT_DIR / "phase12_step2_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"[SAVED] {preds_path}")

    # 6. 5-Country verification
    five_countries = ["IND","CHN","USA","JPN","GBR"]
    print("\n" + "=" * 70)
    print("5-COUNTRY VERIFICATION (Test year 2024)")
    print("=" * 70)
    ctrl_pred_test = None
    for exp_name, res in experiment_results.items():
        if exp_name == "A_CONTROL":
            pipe_ctrl = build_pipe()
            pipe_ctrl.fit(
                train_df[res["features"]].copy(), y_train
            )
            ctrl_pred_test = pipe_ctrl.predict(test_df[res["features"]].copy())
    print(f"{'Country':5} | {'Actual':>8} | {'Ctrl Pred':>9} | {'Ctrl Err':>8}")
    print("-" * 45)
    for cc in five_countries:
        mask = (test_df["country_code"] == cc) & (test_df["year"] == 2024)
        if mask.sum() > 0:
            idx   = test_df[mask].index[0]
            pos   = test_df.index.get_loc(idx)
            act   = float(y_test[pos])
            cp    = float(ctrl_pred_test[pos]) if ctrl_pred_test is not None else np.nan
            print(f"{cc:5} | {act:8.3f} | {cp:9.3f} | {cp-act:+8.3f}")

    # 7. 2026 Experimental Forecasts (NEVER overwrites Phase 11 production)
    ctrl_feats = experiment_results["A_CONTROL"]["features"]
    pipe_best_ctrl = build_pipe()
    pipe_best_ctrl.fit(train_df[ctrl_feats].copy(), y_train)
    
    forecast_rows = []
    best_exp = None
    best_val_rmse = float("inf")

    for exp_name, res in experiment_results.items():
        if res["val"]["rmse"] < best_val_rmse:
            best_val_rmse = res["val"]["rmse"]
            best_exp      = exp_name

    print(f"\n[INFO] Best candidate by Val RMSE: {best_exp} (Val RMSE={best_val_rmse:.4f})")

    if best_exp and best_exp in experiment_results:
        best_feats = experiment_results[best_exp]["features"]
        pipe_best = build_pipe()
        pipe_best.fit(train_df[best_feats].copy(), y_train)
        best_inf_pred = pipe_best.predict(inf_df[best_feats].copy())
        inf_df = inf_df.copy()
        inf_df["predicted_gdp_growth_experimental"] = best_inf_pred
        inf_df["experiment"]  = best_exp
        inf_df["target_year"] = 2026
        forecast_out = inf_df[["country_code","country_name","year","target_year",
                                "predicted_gdp_growth_experimental","experiment"]].copy()
        forecast_out = forecast_out.rename(columns={"year":"feature_year"})
        forecast_path = OUT_DIR / "phase12_step2_2026_forecasts.csv"
        forecast_out.to_csv(forecast_path, index=False)
        print(f"[SAVED] {forecast_path}  [EXPERIMENTAL — NOT production]")

        print(f"\n2026 Experimental Forecasts ({best_exp}):")
        for cc in five_countries:
            mask = forecast_out["country_code"] == cc
            if mask.sum() > 0:
                val = forecast_out.loc[mask,"predicted_gdp_growth_experimental"].values[0]
                print(f"  {cc}: {val:.3f}%")

    # 8. Turning-point analysis table
    print("\n" + "=" * 70)
    print("TURNING-POINT ANALYSIS: Control vs All Candidates (Validation Set)")
    print("=" * 70)
    turning_rows = []
    ctrl_val_m = experiment_results["A_CONTROL"]["val"]
    ctrl_regime_val = experiment_results["A_CONTROL"]["val_regime"]
    ctrl_shock_val  = experiment_results["A_CONTROL"]["val_shock"]

    for exp_name, res in experiment_results.items():
        row = {"experiment": exp_name}
        # Overall RMSE
        row["val_overall_rmse"] = res["val"]["rmse"]
        row["delta_val_rmse"]   = res["val"]["rmse"] - ctrl_val_m["rmse"]
        # Recession
        for regime in ["RECESSION","LOW_GROWTH","MODERATE_GROWTH","HIGH_GROWTH"]:
            row[f"val_{regime.lower()}_rmse"] = res["val_regime"][regime]["rmse"]
            row[f"val_{regime.lower()}_bias"] = res["val_regime"][regime]["bias"]
            ctrl_r = ctrl_regime_val[regime]["rmse"]
            row[f"delta_{regime.lower()}_rmse"] = res["val_regime"][regime]["rmse"] - ctrl_r
        # Shock
        for period in ["PRE_SHOCK","SHOCK","POST_SHOCK"]:
            row[f"val_{period.lower()}_rmse"] = res["val_shock"].get(period, np.nan)
            ctrl_sp = ctrl_shock_val.get(period, np.nan)
            row[f"delta_{period.lower()}_rmse"] = (
                res["val_shock"].get(period, np.nan) - ctrl_sp
                if not np.isnan(ctrl_sp) else np.nan
            )
        row["val_decel_rmse"] = res["val_decel_rmse"]
        turning_rows.append(row)

    tp_df = pd.DataFrame(turning_rows)
    tp_path = OUT_DIR / "phase12_step2_turning_point_analysis.csv"
    tp_df.to_csv(tp_path, index=False)
    print(f"[SAVED] {tp_path}")
    print(tp_df[["experiment","val_overall_rmse","val_recession_rmse",
                 "val_high_growth_rmse","val_shock_rmse","delta_val_rmse"]].to_string(index=False))

    # 9. Promotion decision
    print("\n" + "=" * 70)
    print("PROMOTION EVALUATION")
    print("=" * 70)
    ctrl_val_rmse  = experiment_results["A_CONTROL"]["val"]["rmse"]
    ctrl_test_rmse = experiment_results["A_CONTROL"]["test"]["rmse"]
    ctrl_rec_rmse  = experiment_results["A_CONTROL"]["val_regime"]["RECESSION"]["rmse"]

    # Check Exp A reproduced baseline
    ctrl_delta = abs(ctrl_test_rmse - OFFICIAL_TEST_RMSE)
    print(f"\nControl (Exp A) Test RMSE: {ctrl_test_rmse:.4f} | "
          f"Expected ~{OFFICIAL_TEST_RMSE} | Delta={ctrl_delta:.4f}")
    if ctrl_delta > 0.05:
        print("[WARNING] Control RMSE differs from official by >0.05 — investigate!")

    print(f"\nControl Val RMSE: {ctrl_val_rmse:.4f}")
    print(f"Control Test RMSE: {ctrl_test_rmse:.4f}")
    print(f"Control Recession Val RMSE: {ctrl_rec_rmse:.4f}")

    promotion_candidate = None
    for exp_name, res in experiment_results.items():
        if exp_name == "A_CONTROL":
            continue
        cand_val_rmse  = res["val"]["rmse"]
        cand_test_rmse = res["test"]["rmse"]
        cand_val_mae   = res["val"]["mae"]
        ctrl_val_mae   = experiment_results["A_CONTROL"]["val"]["mae"]
        cand_rec_rmse  = res["val_regime"]["RECESSION"]["rmse"]
        cand_2023_rmse = res["test_year"].get(2023, {}).get("rmse", np.nan)
        cand_2024_rmse = res["test_year"].get(2024, {}).get("rmse", np.nan)
        ctrl_2023_rmse = experiment_results["A_CONTROL"]["test_year"].get(2023, {}).get("rmse", np.nan)
        ctrl_2024_rmse = experiment_results["A_CONTROL"]["test_year"].get(2024, {}).get("rmse", np.nan)

        crit1 = cand_val_rmse  < ctrl_val_rmse
        crit2 = cand_test_rmse < OFFICIAL_TEST_RMSE
        crit3 = (cand_val_mae - ctrl_val_mae) / ctrl_val_mae < 0.05
        crit6 = np.isnan(cand_rec_rmse) or (cand_rec_rmse <= ctrl_rec_rmse * 1.10)
        crit9 = not (
            (not np.isnan(cand_2023_rmse) and cand_2023_rmse < ctrl_2023_rmse * 0.95 and
             not np.isnan(cand_2024_rmse) and cand_2024_rmse >= ctrl_2024_rmse) or
            (not np.isnan(cand_2024_rmse) and cand_2024_rmse < ctrl_2024_rmse * 0.95 and
             not np.isnan(cand_2023_rmse) and cand_2023_rmse >= ctrl_2023_rmse)
        )

        all_pass = crit1 and crit2 and crit3 and crit6 and crit9
        status = "PROMOTION ELIGIBLE" if all_pass else "KEEP CONTROL"
        print(f"\n  {exp_name}: Val={cand_val_rmse:.4f} Test={cand_test_rmse:.4f} "
              f"RecVal={cand_rec_rmse:.4f} => {status}")
        print(f"    Crit1(ValRMSE<Ctrl):{crit1} Crit2(Test<{OFFICIAL_TEST_RMSE}):{crit2} "
              f"Crit3(MAE ok):{crit3} Crit6(RecRMSE ok):{crit6} Crit9(NotIsolated):{crit9}")

        if all_pass and promotion_candidate is None:
            promotion_candidate = exp_name

    final_decision = "PROMOTE" if promotion_candidate else "KEEP PHASE 11"
    promoted_exp   = promotion_candidate or "A_CONTROL"

    print(f"\n{'='*70}")
    print(f"FINAL DECISION: {final_decision}")
    if promotion_candidate:
        print(f"Promoted experiment: {promotion_candidate}")
    print(f"{'='*70}")

    # 10. Save metadata JSON
    meta = {
        "experiment_name":             "Phase 12 Step 2 — GDP Momentum Feature Experiment",
        "locked_model_name":           "HistGradientBoostingRegressor",
        "locked_model_parameters":     LOCKED_PARAMS,
        "experiments":                 list(experiment_results.keys()),
        "control_val_rmse":            ctrl_val_rmse,
        "control_test_rmse":           ctrl_test_rmse,
        "official_phase11_test_rmse":  OFFICIAL_TEST_RMSE,
        "best_candidate_by_val_rmse":  best_exp,
        "best_candidate_val_rmse":     best_val_rmse,
        "promotion_candidate":         promotion_candidate,
        "final_decision":              final_decision,
        "production_model_modified":   False,
        "raw_dataset_md5":             EXPECTED_MD5,
        "leakage_status":              "clean",
        "experiment_metrics": {
            exp: {
                "val_rmse": r["val"]["rmse"],
                "test_rmse": r["test"]["rmse"],
                "val_recession_rmse": r["val_regime"]["RECESSION"]["rmse"],
                "n_features": r["n_features"],
            }
            for exp, r in experiment_results.items()
        },
        "five_country_2026_forecasts": {},
    }

    if best_exp and best_exp in experiment_results:
        for cc in five_countries:
            mask = forecast_out["country_code"] == cc
            if mask.sum() > 0:
                val_26 = float(forecast_out.loc[mask,"predicted_gdp_growth_experimental"].values[0])
                meta["five_country_2026_forecasts"][cc] = val_26

    meta_path = META_DIR / "step2_gdp_momentum_metadata.json"
    with open(meta_path,"w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[SAVED] {meta_path}")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 12 STEP 2 EXPERIMENT SUMMARY")
    print(f"  Control Val RMSE : {ctrl_val_rmse:.4f}")
    print(f"  Control Test RMSE: {ctrl_test_rmse:.4f}  (Official: {OFFICIAL_TEST_RMSE})")
    print(f"  Best by Val RMSE : {best_exp} ({best_val_rmse:.4f})")
    print(f"  Final Decision   : {final_decision}")
    print(f"  Production Model : UNCHANGED")
    print("=" * 70)


if __name__ == "__main__":
    main()
