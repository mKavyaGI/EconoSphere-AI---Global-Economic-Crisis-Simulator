"""
EconoSphere AI — Phase 12, Step 2: Walk-Forward Confirmation
==============================================================
Script   : apps/api/ml/evaluate_t1_gdp_momentum_walk_forward.py
Purpose  : Expanding-window walk-forward comparison between the locked
           Phase 11 control model (Exp A) and the best GDP momentum
           candidate from the Phase 12 Step 2 experiment.

           This script is DIAGNOSTIC ONLY. It does NOT modify the
           production model or any Phase 11 artifacts.

Inputs   : data/processed/phase12_step2_momentum_features.csv
           models/phase12/step2_gdp_momentum_metadata.json  (reads best candidate)

Outputs  : data/processed/phase12_step2_walk_forward_metrics.csv
           data/processed/phase12_step2_walk_forward_predictions.csv
"""
import hashlib
import json
import math
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
META_PATH    = PROJECT_ROOT / "models" / "phase12" / "step2_gdp_momentum_metadata.json"
RAW_PATH     = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
OUT_DIR      = PROJECT_ROOT / "data" / "processed"

EXPECTED_MD5 = "8ac7e0b2bf09fbe89289f82d0c7cf25e"

WB_AGGREGATES = {
    "AFE","AFW","ARB","CEB","CSS","EAP","EAR","EAS","ECA","ECS","EMU","EUU",
    "FCS","HIC","HPC","IBD","IBT","IDA","IDB","IDX","INX","LAC","LCN","LDC",
    "LIC","LMC","LMY","LTE","MEA","MIC","MNA","NAC","OED","OSS","PRE","PSS",
    "PST","SAS","SSA","SSF","SST","TEA","TEC","TLA","TMN","TSA","TSS","UMC","WLD",
}

LOCKED_PARAMS = {
    "l2_regularization": 5.0, "learning_rate": 0.05,
    "max_depth": 5, "max_iter": 300, "random_state": 42,
}

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

ALL_MOMENTUM = [
    "gdp_growth_accel_1y","gdp_growth_accel_2y","gdp_growth_trend_change",
    "gdp_growth_mean_2y","gdp_growth_mean_3y","gdp_growth_std_3y",
    "gdp_growth_decelerating","gdp_growth_accelerating",
    "gdp_growth_negative_lag1","gdp_growth_decline_2y",
]

TARGET_COL = "gdp_growth_next_year"
AVAIL_COL  = "next_year_target_available"

WINDOWS = [
    {"window_id": 1,  "eval_feature_year": 2012, "eval_target_year": 2013},
    {"window_id": 2,  "eval_feature_year": 2013, "eval_target_year": 2014},
    {"window_id": 3,  "eval_feature_year": 2014, "eval_target_year": 2015},
    {"window_id": 4,  "eval_feature_year": 2015, "eval_target_year": 2016},
    {"window_id": 5,  "eval_feature_year": 2016, "eval_target_year": 2017},
    {"window_id": 6,  "eval_feature_year": 2017, "eval_target_year": 2018},
    {"window_id": 7,  "eval_feature_year": 2018, "eval_target_year": 2019},
    {"window_id": 8,  "eval_feature_year": 2019, "eval_target_year": 2020},
    {"window_id": 9,  "eval_feature_year": 2020, "eval_target_year": 2021},
    {"window_id": 10, "eval_feature_year": 2021, "eval_target_year": 2022},
    {"window_id": 11, "eval_feature_year": 2022, "eval_target_year": 2023},
    {"window_id": 12, "eval_feature_year": 2023, "eval_target_year": 2024},
]

TRAIN_START = 2000


def build_pipe() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   HistGradientBoostingRegressor(**LOCKED_PARAMS)),
    ])


def calc_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, r2=np.nan, bias=np.nan)
    return dict(
        n    = len(y_true),
        mae  = float(mean_absolute_error(y_true, y_pred)),
        rmse = float(math.sqrt(mean_squared_error(y_true, y_pred))),
        r2   = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        bias = float(np.mean(y_pred - y_true)),
    )


def growth_regime(v):
    if v < 0:    return "RECESSION"
    if v < 2.0:  return "LOW_GROWTH"
    if v < 5.0:  return "MODERATE_GROWTH"
    return "HIGH_GROWTH"


def main():
    print("=" * 70)
    print("PHASE 12 STEP 2 — Walk-Forward Confirmation")
    print("Comparing CONTROL vs BEST MOMENTUM CANDIDATE")
    print("=" * 70)

    actual_md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    assert actual_md5 == EXPECTED_MD5, f"MD5 mismatch: {actual_md5}"
    print(f"[OK] MD5 verified.")

    # Load best candidate from metadata
    with open(META_PATH) as f:
        meta = json.load(f)
    best_exp = meta.get("best_candidate_by_val_rmse", "F_FULL_MOMENTUM")
    print(f"[INFO] Best candidate from experiments: {best_exp}")

    # Determine candidate features
    exp_feature_map = {
        "A_CONTROL":       CONTROL_FEATURES,
        "C_MOMENTUM":      CONTROL_FEATURES + ["gdp_growth_accel_1y","gdp_growth_accel_2y","gdp_growth_trend_change"],
        "D_ROLLING":       CONTROL_FEATURES + ["gdp_growth_mean_2y","gdp_growth_mean_3y","gdp_growth_std_3y"],
        "E_TURNING_POINT": CONTROL_FEATURES + ["gdp_growth_decelerating","gdp_growth_accelerating","gdp_growth_negative_lag1","gdp_growth_decline_2y"],
        "F_FULL_MOMENTUM": CONTROL_FEATURES + ALL_MOMENTUM,
    }
    cand_features = exp_feature_map.get(best_exp, CONTROL_FEATURES + ALL_MOMENTUM)
    cand_features = list(dict.fromkeys(cand_features))  # deduplicate

    # Load data
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()

    ctrl_metrics_rows  = []
    cand_metrics_rows  = []
    all_preds          = []

    print(f"\n[INFO] Running {len(WINDOWS)} walk-forward windows...\n")
    print(f"{'Win':>3} | {'TargYr':>6} | {'N':>4} | {'CTRL RMSE':>9} | {'CAND RMSE':>9} | {'Delta':>7}")
    print("-" * 55)

    for w in WINDOWS:
        wid          = w["window_id"]
        eval_feat_yr = w["eval_feature_year"]
        eval_tgt_yr  = w["eval_target_year"]

        train_mask = (
            (df["year"] >= TRAIN_START) &
            (df["year"] <  eval_feat_yr) &
            (df[AVAIL_COL] == 1)
        )
        eval_mask = (
            (df["year"] == eval_feat_yr) &
            (df[AVAIL_COL] == 1)
        )
        train_df = df[train_mask].copy()
        eval_df  = df[eval_mask].copy()
        n_eval   = len(eval_df)

        if len(train_df) < 50 or n_eval == 0:
            print(f"{wid:>3} | {eval_tgt_yr:>6} | SKIPPED (train={len(train_df)}, eval={n_eval})")
            continue

        y_train = train_df[TARGET_COL].values
        y_eval  = eval_df[TARGET_COL].values

        # Control
        ctrl_pipe = build_pipe()
        ctrl_pipe.fit(train_df[CONTROL_FEATURES].copy(), y_train)
        ctrl_pred = ctrl_pipe.predict(eval_df[CONTROL_FEATURES].copy())
        ctrl_m    = calc_metrics(y_eval, ctrl_pred)

        # Candidate
        avail_cand = [f for f in cand_features if f in eval_df.columns]
        cand_pipe  = build_pipe()
        cand_pipe.fit(train_df[avail_cand].copy(), y_train)
        cand_pred  = cand_pipe.predict(eval_df[avail_cand].copy())
        cand_m     = calc_metrics(y_eval, cand_pred)

        delta = cand_m["rmse"] - ctrl_m["rmse"]
        print(f"{wid:>3} | {eval_tgt_yr:>6} | {n_eval:>4} | "
              f"{ctrl_m['rmse']:>9.4f} | {cand_m['rmse']:>9.4f} | {delta:>+7.4f}")

        ctrl_metrics_rows.append({
            "model":"CONTROL","window_id":wid,"eval_feature_year":eval_feat_yr,
            "eval_target_year":eval_tgt_yr, **ctrl_m
        })
        cand_metrics_rows.append({
            "model":best_exp,"window_id":wid,"eval_feature_year":eval_feat_yr,
            "eval_target_year":eval_tgt_yr, **cand_m
        })

        # Store predictions for regime analysis
        for i in range(n_eval):
            row = eval_df.iloc[i]
            for model_name, pred_arr in [("CONTROL",ctrl_pred),(best_exp,cand_pred)]:
                actual = float(y_eval[i])
                pred   = float(pred_arr[i])
                all_preds.append({
                    "model":model_name,"window_id":wid,
                    "eval_target_year":eval_tgt_yr,
                    "country_code":row["country_code"],
                    "country_name":row.get("country_name",""),
                    "actual_gdp_growth":actual,
                    "predicted_gdp_growth":pred,
                    "residual":actual-pred,
                    "absolute_error":abs(actual-pred),
                    "squared_error":(actual-pred)**2,
                    "growth_regime":growth_regime(actual),
                })

    # Aggregate metrics
    metrics_df = pd.DataFrame(ctrl_metrics_rows + cand_metrics_rows)
    metrics_path = OUT_DIR / "phase12_step2_walk_forward_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[SAVED] {metrics_path}")

    preds_df  = pd.DataFrame(all_preds)
    preds_path = OUT_DIR / "phase12_step2_walk_forward_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"[SAVED] {preds_path}")

    # Overall comparison
    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY: CONTROL vs CANDIDATE")
    print("=" * 70)
    for model_name in ["CONTROL", best_exp]:
        model_preds = preds_df[preds_df["model"] == model_name]
        overall_rmse = math.sqrt(model_preds["squared_error"].mean())
        overall_mae  = model_preds["absolute_error"].mean()
        print(f"\n  {model_name}:")
        print(f"    Overall RMSE: {overall_rmse:.4f} | MAE: {overall_mae:.4f}")

        for period, (lo, hi) in [("PRE_SHOCK",(2013,2019)),("SHOCK",(2020,2021)),("POST_SHOCK",(2022,2024))]:
            sub = model_preds[model_preds["eval_target_year"].between(lo, hi)]
            if len(sub) > 0:
                rmse = math.sqrt(sub["squared_error"].mean())
                print(f"    {period}: RMSE={rmse:.4f} (N={len(sub)})")

        for regime in ["RECESSION","LOW_GROWTH","MODERATE_GROWTH","HIGH_GROWTH"]:
            sub = model_preds[model_preds["growth_regime"] == regime]
            if len(sub) > 0:
                rmse = math.sqrt(sub["squared_error"].mean())
                print(f"    {regime}: RMSE={rmse:.4f} (N={len(sub)})")

    print("\n[COMPLETE] Walk-forward confirmation finished. Production model UNCHANGED.")


if __name__ == "__main__":
    main()
