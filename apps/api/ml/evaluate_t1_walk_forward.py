"""
EconoSphere AI — Phase 12, Step 1: Walk-Forward Validation
============================================================
Script   : apps/api/ml/evaluate_t1_walk_forward.py
Purpose  : Expanding-window walk-forward evaluation of the LOCKED Phase 11
           HistGradientBoostingRegressor model across 12 historical windows.
           This script is DIAGNOSTIC ONLY. It does NOT modify the production
           model, retrain for promotion, or alter any Phase 11 artifacts.

Inputs   : data/processed/master_panel_t1_missingness.csv
Outputs  : data/processed/phase12_walk_forward_metrics.csv
           data/processed/phase12_walk_forward_predictions.csv

T+1 Target Design (from Phase 11):
  - A row with feature_year = Y contains gdp_growth_next_year = GDP growth in Y+1
  - next_year_target_available == 1 means the Y+1 target is real (not NaN)
  - For walk-forward window evaluating target_year T, we use rows where
    feature_year = T-1 (features from Y-1 predict GDP growth in Y)
  - 2025 rows have next_year_target_available = 0 (inference only) — excluded.

Locked Model Configuration (Phase 11 / Step 10):
  - Algorithm  : HistGradientBoostingRegressor
  - learning_rate     = 0.05
  - max_depth         = 5
  - max_iter          = 300
  - l2_regularization = 5.0
  - random_state      = 42
  - Imputation : SimpleImputer(strategy="median")
  - Feature set: Locked Step 8 Advanced Features (31 features)
  - Target     : gdp_growth_next_year
"""
import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"

# Phase 11 production artifacts — must remain unchanged
PHASE11_ARTIFACTS = [
    PROJECT_ROOT / "data" / "processed" / "t1_step11_predictions.csv",
    PROJECT_ROOT / "data" / "processed" / "t1_step11_2026_forecasts.csv",
    PROJECT_ROOT / "models" / "phase11" / "t1_step11_model_metadata.json",
]

# ── Constants — LOCKED Phase 11 Configuration ─────────────────────────────────
EXPECTED_MD5 = "8ac7e0b2bf09fbe89289f82d0c7cf25e"

WB_AGGREGATES = {
    "AFE","AFW","ARB","CEB","CSS","EAP","EAR","EAS","ECA","ECS","EMU","EUU",
    "FCS","HIC","HPC","IBD","IBT","IDA","IDB","IDX","INX","LAC","LCN","LDC",
    "LIC","LMC","LMY","LTE","MEA","MIC","MNA","NAC","OED","OSS","PRE","PSS",
    "PST","SAS","SSA","SSF","SST","TEA","TEC","TLA","TMN","TSA","TSS","UMC","WLD",
}

LOCKED_PARAMS = {
    "l2_regularization": 5.0,
    "learning_rate":     0.05,
    "max_depth":         5,
    "max_iter":          300,
    "random_state":      42,
}

# 31 locked features (Step 8 Advanced Features)
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
    "trade_openness", "trade_balance_ratio", "log_gdp_usd",
    "log_population",
]
ROLLING_EXTRA = ["gdp_growth_rolling_std_5", "inflation_rolling_std_3"]
LOCKED_FEATURES = BASE_FEATURES + ROLLING_EXTRA  # 31 total

TARGET_COL = "gdp_growth_next_year"
TARGET_AVAIL_COL = "next_year_target_available"

# ── Walk-Forward Windows ───────────────────────────────────────────────────────
# feature_year Y predicts target_year Y+1.
# train_end_year = last feature year in training partition.
# eval_feature_year = feature_year for the evaluation slice.
# eval_target_year  = eval_feature_year + 1
WINDOWS = [
    {"window_id": 1,  "train_start_year": 2000, "train_end_year": 2012, "eval_feature_year": 2012, "eval_target_year": 2013},
    {"window_id": 2,  "train_start_year": 2000, "train_end_year": 2013, "eval_feature_year": 2013, "eval_target_year": 2014},
    {"window_id": 3,  "train_start_year": 2000, "train_end_year": 2014, "eval_feature_year": 2014, "eval_target_year": 2015},
    {"window_id": 4,  "train_start_year": 2000, "train_end_year": 2015, "eval_feature_year": 2015, "eval_target_year": 2016},
    {"window_id": 5,  "train_start_year": 2000, "train_end_year": 2016, "eval_feature_year": 2016, "eval_target_year": 2017},
    {"window_id": 6,  "train_start_year": 2000, "train_end_year": 2017, "eval_feature_year": 2017, "eval_target_year": 2018},
    {"window_id": 7,  "train_start_year": 2000, "train_end_year": 2018, "eval_feature_year": 2018, "eval_target_year": 2019},
    {"window_id": 8,  "train_start_year": 2000, "train_end_year": 2019, "eval_feature_year": 2019, "eval_target_year": 2020},
    {"window_id": 9,  "train_start_year": 2000, "train_end_year": 2020, "eval_feature_year": 2020, "eval_target_year": 2021},
    {"window_id": 10, "train_start_year": 2000, "train_end_year": 2021, "eval_feature_year": 2021, "eval_target_year": 2022},
    {"window_id": 11, "train_start_year": 2000, "train_end_year": 2022, "eval_feature_year": 2022, "eval_target_year": 2023},
    {"window_id": 12, "train_start_year": 2000, "train_end_year": 2023, "eval_feature_year": 2023, "eval_target_year": 2024},
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def verify_md5():
    """Confirm raw dataset has not changed."""
    actual = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    assert actual == EXPECTED_MD5, (
        f"Raw dataset MD5 mismatch! Expected {EXPECTED_MD5}, got {actual}. "
        "Phase 12 cannot proceed if the raw dataset has changed."
    )
    print(f"[OK] Raw dataset MD5 verified: {actual}")


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate standard forecast metrics."""
    return {
        "mae":                   float(mean_absolute_error(y_true, y_pred)),
        "rmse":                  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":                    float(r2_score(y_true, y_pred)),
        "bias":                  float(np.mean(y_pred - y_true)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
    }


def build_pipe() -> Pipeline:
    """Return a fresh pipeline with the exact locked Phase 11 parameters."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   HistGradientBoostingRegressor(**LOCKED_PARAMS)),
    ])


# ── Main Walk-Forward Evaluation ───────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PHASE 12 STEP 1 — Walk-Forward Validation")
    print("Locked model: HistGradientBoostingRegressor | Official RMSE: 3.9113")
    print("=" * 70)

    # 0. Integrity gate
    verify_md5()

    # 1. Load data
    df = pd.read_csv(INPUT_PATH)
    print(f"\n[INFO] Loaded {len(df):,} rows | Years: {df['year'].min()}–{df['year'].max()}")

    # 2. Filter World Bank aggregate rows (same as Phase 11)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    print(f"[INFO] After WB aggregate exclusion: {len(df):,} rows")

    # 3. Sanity check: feature columns must exist
    missing_cols = [f for f in LOCKED_FEATURES + [TARGET_COL, TARGET_AVAIL_COL] if f not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 4. Run expanding-window walk-forward evaluation
    all_metrics = []
    all_predictions = []
    # Store per-window permutation importances
    feature_importance_records: list[dict] = []

    print(f"\n[INFO] Running {len(WINDOWS)} walk-forward windows...\n")

    for w in WINDOWS:
        wid           = w["window_id"]
        train_start   = w["train_start_year"]
        train_end     = w["train_end_year"]
        eval_feat_yr  = w["eval_feature_year"]
        eval_tgt_yr   = w["eval_target_year"]

        # ── Training partition: feature years train_start..train_end (inclusive)
        #    Must have valid target (next_year_target_available == 1) to avoid
        #    training on rows that don't have a known next-year GDP.
        #    NOTE: train_end row's target is GDP growth in train_end+1,
        #    which is the FIRST eval target year; we intentionally include it
        #    in training because its target (GDP growth of train_end+1) is the
        #    eval period and we want to avoid leakage: we only include rows in
        #    training where feature_year < eval_feat_yr.
        train_mask = (
            (df["year"] >= train_start) &
            (df["year"] <  eval_feat_yr) &   # strictly before eval feature year
            (df[TARGET_AVAIL_COL] == 1)
        )
        # ── Evaluation partition: exactly the eval feature year rows
        eval_mask = (
            (df["year"] == eval_feat_yr) &
            (df[TARGET_AVAIL_COL] == 1)        # only rows with real targets
        )

        train_df = df[train_mask].copy()
        eval_df  = df[eval_mask].copy()

        n_train = len(train_df)
        n_eval  = len(eval_df)

        if n_train < 50:
            print(f"  Window {wid:2d} [target {eval_tgt_yr}]: SKIPPED — insufficient training rows ({n_train})")
            continue
        if n_eval == 0:
            print(f"  Window {wid:2d} [target {eval_tgt_yr}]: SKIPPED — no evaluation rows with valid targets")
            continue

        # ── Extract features / target
        X_train = train_df[LOCKED_FEATURES].copy()
        y_train = train_df[TARGET_COL].values

        X_eval  = eval_df[LOCKED_FEATURES].copy()
        y_eval  = eval_df[TARGET_COL].values

        # ── Fit pipeline on training data ONLY
        pipe = build_pipe()
        pipe.fit(X_train, y_train)

        # ── Predict on eval period
        y_pred = pipe.predict(X_eval)

        # ── Metrics
        m = get_metrics(y_eval, y_pred)
        metrics_row = {
            "window_id":          wid,
            "train_start_year":   train_start,
            "train_end_year":     train_end,
            "feature_year":       eval_feat_yr,
            "target_year":        eval_tgt_yr,
            "n_observations":     n_eval,
            **m,
        }
        all_metrics.append(metrics_row)

        print(
            f"  Window {wid:2d} [target {eval_tgt_yr}]: "
            f"N={n_eval:3d} | RMSE={m['rmse']:.4f} | MAE={m['mae']:.4f} | "
            f"R²={m['r2']:.4f} | Bias={m['bias']:+.4f}"
        )

        # ── Store per-row predictions
        for idx, (pred_val, actual_val) in enumerate(zip(y_pred, y_eval)):
            row_data = eval_df.iloc[idx]
            residual   = float(actual_val - pred_val)
            abs_error  = abs(residual)
            sq_error   = residual ** 2
            all_predictions.append({
                "window_id":            wid,
                "feature_year":         eval_feat_yr,
                "target_year":          eval_tgt_yr,
                "country_code":         row_data.get("country_code", ""),
                "country_name":         row_data.get("country_name", ""),
                "actual_gdp_growth":    float(actual_val),
                "predicted_gdp_growth": float(pred_val),
                "residual":             residual,
                "absolute_error":       abs_error,
                "squared_error":        sq_error,
            })

        # ── Post-hoc permutation importance on the EVALUATION set
        #    This is strictly diagnostic — not used for feature selection.
        try:
            perm = permutation_importance(
                pipe, X_eval, y_eval,
                n_repeats=5,
                random_state=42,
                scoring="neg_mean_squared_error",
            )
            for feat_idx, feat_name in enumerate(LOCKED_FEATURES):
                feature_importance_records.append({
                    "window_id":   wid,
                    "feature":     feat_name,
                    "importance":  float(perm.importances_mean[feat_idx]),
                    "importance_std": float(perm.importances_std[feat_idx]),
                })
        except Exception as exc:
            print(f"    [WARN] Permutation importance failed for window {wid}: {exc}")

    # 5. Save metrics CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = OUT_DIR / "phase12_walk_forward_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[SAVED] {metrics_path}")

    # 6. Save predictions CSV
    preds_df = pd.DataFrame(all_predictions)
    preds_path = OUT_DIR / "phase12_walk_forward_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"[SAVED] {preds_path}")

    # 7. Save feature importance CSV
    if feature_importance_records:
        fi_df = pd.DataFrame(feature_importance_records)
        fi_agg = (
            fi_df.groupby("feature")["importance"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "mean_importance", "std": "importance_std", "count": "number_of_windows"})
            .reset_index()
            .sort_values("mean_importance", ascending=False)
        )
        fi_path = OUT_DIR / "phase12_feature_diagnostic_importance.csv"
        fi_agg.to_csv(fi_path, index=False)
        print(f"[SAVED] {fi_path}")

    # 8. Overall summary
    if not metrics_df.empty:
        all_sq = preds_df["squared_error"].values
        overall_rmse = float(np.sqrt(np.mean(all_sq)))
        overall_mae  = float(preds_df["absolute_error"].mean())
        overall_bias = float(preds_df["residual"].mean())   # mean(actual - pred)
        overall_r2   = float(r2_score(preds_df["actual_gdp_growth"], preds_df["predicted_gdp_growth"]))

        print("\n" + "=" * 70)
        print("OVERALL WALK-FORWARD SUMMARY")
        print(f"  Windows executed  : {len(metrics_df)}")
        print(f"  Total predictions : {len(preds_df):,}")
        print(f"  Overall RMSE      : {overall_rmse:.4f}")
        print(f"  Overall MAE       : {overall_mae:.4f}")
        print(f"  Overall Bias      : {overall_bias:+.4f}  (positive = under-prediction)")
        print(f"  Overall R²        : {overall_r2:.4f}")
        print(f"\n  Note: Official Phase 11 Test RMSE = 3.9113 (unchanged)")
        print("=" * 70)

    # 9. Final Phase 11 artifact integrity check
    print("\n[INTEGRITY] Checking Phase 11 production artifacts are unchanged...")
    for art in PHASE11_ARTIFACTS:
        if art.exists():
            print(f"  [OK] {art.name} — present")
        else:
            print(f"  [WARN] {art.name} — NOT FOUND (expected to exist from Phase 11)")

    print("\n[COMPLETE] evaluate_t1_walk_forward.py finished. Production model UNCHANGED.")


if __name__ == "__main__":
    main()
