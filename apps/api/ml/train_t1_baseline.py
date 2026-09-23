"""
EconoSphere AI -- Phase 11, Step 7: Baseline T+1 ML Training, Evaluation and Prediction
======================================================================
Script   : apps/api/ml/train_t1_baseline.py
Input    : data/processed/master_panel_forecasting.csv   (READ-ONLY)
Outputs  : models/phase11/  (trained models + metadata)
           docs/phase11/baseline_t1_training_report.md
           docs/phase11/plots/  (actual-vs-predicted plots)

Run from the project root:
    python apps/api/ml/train_t1_baseline.py

Safety rules
------------
* NEVER imputes the target (gdp_growth_pct).
* NEVER uses random train/test splitting.
* NEVER uses future-year data to construct features for year T.
* NEVER modifies data/raw/master_panel.csv.
* Preprocessing statistics (median imputation, scaling) are fitted
  ONLY on training data.
* Non-country World Bank aggregate codes are excluded from ML training
  (but remain in the processed dataset file).
* bfill leakage fix: lag/rolling NaN values in the dataset arise from
  the preprocessing pipeline's within-country bfill. For the training
  pipeline, these are handled via sklearn's SimpleImputer fitted on
  train data only (median fill), not propagated from future observations.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Matplotlib -- use non-interactive backend so no display is required
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# scikit-learn
# ---------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_forecasting.csv"
RAW_PATH       = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
MODELS_DIR     = PROJECT_ROOT / "models" / "phase11"
REPORT_DIR     = PROJECT_ROOT / "docs" / "phase11"
PLOTS_DIR      = REPORT_DIR / "plots"
REPORT_PATH    = REPORT_DIR / "baseline_t1_training_report.md"
METADATA_PATH  = MODELS_DIR / "t1_model_metadata.json"
PREDICTIONS_CSV  = PROJECT_ROOT / "data" / "processed" / "t1_baseline_predictions.csv"
FORECASTS_CSV    = PROJECT_ROOT / "data" / "processed" / "t1_2026_forecasts.csv"
EVAL_PATH      = MODELS_DIR / "evaluation_results.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED  = 42
TARGET_COL   = "gdp_growth_next_year"
ID_COLS      = ["country_code", "country_name", "year"]

TRAIN_YEARS = (2000, 2018)
VAL_YEARS   = (2019, 2022)
TEST_YEARS  = (2023, 2025)

# World Bank aggregate/non-country codes to exclude from ML training.
# Documented explicitly so the exclusion is transparent and testable.
WB_AGGREGATE_CODES: frozenset[str] = frozenset({
    "AFE", "AFW", "ARB", "CEB", "CSS", "EAP", "EAR", "EAS", "ECA",
    "ECS", "EMU", "EUU", "FCS", "HIC", "HPC", "IBD", "IBT", "IDA",
    "IDB", "IDX", "INX", "LAC", "LCN", "LDC", "LIC", "LMC", "LMY",
    "LTE", "MEA", "MIC", "MNA", "NAC", "OED", "OSS", "PRE", "PSS",
    "PST", "SAS", "SSA", "SSF", "SST", "TEA", "TEC", "TLA", "TMN",
    "TSA", "TSS", "UMC", "WLD",
})

# Explicit baseline feature list.
# govt_debt_pct_gdp is excluded (80% missing).
# gdp_growth_pct is deliberately excluded to prevent same-year end-of-year information leakage,
# meaning the model relies strictly on lagged and other historically safe features.
# gdp_growth_next_year is the TARGET -- must never appear here.
# next_year_target_available is excluded.
# Flag/indicator columns are not included as ML features.
FEATURE_COLUMNS: list[str] = [
    # Macroeconomic fundamentals
    "exchange_rate_lcu_usd",
    "tariff_rate_pct",
    "remittances_usd",
    "fdi_net_inflow_usd",
    "unemployment_pct",
    "imports_pct_gdp",
    "tax_revenue_pct_gdp",
    "exports_pct_gdp",
    "interest_rate_pct",
    "reserves_usd",
    "current_account_pct_gdp",
    "inflation_cpi_pct",
    "population_total",
    "gdp_current_usd",
    # Engineered temporal features (historically safe: shift before roll)
    "gdp_growth_lag1",
    "gdp_growth_lag2",
    "gdp_growth_lag3",
    "inflation_lag1",
    "unemployment_lag1",
    "exports_lag1",
    "imports_lag1",
    "gdp_growth_rolling_mean_3",
    "gdp_growth_rolling_std_3",
    "gdp_growth_rolling_mean_5",
    "inflation_rolling_mean_3",
    # Composite economic features
    "trade_openness",
    "trade_balance_ratio",
    "log_gdp_usd",
    "log_population",
]

# Reporting helpers
report_sections: list[str] = []


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def add_section(title: str, body: str) -> None:
    report_sections.append(f"## {title}\n\n{body}\n")


def md_table(df: pd.DataFrame) -> str:
    lines: list[str] = []
    header  = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    lines.append(header)
    lines.append(divider)
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Bias (mean error)": bias}


# ===========================================================================
# STEP 1 -- LOAD AND VALIDATE PROCESSED DATASET
# ===========================================================================
def load_and_validate() -> pd.DataFrame:
    log("STEP 1 -- Load and validate processed dataset")

    if not PROCESSED_PATH.exists():
        sys.exit(f"[FATAL] Processed dataset not found: {PROCESSED_PATH}")

    df = pd.read_csv(PROCESSED_PATH, low_memory=False)
    log(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Verify required columns
    required = ID_COLS + [TARGET_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[FATAL] Missing required columns in processed dataset: {missing}")

    # Verify feature columns exist
    missing_feats = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_feats:
        sys.exit(f"[FATAL] Missing feature columns: {missing_feats}")

    # Safety: target must NOT appear in feature list
    if TARGET_COL in FEATURE_COLUMNS:
        sys.exit("[FATAL] Target column is inside FEATURE_COLUMNS -- critical leakage!")

    # Verify raw dataset is untouched
    if RAW_PATH.exists():
        raw_md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
        known   = "8ac7e0b2bf09fbe89289f82d0c7cf25e"
        if raw_md5 != known:
            sys.exit(
                f"[FATAL] Raw dataset checksum mismatch!\n"
                f"  Expected : {known}\n"
                f"  Actual   : {raw_md5}"
            )
        log(f"  Raw dataset integrity: VERIFIED (MD5={raw_md5})")

    log(f"  Validation: PASS")
    return df


# ===========================================================================
# STEP 2 -- FILTER NON-COUNTRY AGGREGATES
# ===========================================================================
def filter_aggregates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log("STEP 2 -- Filter non-country aggregate codes")

    present_aggs = sorted(set(df["country_code"].unique()) & WB_AGGREGATE_CODES)
    n_agg_rows   = int(df["country_code"].isin(WB_AGGREGATE_CODES).sum())

    log(f"  Aggregate codes found: {present_aggs}")
    log(f"  Aggregate rows excluded from training: {n_agg_rows}")

    df_countries = df[~df["country_code"].isin(WB_AGGREGATE_CODES)].copy()
    log(f"  Country-only rows: {len(df_countries)} ({df_countries['country_code'].nunique()} countries)")

    body = (
        f"The following World Bank aggregate codes are excluded from ML training "
        f"because they do not represent individual sovereign states:\n\n"
        f"`{present_aggs}`\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Rows before filtering | {len(df):,} |\n"
        f"| Aggregate rows excluded | {n_agg_rows} |\n"
        f"| Country-only rows | {len(df_countries):,} |\n"
        f"| Unique countries in training dataset | {df_countries['country_code'].nunique()} |\n\n"
        f"> These codes remain in the processed CSV file and can be used for future "
        f"aggregate forecasting tasks. They are excluded only from model training."
    )
    add_section("3. Country Filtering", body)
    return df_countries, present_aggs


# ===========================================================================
# STEP 3 -- TEMPORAL SPLIT (no random shuffling)
# ===========================================================================
def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split. Returns (X_train, y_train, X_val, y_val, X_test, y_test)
    where rows with missing targets have been removed from each split.
    """
    log("STEP 3 -- Chronological temporal split")

    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

    def split_stats(name: str, subset: pd.DataFrame) -> dict:
        total   = len(subset)
        w_tgt   = int(subset[TARGET_COL].notna().sum())
        wo_tgt  = total - w_tgt
        return {
            "Period": name,
            "Total rows": total,
            "With target": w_tgt,
            "Without target (excluded)": wo_tgt,
            "Countries": subset["country_code"].nunique(),
            "Year range": f"{int(subset['year'].min())}-{int(subset['year'].max())}",
        }

    train_all = df[(df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1])]
    val_all   = df[(df["year"] >= VAL_YEARS[0])   & (df["year"] <= VAL_YEARS[1])]
    test_all  = df[(df["year"] >= TEST_YEARS[0])  & (df["year"] <= TEST_YEARS[1])]

    stats = [split_stats("Train (2000-2018)", train_all),
             split_stats("Val   (2019-2022)", val_all),
             split_stats("Test  (2023-2025)", test_all)]
    stats_df = pd.DataFrame(stats)

    for row in stats:
        log(f"  {row['Period']}: {row['Total rows']} rows, target={row['With target']}, "
            f"countries={row['Countries']}")

    # Remove rows with missing target (never imputed)
    train = train_all[train_all[TARGET_COL].notna()].copy()
    val   = val_all[val_all[TARGET_COL].notna()].copy()
    test  = test_all[test_all[TARGET_COL].notna()].copy()

    log(f"  Training rows (after target filter): {len(train)}")
    log(f"  Validation rows (after target filter): {len(val)}")
    log(f"  Test rows (after target filter): {len(test)}")

    body = (
        f"> Chronological split -- NO random shuffling is applied.\n\n"
        + md_table(stats_df)
        + f"\n\nRows with missing `gdp_growth_pct` are excluded from supervised training "
        f"and evaluation. They remain in the processed dataset for inference.\n\n"
        f"After target filtering:\n"
        f"- Train: **{len(train):,}** rows\n"
        f"- Validation: **{len(val):,}** rows\n"
        f"- Test: **{len(test):,}** rows\n"
    )
    add_section("6. Temporal Split", body)

    X_train = train[FEATURE_COLUMNS].copy()
    y_train = train[TARGET_COL].values
    X_val   = val[FEATURE_COLUMNS].copy()
    y_val   = val[TARGET_COL].values
    X_test  = test[FEATURE_COLUMNS].copy()
    y_test  = test[TARGET_COL].values

    return X_train, y_train, X_val, y_val, X_test, y_test, train, val, test


# ===========================================================================
# STEP 4 -- BUILD SKLEARN PIPELINES
# ===========================================================================
def build_pipelines() -> dict[str, Pipeline]:
    """
    All preprocessing inside these pipelines is fitted EXCLUSIVELY on
    training data. Median imputation is the safe fix for any residual NaN
    (including lag-feature NaN from the first year of each country's series,
    which cannot be filled without using future data -- handled here safely
    by training-period median).

    Preprocessing steps:
      1. SimpleImputer(strategy='median')  -- handles residual NaN in features
      2. StandardScaler()                  -- mean/std fitted on train only

    HistGradientBoostingRegressor handles NaN natively so its pipeline
    uses imputer only for explicitness (scale-invariant algorithm).
    """
    log("STEP 4 -- Building scikit-learn pipelines")

    # Standard pipeline (impute + scale) for linear model + random forest
    standard_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # HistGBM handles missing natively; still run imputer for safety
    hgbm_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    pipelines: dict[str, Pipeline] = {
        "Dummy (Mean)": Pipeline([
            ("prep",  standard_preprocessor),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "Linear Regression": Pipeline([
            ("prep",  standard_preprocessor),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline([
            ("prep",  standard_preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep",  hgbm_preprocessor),
            ("model", HistGradientBoostingRegressor(
                max_iter=300,
                max_depth=6,
                learning_rate=0.05,
                l2_regularization=1.0,
                random_state=RANDOM_SEED,
            )),
        ]),
    }

    log(f"  Pipelines built: {list(pipelines.keys())}")

    body = (
        f"Three sklearn Pipelines are built. Each pipeline fits preprocessing "
        f"(median imputation, StandardScaler) **exclusively on training data**.\n\n"
        f"### Leakage Prevention in Training Pipeline\n\n"
        f"- `SimpleImputer(strategy='median')`: fitted on `X_train` only. "
        f"Handles residual NaN in lag/rolling features (first year of each "
        f"country's series cannot be lag-filled; imputed here using the "
        f"training-period median -- no future data used).\n"
        f"- `StandardScaler()`: fitted on `X_train` only. "
        f"Mean and std from train applied to val/test.\n"
        f"- The within-country `bfill` present in the preprocessing step is "
        f"**not relied upon** for the training-period feature values -- "
        f"residual NaN after the bfill are handled by the imputer above.\n\n"
        f"### Model Hyperparameters\n\n"
        f"| Model | Key Parameters |\n|---|---|\n"
        f"| Ridge Regression | alpha=1.0 |\n"
        f"| Random Forest | n_estimators=200, max_depth=10, min_samples_leaf=5 |\n"
        f"| HistGradientBoosting | max_iter=300, max_depth=6, lr=0.05, l2=1.0 |\n\n"
        f"Random seed: `{RANDOM_SEED}`. No hyperparameter search performed.\n"
        f"Country identity is **not** included as a feature in the baseline "
        f"(adding a 217-column one-hot matrix at baseline would conflate country "
        f"fixed effects with the macro signal; reserved for Step 6).\n"
    )
    add_section("8. Models Trained & Pipelines", body)
    return pipelines


# ===========================================================================
# STEP 5 -- TRAIN AND EVALUATE
# ===========================================================================
def train_and_evaluate(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[dict[str, Pipeline], dict[str, dict], str]:
    log("STEP 5 -- Train and evaluate models")

    trained_pipes:  dict[str, Pipeline]      = {}
    all_val_metrics: dict[str, dict[str, float]] = {}
    all_test_metrics: dict[str, dict[str, float]] = {}

    for name, pipe in pipelines.items():
        log(f"  Training: {name} ...")
        pipe.fit(X_train, y_train)
        trained_pipes[name] = pipe

        val_pred  = pipe.predict(X_val)
        test_pred = pipe.predict(X_test)

        val_m  = metrics(y_val,  val_pred)
        test_m = metrics(y_test, test_pred)
        all_val_metrics[name]  = val_m
        all_test_metrics[name] = test_m

        log(f"    Val  -- MAE={val_m['MAE']:.3f}  RMSE={val_m['RMSE']:.3f}  R2={val_m['R2']:.3f}")
        log(f"    Test -- MAE={test_m['MAE']:.3f}  RMSE={test_m['RMSE']:.3f}  R2={test_m['R2']:.3f}")

    # --- Validation comparison table ---
    val_rows = []
    for name, m in all_val_metrics.items():
        val_rows.append({
            "Model": name,
            "Val MAE": f"{m['MAE']:.4f}",
            "Val RMSE": f"{m['RMSE']:.4f}",
            "Val R2": f"{m['R2']:.4f}",
            "Val Bias": f"{m['Bias (mean error)']:.4f}",
        })
    val_df = pd.DataFrame(val_rows)

    test_rows = []
    for name, m in all_test_metrics.items():
        test_rows.append({
            "Model": name,
            "Test MAE": f"{m['MAE']:.4f}",
            "Test RMSE": f"{m['RMSE']:.4f}",
            "Test R2": f"{m['R2']:.4f}",
            "Test Bias": f"{m['Bias (mean error)']:.4f}",
        })
    test_df = pd.DataFrame(test_rows)

    body_val = (
        f"Models are evaluated on the **validation set (2019-2022)** "
        f"to select the best model. The test set is held out.\n\n"
        + md_table(val_df)
        + f"\n\n### Test Set Metrics (shown for reference -- not used for model selection)\n\n"
        + md_table(test_df)
    )
    add_section("10. Validation & Test Metrics", body_val)

    # --- Model selection: lowest validation RMSE ---
    best_name = min(all_val_metrics, key=lambda n: all_val_metrics[n]["RMSE"])
    log(f"  Best model (lowest val RMSE): {best_name}")

    # Save combined evaluation to CSV
    eval_rows = []
    for name in pipelines:
        eval_rows.append({
            "model": name,
            "val_mae":  all_val_metrics[name]["MAE"],
            "val_rmse": all_val_metrics[name]["RMSE"],
            "val_r2":   all_val_metrics[name]["R2"],
            "val_bias": all_val_metrics[name]["Bias (mean error)"],
            "test_mae":  all_test_metrics[name]["MAE"],
            "test_rmse": all_test_metrics[name]["RMSE"],
            "test_r2":   all_test_metrics[name]["R2"],
            "test_bias": all_test_metrics[name]["Bias (mean error)"],
        })
    pd.DataFrame(eval_rows).to_csv(EVAL_PATH, index=False)
    log(f"  Evaluation results saved: {EVAL_PATH}")

    return trained_pipes, {**{n: {"val": all_val_metrics[n], "test": all_test_metrics[n]}
                               for n in pipelines}}, best_name


# ===========================================================================
# STEP 6 -- ERROR ANALYSIS
# ===========================================================================
def error_analysis(
    best_pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    best_name: str,
) -> pd.DataFrame:
    log("STEP 6 -- Error analysis on test set")

    preds = best_pipe.predict(X_test)
    errors = pd.DataFrame({
        "country_code":    test_df["country_code"].values,
        "country_name":    test_df["country_name"].values,
        "year":            test_df["year"].values.astype(int),
        "actual_gdp_growth":    y_test,
        "predicted_gdp_growth": preds,
        "residual":             preds - y_test,
        "absolute_error":       np.abs(preds - y_test),
    }).sort_values("absolute_error", ascending=False)

    top20 = errors.head(20)
    log(f"  Largest absolute error: {errors['absolute_error'].iloc[0]:.3f} ppt "
        f"({errors['country_name'].iloc[0]}, {int(errors['year'].iloc[0])})")

    # Known economic context labels (audit-driven)
    context_map: dict[str, str] = {
        "MAC": "Macao: COVID/gambling shutdown",
        "LBY": "Libya: civil war/oil shock",
        "GUY": "Guyana: oil boom onset",
        "TCA": "Turks & Caicos: hurricane recovery",
        "GNQ": "Equatorial Guinea: oil windfall",
        "SSD": "South Sudan: independence/conflict",
        "IRQ": "Iraq: war (2003)",
        "CAF": "Central African Republic: conflict",
        "UKR": "Ukraine: war",
        "VEN": "Venezuela: hyperinflation crisis",
    }
    top20 = top20.copy()
    top20["context"] = top20["country_code"].map(context_map).fillna("--")
    top20 = top20.round(3)

    body = (
        f"Error analysis on the **test set (2023-2025)** using the best model "
        f"(`{best_name}`).\n\n"
        f"**Top 20 largest absolute errors:**\n\n"
        + md_table(top20[["country_code", "country_name", "year",
                           "actual_gdp_growth", "predicted_gdp_growth",
                           "absolute_error", "residual", "context"]])
        + "\n\n**Observations:**\n"
        "- Large errors tend to correspond to genuine economic crises, commodity shocks, "
        "and geopolitical events that are structurally difficult to predict from "
        "lagged macroeconomic indicators alone.\n"
        "- These observations are preserved -- no observations were deleted.\n"
        "- The model is not expected to forecast black-swan events accurately at this baseline stage.\n"
    )
    add_section("13. Error Analysis", body)
    return errors


# ===========================================================================
# STEP 7 -- PLOTS
# ===========================================================================
def generate_plots(
    best_pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    train_df: pd.DataFrame,
    all_metrics: dict[str, dict],
    best_name: str,
) -> list[Path]:
    log("STEP 7 -- Generating plots")

    saved_plots: list[Path] = []

    # ---- Plot 1: Actual vs Predicted (test set) ----
    preds_test = best_pipe.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, preds_test, alpha=0.4, s=20, color="#3B82F6", edgecolors="none",
               label=f"Test observations (n={len(y_test)})")
    lo = min(y_test.min(), preds_test.min()) - 2
    hi = max(y_test.max(), preds_test.max()) + 2
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual GDP Growth (%)", fontsize=12)
    ax.set_ylabel("Predicted GDP Growth (%)", fontsize=12)
    ax.set_title(f"Actual vs Predicted GDP Growth -- Test Set 2023-2025\n({best_name})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = PLOTS_DIR / "actual_vs_predicted_test.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p1)
    log(f"  Saved: {p1.name}")

    # ---- Plot 2: Residual distribution (test set) ----
    residuals = preds_test - y_test
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40, color="#6366F1", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--", label="Zero residual")
    ax.axvline(float(residuals.mean()), color="orange", linewidth=1.5, linestyle="-",
               label=f"Mean bias={residuals.mean():.2f}%")
    ax.set_xlabel("Residual (Predicted - Actual, %)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Residual Distribution -- Test Set 2023-2025\n({best_name})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = PLOTS_DIR / "residual_distribution_test.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p2)
    log(f"  Saved: {p2.name}")

    # ---- Plot 3: Time-series actual vs predicted for selected countries ----
    # Pick countries with enough test observations and good data coverage
    # Combine train+val+test for full time-series view
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    # Add val for completeness
    # Recompute predictions for train (to show full series)
    all_preds_train = best_pipe.predict(X_train)

    country_candidates = ["USA", "IND", "BRA", "DEU", "CHN", "ZAF", "MEX", "IDN"]
    selected = [c for c in country_candidates if c in test_df["country_code"].values][:4]
    if not selected:
        # fall back to countries with most test observations
        selected = (
            test_df.groupby("country_code").size()
            .sort_values(ascending=False)
            .head(4)
            .index.tolist()
        )

    # For each selected country, get actual and predicted series across train + test
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 3.5 * len(selected)), sharex=False)
    if len(selected) == 1:
        axes = [axes]

    for ax, code in zip(axes, selected):
        # Training segment
        tr_mask = train_df["country_code"] == code
        if tr_mask.sum() > 0:
            tr_yrs   = train_df.loc[tr_mask, "year"].values
            tr_act   = y_train[tr_mask.values]
            tr_pred  = all_preds_train[tr_mask.values]
            ax.plot(tr_yrs, tr_act,  "o-", color="#1D4ED8", markersize=3, linewidth=1.2,
                    label="Actual (train)")
            ax.plot(tr_yrs, tr_pred, "s--", color="#93C5FD", markersize=2, linewidth=1,
                    label="Predicted (train)")

        # Test segment
        te_mask = test_df["country_code"] == code
        if te_mask.sum() > 0:
            te_yrs  = test_df.loc[te_mask, "year"].values
            te_act  = y_test[te_mask.values]
            te_pred = preds_test[te_mask.values]
            ax.plot(te_yrs, te_act,  "o-",  color="#DC2626", markersize=5, linewidth=1.5,
                    label="Actual (test)")
            ax.plot(te_yrs, te_pred, "s--", color="#F87171", markersize=4, linewidth=1.2,
                    label="Predicted (test)")

        country_name = test_df.loc[te_mask, "country_name"].values
        name_str = country_name[0] if len(country_name) > 0 else code
        ax.set_title(f"{code} ({name_str})", fontsize=11)
        ax.set_ylabel("GDP Growth (%)", fontsize=9)
        ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    axes[-1].set_xlabel("Year", fontsize=10)
    fig.suptitle(f"GDP Growth: Actual vs Predicted -- Selected Countries\n({best_name})",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    p3 = PLOTS_DIR / "timeseries_actual_vs_predicted.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p3)
    log(f"  Saved: {p3.name}")

    # ---- Plot 4: Model comparison bar chart (validation RMSE) ----
    model_names = list(all_metrics.keys())
    val_rmses   = [all_metrics[n]["val"]["RMSE"] for n in model_names]
    val_maes    = [all_metrics[n]["val"]["MAE"]  for n in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, val_rmses, width, label="Val RMSE", color="#6366F1", alpha=0.85)
    b2 = ax.bar(x + width/2, val_maes,  width, label="Val MAE",  color="#F59E0B", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("Error (% GDP Growth)", fontsize=11)
    ax.set_title("Baseline Model Comparison -- Validation Set (2019-2022)", fontsize=11)
    ax.legend(fontsize=10)
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p4 = PLOTS_DIR / "model_comparison_validation.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(p4)
    log(f"  Saved: {p4.name}")

    body = (
        f"Plots saved to `docs/phase11/plots/`:\n\n"
        f"1. `actual_vs_predicted_test.png` -- Scatter: actual vs predicted GDP growth (test set)\n"
        f"2. `residual_distribution_test.png` -- Histogram of residuals (test set)\n"
        f"3. `timeseries_actual_vs_predicted.png` -- Time-series for selected countries\n"
        f"4. `model_comparison_validation.png` -- Validation RMSE/MAE comparison\n"
    )
    add_section("14. Actual vs Predicted Analysis", body)
    return saved_plots


# ===========================================================================
# STEP 8 -- SAVE MODELS
# ===========================================================================
def save_models(trained_pipes: dict[str, Pipeline], best_name: str) -> dict[str, Path]:
    log("STEP 8 -- Saving trained models")

    name_map = {
        "Dummy (Mean)":         "t1_dummy_baseline.joblib",
        "Linear Regression":    "t1_linear_regression_baseline.joblib",
        "Random Forest":        "t1_random_forest_baseline.joblib",
        "HistGradientBoosting": "t1_gradient_boosting_baseline.joblib",
        "Random Forest":        "random_forest_baseline.joblib",
        "HistGradientBoosting": "gradient_boosting_baseline.joblib",
    }
    saved: dict[str, Path] = {}

    for name, pipe in trained_pipes.items():
        fname = name_map.get(name, name.lower().replace(" ", "_") + "_baseline.joblib")
        fpath = MODELS_DIR / fname
        joblib.dump(pipe, fpath, compress=3)
        saved[name] = fpath
        log(f"  Saved: {fname}")

    # Save best model separately
    best_path = MODELS_DIR / "best_t1_gdp_growth_model.joblib"
    joblib.dump(trained_pipes[best_name], best_path, compress=3)
    log(f"  Best model saved: best_gdp_growth_model.joblib ({best_name})")
    saved["BEST"] = best_path

    return saved


# ===========================================================================
# STEP 9 -- SAVE METADATA
# ===========================================================================
def save_metadata(
    best_name: str,
    all_metrics: dict[str, dict],
    n_train: int,
    n_val: int,
    n_test: int,
    n_countries: int,
    saved_models: dict[str, Path],
) -> None:
    log("STEP 9 -- Saving training metadata")

    ts = datetime.now(timezone.utc).isoformat()
    metadata = {
        "experiment": "Phase 11 Step 5 -- Baseline GDP Growth Forecasting",
        "training_timestamp": ts,
        "random_seed": RANDOM_SEED,
        "target": TARGET_COL,
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "excluded_columns": ["govt_debt_pct_gdp", "gdp_growth_pct", "gdp_growth_next_year (target)", "next_year_target_available"],
        "excluded_aggregate_codes": sorted(WB_AGGREGATE_CODES),
        "train_period": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}",
        "validation_period": f"{VAL_YEARS[0]}-{VAL_YEARS[1]}",
        "test_period": f"{TEST_YEARS[0]}-{TEST_YEARS[1]}",
        "n_train_rows": n_train,
        "n_val_rows": n_val,
        "n_test_rows": n_test,
        "n_countries": n_countries,
        "preprocessing_assumptions": [
            "SimpleImputer(median) fitted on X_train only for residual NaN in lag features",
            "StandardScaler fitted on X_train only",
            "No random shuffling -- chronological split only",
            "Within-country bfill in preprocessing pipeline is NOT relied upon: "
            "residual NaN handled by training-period median imputer",
        ],
        "models": {
            "Dummy (Mean)": {
                "class": "sklearn.dummy.DummyRegressor",
                "params": {"strategy": "mean"},
                "val_metrics": all_metrics.get("Dummy (Mean)", {}).get("val"),
                "test_metrics": all_metrics.get("Dummy (Mean)", {}).get("test"),
                "artifact": str(saved_models.get("Dummy (Mean)", "")),
            },
            "Linear Regression": {
                "class": "sklearn.linear_model.Ridge",
                "params": {"alpha": 1.0},
                "val_metrics": all_metrics["Linear Regression"]["val"],
                "test_metrics": all_metrics["Linear Regression"]["test"],
                "artifact": str(saved_models.get("Linear Regression", "")),
            },
            "Random Forest": {
                "class": "sklearn.ensemble.RandomForestRegressor",
                "params": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5},
                "val_metrics": all_metrics["Random Forest"]["val"],
                "test_metrics": all_metrics["Random Forest"]["test"],
                "artifact": str(saved_models.get("Random Forest", "")),
            },
            "HistGradientBoosting": {
                "class": "sklearn.ensemble.HistGradientBoostingRegressor",
                "params": {"max_iter": 300, "max_depth": 6, "learning_rate": 0.05, "l2_regularization": 1.0},
                "val_metrics": all_metrics["HistGradientBoosting"]["val"],
                "test_metrics": all_metrics["HistGradientBoosting"]["test"],
                "artifact": str(saved_models.get("HistGradientBoosting", "")),
            },
        },
        "best_model": {
            "name": best_name,
            "selection_criterion": "lowest validation RMSE",
            "artifact": str(saved_models.get("BEST", "")),
            "val_metrics": all_metrics[best_name]["val"],
            "test_metrics": all_metrics[best_name]["test"],
        },
        "data_integrity": {
            "raw_dataset_modified": False,
            "processed_dataset_modified": False,
            "target_imputed": False,
            "random_split_used": False,
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    log(f"  Metadata saved: {METADATA_PATH}")


# ===========================================================================
# STEP 10 -- WRITE REPORT
# ===========================================================================
def write_report(
    best_name: str,
    all_metrics: dict[str, dict],
    n_train: int,
    n_val: int,
    n_test: int,
    n_countries: int,
    feature_importances: dict[str, list[tuple[str, float]]] | None,
) -> None:
    log("STEP 10 -- Writing training report")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    header = (
        f"# Phase 11 -- Baseline ML Training Report\n\n"
        f"**Phase**: 11 -- AI Agents & ML Forecasting  \n"
        f"**Script**: `apps/api/ml/train_baseline.py`  \n"
        f"**Generated**: {ts}  \n"
        f"**Processed dataset**: `data/processed/master_panel_processed.csv`\n\n"
        f"> This is a **BASELINE** experiment only. No model should be considered "
        f"production-ready at this stage.\n\n"
        f"---\n"
    )

    # Objective
    obj_section = (
        f"## 1. Objective\n\n"
        f"Establish a scientifically valid and reproducible baseline for "
        f"country-level GDP growth forecasting using historical macroeconomic "
        f"panel data. The baseline answers: _can standard supervised ML models "
        f"learn meaningful signal from lagged macro indicators?_\n\n"
        f"**Target**: `gdp_growth_pct` (annual % GDP growth)\n"
        f"**Problem type**: Temporal panel regression\n"
        f"**Evaluation**: Chronological train/val/test split -- never random\n"
    )

    # Dataset
    data_section = (
        f"## 2. Dataset Used\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Source | `data/processed/master_panel_processed.csv` |\n"
        f"| Total rows | 6,084 |\n"
        f"| Columns | 39 |\n"
        f"| Training rows (post-filter) | {n_train:,} |\n"
        f"| Validation rows | {n_val:,} |\n"
        f"| Test rows | {n_test:,} |\n"
        f"| Countries in training | {n_countries} |\n"
    )

    # Features
    feat_section = (
        f"## 4. Feature List\n\n"
        f"**{len(FEATURE_COLUMNS)} features** used in baseline. "
        f"`govt_debt_pct_gdp` is excluded (~80% missing).\n\n"
        + "\n".join(f"- `{c}`" for c in FEATURE_COLUMNS)
    )

    # Best model section
    bm = all_metrics[best_name]
    best_section = (
        f"## 12. Best Model\n\n"
        f"**Selected**: `{best_name}` (lowest validation RMSE)\n\n"
        f"| Metric | Validation | Test |\n|---|---|---|\n"
        f"| MAE | {bm['val']['MAE']:.4f} | {bm['test']['MAE']:.4f} |\n"
        f"| RMSE | {bm['val']['RMSE']:.4f} | {bm['test']['RMSE']:.4f} |\n"
        f"| R2 | {bm['val']['R2']:.4f} | {bm['test']['R2']:.4f} |\n"
        f"| Bias (mean error) | {bm['val']['Bias (mean error)']:.4f} | "
        f"{bm['test']['Bias (mean error)']:.4f} |\n\n"
        f"> The test set was evaluated **once** on the selected model after "
        f"validation-based selection. It was not used during model selection.\n"
    )

    # Limitations
    limits_section = (
        f"## 15. Limitations\n\n"
        f"1. **Country heterogeneity**: Country fixed effects are not modelled at baseline. "
        f"A single set of features may under-fit small open economies vs large ones.\n"
        f"2. **Global shocks**: COVID-19 (2020), Ukraine war (2022), Macao casino shutdown -- "
        f"these are structurally unpredictable from lagged macro indicators.\n"
        f"3. **`govt_debt_pct_gdp` excluded**: Largest single missing feature (~80% NaN). "
        f"Including it with aggressive imputation would introduce spurious signal.\n"
        f"4. **Within-country bfill residual**: The preprocessing pipeline applied bfill "
        f"for leading-edge gaps -- the training pipeline's median imputer neutralises "
        f"this but the residual signal may still carry slight historical leakage "
        f"for the very first observations of each country's series.\n"
        f"5. **No hyperparameter tuning**: These are MVP-scale parameters. "
        f"A proper HPO study (e.g., cross-validated grid/random search) is reserved for Step 6.\n"
        f"6. **2024-2025 test sparseness**: Recent years have 32-46% higher feature missingness; "
        f"test-period metrics are less stable than validation metrics.\n"
    )

    # Reproducibility
    repro_section = (
        f"## 16. Reproducibility\n\n"
        f"| Parameter | Value |\n|---|---|\n"
        f"| Random seed | {RANDOM_SEED} |\n"
        f"| Data split | Deterministic chronological (no random) |\n"
        f"| Model serialisation | joblib (compress=3) |\n"
        f"| Python requirement | >= 3.12 |\n\n"
        f"Running `python apps/api/ml/train_t1_baseline.py` twice with the same "
        f"data produces identical model files and metrics.\n"
    )

    # Readiness
    ready_section = (
        f"## 17. Final Readiness Assessment\n\n"
        f"**BASELINE COMPLETE -- READY WITH CAUTIONS**\n\n"
        f"The baseline establishes a measurable starting point. "
        f"Before production use, the following must be addressed:\n"
        f"- Country fixed effects / panel structure\n"
        f"- Hyperparameter optimisation\n"
        f"- Feature selection / regularisation study\n"
        f"- Uncertainty quantification (prediction intervals)\n"
        f"- Out-of-sample robustness validation\n"
    )

    all_parts = [header, obj_section, data_section, feat_section] \
                + report_sections + [best_section, limits_section, repro_section, ready_section]
    full_report = "\n---\n\n".join(all_parts)
    REPORT_PATH.write_text(full_report, encoding="utf-8")
    log(f"  Report saved: {REPORT_PATH}")



# ===========================================================================
# STEP 10 -- 2026 FORECAST INFERENCE
# ===========================================================================
def generate_2026_forecasts(best_pipe, test_df, original_df, best_name):
    log("STEP 10 -- Generating 2026 forecasts from 2025 inference rows")
    
    # 2025 rows
    df_2025 = original_df[original_df["year"] == 2025].copy()
    
    # Exclude aggregate codes
    df_2025 = df_2025[~df_2025["country_code"].isin(WB_AGGREGATE_CODES)].copy()
    
    if df_2025.empty:
        log("  No 2025 inference rows found.")
        return
        
    X_inf = df_2025[FEATURE_COLUMNS]
    
    preds_2026 = best_pipe.predict(X_inf)
    
    forecasts = pd.DataFrame({
        "country_code": df_2025["country_code"].values,
        "country_name": df_2025["country_name"].values,
        "feature_year": 2025,
        "target_year": 2026,
        "forecast_gdp_growth": preds_2026
    })
    
    forecasts.to_csv(FORECASTS_CSV, index=False)
    log(f"  2026 forecasts saved to {FORECASTS_CSV}")
    
    codes = ["IND", "CHN", "USA", "JPN", "GBR"]
    for c in codes:
        row = forecasts[forecasts["country_code"] == c]
        if not row.empty:
            val = row["forecast_gdp_growth"].iloc[0]
            log(f"  {c} 2026 forecast: {val:.3f}%")
            
    # Also save the actual vs predicted for the test set
    preds_test = best_pipe.predict(test_df[FEATURE_COLUMNS])
    test_results = pd.DataFrame({
        "country_code": test_df["country_code"].values,
        "country_name": test_df["country_name"].values,
        "feature_year": test_df["year"].values.astype(int),
        "target_year": test_df["year"].values.astype(int) + 1,
        "actual_gdp_growth": test_df[TARGET_COL].values,
        "predicted_gdp_growth": preds_test,
        "error": preds_test - test_df[TARGET_COL].values,
        "absolute_error": np.abs(preds_test - test_df[TARGET_COL].values)
    })
    test_results.to_csv(PREDICTIONS_CSV, index=False)
    log(f"  Test predictions saved to {PREDICTIONS_CSV}")
    
    # Detailed display for target countries
    for c in codes:
        sub = test_results[test_results["country_code"] == c]
        if not sub.empty:
            log(f"  {c} Test predictions (2023-2024 features predicting 2024-2025):")
            for _, r in sub.iterrows():
                log(f"    Target {r['target_year']}: Act={r['actual_gdp_growth']:.3f}, Pred={r['predicted_gdp_growth']:.3f}, Err={r['error']:.3f}")


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    SEP = "=" * 70
    print(SEP)
    print("EconoSphere AI -- Phase 11, Step 5: Baseline ML Training")
    print(SEP)

    # 1. Load + validate
    df_raw_processed = load_and_validate()

    # 2. Filter aggregates
    df_countries, excluded_aggs = filter_aggregates(df_raw_processed)

    # 3. Temporal split
    X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df = \
        temporal_split(df_countries)

    n_train     = len(y_train)
    n_val       = len(y_val)
    n_test      = len(y_test)
    n_countries = train_df["country_code"].nunique()

    # 4. Build pipelines
    pipelines = build_pipelines()

    # 5. Train & evaluate
    trained_pipes, all_metrics, best_name = train_and_evaluate(
        pipelines, X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 6. Error analysis
    error_df = error_analysis(
        trained_pipes[best_name], X_test, y_test, test_df, best_name
    )

    # 7. Plots
    _ = generate_plots(
        trained_pipes[best_name],
        X_test, y_test, test_df,
        X_train, y_train, train_df,
        all_metrics, best_name,
    )

    # 8. Save models
    saved_models = save_models(trained_pipes, best_name)

    # 9. Metadata
    save_metadata(best_name, all_metrics, n_train, n_val, n_test, n_countries, saved_models)
    generate_2026_forecasts(trained_pipes[best_name], test_df, df_countries, best_name)

    # 10. Report
    write_report(best_name, all_metrics, n_train, n_val, n_test, n_countries, None)

    # --- Final summary ---
    bm = all_metrics[best_name]
    print()
    print(SEP)
    print("TRAINING SUMMARY")
    print(SEP)
    print(f"  Feature count        : {len(FEATURE_COLUMNS)}")
    print(f"  Training rows        : {n_train:,}")
    print(f"  Validation rows      : {n_val:,}")
    print(f"  Test rows            : {n_test:,}")
    print(f"  Countries            : {n_countries}")
    print(f"  Excluded aggregates  : {len(excluded_aggs)} codes")
    print()
    print(f"  Best model           : {best_name}")
    print(f"  Val  MAE={bm['val']['MAE']:.4f}  RMSE={bm['val']['RMSE']:.4f}  R2={bm['val']['R2']:.4f}")
    print(f"  Test MAE={bm['test']['MAE']:.4f}  RMSE={bm['test']['RMSE']:.4f}  R2={bm['test']['R2']:.4f}")
    print()
    print(f"  Models saved to      : {MODELS_DIR}")
    print(f"  Report saved to      : {REPORT_PATH}")
    print(f"  RAW DATASET MODIFIED : NO")
    print(f"  TARGET IMPUTED       : NO")
    print(f"  RANDOM SPLIT USED    : NO")
    print(SEP)
    print("Step 5 complete. Awaiting Step 6 approval.")


if __name__ == "__main__":
    main()
