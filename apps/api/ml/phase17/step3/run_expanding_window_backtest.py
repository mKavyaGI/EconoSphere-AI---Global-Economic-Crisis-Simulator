import os
import sys
import re
import json
import hashlib
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# --- Constants & Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PHASE17_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PHASE17_DIR.parents[3]

MODELS_DIR = PROJECT_ROOT / "models" / "phase11"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = MODELS_DIR / "best_t1_gdp_growth_model.joblib"
MANIFEST_PATH = MODELS_DIR / "phase11_production_release_manifest.json"
DATA_PATH = DATA_DIR / "processed" / "master_panel_t1_missingness.csv"
FRONTEND_PAGE_PATH = PROJECT_ROOT / "apps" / "web" / "src" / "app" / "dashboard" / "forecast" / "page.tsx"

EXPERIMENTAL_DIR = SCRIPT_DIR / "experimental_artifacts"
REPORT_DIR = PROJECT_ROOT / "docs" / "model_accuracy" / "phase17" / "step3"

WB_AGGREGATES = {'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'ECA', 'ECS', 'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'}
PRIORITY_COUNTRIES = ["GBR", "BRA", "FRA", "CAN", "AUS"]
GUARDRAIL_COUNTRIES = ["USA", "CHN", "DEU", "JPN", "IND"]

BASE_FEATURES = [
    "exchange_rate_lcu_usd", "tariff_rate_pct", "remittances_usd", "fdi_net_inflow_usd", 
    "unemployment_pct", "imports_pct_gdp", "tax_revenue_pct_gdp", "exports_pct_gdp", 
    "interest_rate_pct", "reserves_usd", "current_account_pct_gdp", "inflation_cpi_pct", 
    "population_total", "gdp_current_usd", "gdp_growth_lag1", "gdp_growth_lag2", 
    "gdp_growth_lag3", "inflation_lag1", "unemployment_lag1", "exports_lag1", 
    "imports_lag1", "gdp_growth_rolling_mean_3", "gdp_growth_rolling_std_3", 
    "gdp_growth_rolling_mean_5", "inflation_rolling_mean_3", "trade_openness", 
    "trade_balance_ratio", "log_gdp_usd", "log_population", "gdp_growth_rolling_std_5", 
    "inflation_rolling_std_3"
]

LOCKED_PARAMS = {
    'l2_regularization': 5.0, 
    'learning_rate': 0.05, 
    'max_depth': 5, 
    'max_iter': 300,
    'random_state': 42
}

def get_hashes(filepath: Path):
    if not filepath.exists(): return None, None
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()

def get_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "R2": np.nan, "N": 0}
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "R2": np.nan, "N": 0}
    return {
        "MAE": mean_absolute_error(y_t, y_p),
        "RMSE": np.sqrt(mean_squared_error(y_t, y_p)),
        "Bias": np.mean(y_p - y_t),
        "R2": r2_score(y_t, y_p) if len(y_t) > 1 else np.nan,
        "N": len(y_t)
    }

def get_pct_improvement(baseline_rmse, candidate_rmse):
    if pd.isna(baseline_rmse) or baseline_rmse == 0:
        return "PERCENTAGE_IMPROVEMENT_NOT_DEFINED"
    return ((baseline_rmse - candidate_rmse) / baseline_rmse) * 100

def get_supported_countries():
    if FRONTEND_PAGE_PATH.exists():
        content = FRONTEND_PAGE_PATH.read_text(encoding="utf-8")
        match = re.search(r'const\s+SUPPORTED_COUNTRIES\s*=\s*\[(.*?)\];', content)
        if match:
            sc = re.findall(r'["\'](.*?)["\']', match.group(1))
            return sc
    return PRIORITY_COUNTRIES + GUARDRAIL_COUNTRIES

def evaluate_predictions(df_eval, y_pred_col, y_true_col="gdp_growth_next_year"):
    met = get_metrics(df_eval[y_true_col].values, df_eval[y_pred_col].values)
    p_eval = df_eval[df_eval["country_code"].isin(PRIORITY_COUNTRIES)]
    g_eval = df_eval[df_eval["country_code"].isin(GUARDRAIL_COUNTRIES)]
    p_met = get_metrics(p_eval[y_true_col].values, p_eval[y_pred_col].values)
    g_met = get_metrics(g_eval[y_true_col].values, g_eval[y_pred_col].values)
    
    country_metrics = {}
    for c in df_eval["country_code"].unique():
        c_df = df_eval[df_eval["country_code"] == c]
        c_met = get_metrics(c_df[y_true_col].values, c_df[y_pred_col].values)
        if c_met["N"] > 0:
            country_metrics[c] = c_met
            
    return met, p_met, g_met, country_metrics

def main():
    print("=======================================================================")
    print("PHASE 17 STEP 3: Expanding-Window Backtesting & Promotion Evaluation")
    print("=======================================================================")
    
    EXPERIMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    (EXPERIMENTAL_DIR / "README_EXPERIMENTAL.md").write_text(
        "WARNING:\n"
        "All models and outputs in this directory are EXPERIMENTAL_ONLY.\n"
        "These artifacts must not be loaded by the production forecasting API.\n"
        "These artifacts do not replace the Phase 11 production model.\n"
        "Promotion requires a separate explicitly approved production-governance phase.\n"
    )
    
    pre_model_md5, pre_model_sha256 = get_hashes(MODEL_PATH)
    pre_man_md5, pre_man_sha256 = get_hashes(MANIFEST_PATH)
    pre_data_md5, pre_data_sha256 = get_hashes(DATA_PATH)
    
    supported_countries = get_supported_countries()
    
    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw[~df_raw["country_code"].isin(WB_AGGREGATES)].copy()
    
    # 1. Target Availability & Fold Generation
    years = sorted(df["year"].unique())
    target_audit_rows = []
    
    valid_target_years = []
    for yr in years:
        yr_df = df[df["year"] == yr]
        total_n = len(yr_df)
        valid_n = yr_df["gdp_growth_next_year"].notna().sum()
        pct_cov = (valid_n / total_n) * 100 if total_n > 0 else 0
        target_audit_rows.append({
            "feature_year": int(yr),
            "target_year": int(yr) + 1,
            "total_rows": total_n,
            "labeled_rows": int(valid_n),
            "coverage_pct": round(pct_cov, 2)
        })
        if pct_cov >= 95.0:
            valid_target_years.append(int(yr))
            
    folds = []
    for f_year in valid_target_years:
        if f_year >= 2017:
            train_candidates = [y for y in valid_target_years if y < f_year]
            if not train_candidates:
                continue
            max_train_f_year = max(train_candidates)
            folds.append({
                "eval_feature_year": f_year,
                "eval_target_year": f_year + 1,
                "train_cutoff_feature_year": max_train_f_year
            })
            
    audit_df = pd.DataFrame(target_audit_rows)
    (REPORT_DIR / "phase17_step3_target_availability_audit.md").write_text(
        "# Phase 17 Step 3: Target Availability Audit\n\n"
        f"{audit_df.to_markdown(index=False)}\n\n"
        "## Selected Folds\n"
        f"{pd.DataFrame(folds).to_markdown(index=False)}\n"
    )
    
    # 2. Methodologies Evaluation
    print("\n--- Evaluating Models ---")
    results = []
    fold_details = []
    
    m0_model = joblib.load(MODEL_PATH)
    
    for fold in folds:
        f_eval = fold['eval_feature_year']
        f_train_end = fold['train_cutoff_feature_year']
        t_eval = fold['eval_target_year']
        
        train_df = df[(df["year"] <= f_train_end) & (df["gdp_growth_next_year"].notna())].copy()
        test_df = df[(df["year"] == f_eval) & (df["gdp_growth_next_year"].notna())].copy()
        
        sample_to_feature = len(train_df) / len(BASE_FEATURES)
        risk_flag = "[HIGH_DIMENSIONALITY_SMALL_SAMPLE_RISK]" if sample_to_feature < 10 else ""
        
        # M0: Frozen Phase 11
        test_df["M0_pred"] = m0_model.predict(test_df[BASE_FEATURES])
        m0_met, m0_p_met, m0_g_met, m0_c_met = evaluate_predictions(test_df, "M0_pred")
        
        # M1: Expanding Window Refit
        m1_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(**LOCKED_PARAMS))
        ])
        m1_pipe.fit(train_df[BASE_FEATURES], train_df["gdp_growth_next_year"])
        test_df["M1_pred"] = m1_pipe.predict(test_df[BASE_FEATURES])
        m1_met, m1_p_met, m1_g_met, m1_c_met = evaluate_predictions(test_df, "M1_pred")
        
        m1_filename = f"expanding_window_refit_2000_{f_train_end}_EXPERIMENTAL_ONLY.joblib"
        joblib.dump(m1_pipe, EXPERIMENTAL_DIR / m1_filename)
        
        # N0: Naive Persistence
        test_df["N0_pred"] = test_df["gdp_growth_lag1"]
        n0_met, n0_p_met, n0_g_met, n0_c_met = evaluate_predictions(test_df, "N0_pred")
        
        # N1: Dummy Mean
        train_mean = train_df["gdp_growth_next_year"].mean()
        test_df["N1_pred"] = train_mean
        n1_met, n1_p_met, n1_g_met, n1_c_met = evaluate_predictions(test_df, "N1_pred")
        
        fold_results = {
            "Fold Target Year": t_eval,
            "Train Window": f"2000-{f_train_end}",
            "N_train": len(train_df),
            "N_test": len(test_df),
            "M0 RMSE": m0_met["RMSE"],
            "M0 MAE": m0_met["MAE"],
            "M1 RMSE": m1_met["RMSE"],
            "M1 MAE": m1_met["MAE"],
            "N0 RMSE": n0_met["RMSE"],
            "N1 RMSE": n1_met["RMSE"],
            "M0 Priority": m0_p_met["RMSE"],
            "M1 Priority": m1_p_met["RMSE"],
            "M0 Guardrail": m0_g_met["RMSE"],
            "M1 Guardrail": m1_g_met["RMSE"],
            "Sample/Feature": round(sample_to_feature, 1),
            "Risk": risk_flag
        }
        results.append(fold_results)
        
        for c in test_df["country_code"].unique():
            fold_details.append({
                "Target Year": t_eval,
                "Country": c,
                "M0 RMSE": m0_c_met.get(c, {}).get("RMSE", np.nan),
                "M0 MAE": m0_c_met.get(c, {}).get("MAE", np.nan),
                "M1 RMSE": m1_c_met.get(c, {}).get("RMSE", np.nan),
                "M1 MAE": m1_c_met.get(c, {}).get("MAE", np.nan),
                "N0 RMSE": n0_c_met.get(c, {}).get("RMSE", np.nan),
                "N1 RMSE": n1_c_met.get(c, {}).get("RMSE", np.nan),
            })

    res_df = pd.DataFrame(results)
    det_df = pd.DataFrame(fold_details)
    
    (REPORT_DIR / "phase17_step3_fold_level_results.md").write_text(
        "# Phase 17 Step 3: Fold Level Results\n\n"
        f"{res_df.to_markdown(index=False)}\n\n"
        "## Country Level Details\n"
        f"{det_df.to_markdown(index=False)}\n"
    )
    
    (REPORT_DIR / "phase17_step3_backtest_design.md").write_text(
        "# Phase 17 Step 3: Backtest Design\n\n"
        "- **Research Question**: Does expanding the training window consistently improve forecasting accuracy?\n"
        "- **Methodology**: Chronological expanding-window backtests dynamically discovering target availability.\n"
        "- **Leakage Prevention**: Training only uses features and targets fully available prior to the forecast year. No future data allowed in training or preprocessing.\n"
        "- **M0**: Frozen Phase 11 model.\n"
        "- **M1**: Expanding window refit using Phase 11 methodology.\n"
        "- **N0**: Naive persistence (`gdp_growth_lag1`).\n"
        "- **N1**: Training set mean.\n"
    )
    
    (REPORT_DIR / "phase17_step3_leakage_and_reproducibility_audit.md").write_text(
        "# Phase 17 Step 3: Leakage & Reproducibility Audit\n\n"
        "- **Temporal Cutoff**: Enforced by strictly using `feature_year <= train_cutoff` for training.\n"
        "- **Preprocessing**: Imputer fitted solely on `train_df` within each fold.\n"
        "- **Test Isolation**: Targets from evaluation year are only used for metric calculation.\n"
        "- **Hash Verification**: Phase 11 artifacts verified via MD5/SHA256 before and after execution.\n"
    )
    
    m0_better_count = (res_df["M0 RMSE"] < res_df["M1 RMSE"]).sum()
    m1_better_count = (res_df["M1 RMSE"] < res_df["M0 RMSE"]).sum()
    
    m1_beats_n0 = (res_df["M1 RMSE"] < res_df["N0 RMSE"]).all()
    m1_beats_n1 = (res_df["M1 RMSE"] < res_df["N1 RMSE"]).all()
    
    m1_improves_priority = (res_df["M1 Priority"] <= res_df["M0 Priority"]).mean() >= 0.5
    m1_harms_guardrail = (res_df["M1 Guardrail"] > res_df["M0 Guardrail"] * 1.05).any()
    m1_stable = (res_df["Risk"] == "").all()
    
    criteria = [
        {"Criterion": "Consistently improves Priority", "Result": str(m1_improves_priority), "Pass/Fail": "Pass" if m1_improves_priority else "Fail"},
        {"Criterion": "No unacceptable Guardrail degradation", "Result": str(not m1_harms_guardrail), "Pass/Fail": "Pass" if not m1_harms_guardrail else "Fail"},
        {"Criterion": "Outperforms N0 baseline", "Result": str(m1_beats_n0), "Pass/Fail": "Pass" if m1_beats_n0 else "Fail"},
        {"Criterion": "Outperforms N1 baseline", "Result": str(m1_beats_n1), "Pass/Fail": "Pass" if m1_beats_n1 else "Fail"},
        {"Criterion": "Fold stability (M1 better overall)", "Result": f"M1 wins {m1_better_count}, M0 wins {m0_better_count}", "Pass/Fail": "Pass" if m1_better_count >= m0_better_count else "Fail"},
        {"Criterion": "Sample size characteristics", "Result": "Stable" if m1_stable else "High Dimensionality Risk", "Pass/Fail": "Pass" if m1_stable else "Fail"},
    ]
    crit_df = pd.DataFrame(criteria)
    
    (REPORT_DIR / "phase17_step3_expanding_window_analysis.md").write_text(
        "# Phase 17 Step 3: Expanding Window Analysis\n\n"
        f"M1 won in {m1_better_count} folds, M0 won in {m0_better_count} folds.\n"
        f"M1 beats Naive Baseline in all folds: {m1_beats_n0}\n"
    )
    
    (REPORT_DIR / "phase17_step3_promotion_evaluation.md").write_text(
        "# Phase 17 Step 3: Promotion Evaluation\n\n"
        f"{crit_df.to_markdown(index=False)}\n"
    )
    
    all_passed = (crit_df["Pass/Fail"] == "Pass").all()
    if all_passed:
        decision = "EXPERIMENTAL_REFIT_PROMISING"
        decision_text = f"EXPERIMENTAL_REFIT_PROMISING\n\nThis does not promote the experimental model to production. A separate controlled production-promotion phase is required."
    elif m1_better_count >= m0_better_count:
        decision = "EXPERIMENTAL_REFIT_INCONCLUSIVE"
        decision_text = decision
    else:
        decision = "FROZEN_PRODUCTION_RETAINED"
        decision_text = decision
        
    (REPORT_DIR / "phase17_step3_final_decision.md").write_text(
        f"# Phase 17 Step 3: Final Decision\n\n`{decision_text}`\n"
    )
    
    print("\n--- Hashes Verification ---")
    post_model_md5, post_model_sha256 = get_hashes(MODEL_PATH)
    post_man_md5, post_man_sha256 = get_hashes(MANIFEST_PATH)
    post_data_md5, post_data_sha256 = get_hashes(DATA_PATH)
    
    if (pre_model_md5 != post_model_md5 or pre_model_sha256 != post_model_sha256 or
        pre_man_md5 != post_man_md5 or pre_man_sha256 != post_man_sha256 or
        pre_data_md5 != post_data_md5 or pre_data_sha256 != post_data_sha256):
        print("[FATAL] Protected Phase 11 artifact mutated")
        sys.exit(1)
        
    print("[SUCCESS] Phase 11 hashes unchanged")
    print("[SUCCESS] Expanding-window backtests completed")

if __name__ == "__main__":
    main()
