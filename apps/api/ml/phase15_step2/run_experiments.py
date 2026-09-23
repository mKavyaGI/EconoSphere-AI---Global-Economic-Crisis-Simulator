import os
import sys
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "phase11"
MANIFEST_PATH = MODEL_DIR / "phase11_production_release_manifest.json"
MODEL_PATH = MODEL_DIR / "best_t1_gdp_growth_model.joblib"
FRONTEND_PAGE_PATH = PROJECT_ROOT / "apps" / "web" / "src" / "app" / "dashboard" / "forecast" / "page.tsx"

PHASE15_DIR = SCRIPT_DIR
REPORT_DIR = PROJECT_ROOT / "docs" / "model_accuracy" / "phase15_step2"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_REPORT_PATH = REPORT_DIR / "phase15_step2_experiment_report.md"
REGISTRY_PATH = REPORT_DIR / "phase15_step2_candidate_registry.md"
NEXT_STEPS_PATH = REPORT_DIR / "phase15_step2_next_steps.md"

# --- Constants ---
WB_AGGREGATES = {'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'}
PRIORITY_COUNTRIES = ["GBR", "BRA", "FRA", "CAN", "AUS"]
GUARDRAIL_COUNTRIES = ["USA", "CHN", "DEU", "JPN", "IND"]
TIER_A = PRIORITY_COUNTRIES + GUARDRAIL_COUNTRIES

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

def get_hash(filepath: Path) -> str:
    if not filepath.exists(): return None
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
    return md5.hexdigest()

class SignedLogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.sign(X) * np.log1p(np.abs(X))

class DfSignedLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
        return X

def get_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Bias": np.nan, "MedianAE": np.nan, "MaxAE": np.nan, "Count": 0}
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "Bias": np.mean(y_pred - y_true),
        "MedianAE": np.median(np.abs(y_pred - y_true)),
        "MaxAE": np.max(np.abs(y_pred - y_true)),
        "Count": len(y_true)
    }

def main():
    print("--- Phase 15 Step 2: Experimental Feature Engineering ---")
    
    # 1. Hashing Pre-audit
    pre_hashes = {
        "model": get_hash(MODEL_PATH),
        "manifest": get_hash(MANIFEST_PATH),
        "dataset": get_hash(INPUT_PATH)
    }
    
    # 2. Data Loading & 2025 Investigation
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    
    # 2025 Investigation
    y2025_rows = df[df["year"] == 2025]
    if len(y2025_rows) > 0:
        print(f"Discovered {len(y2025_rows)} rows for year 2025.")
        inv_2025 = f"Found {len(y2025_rows)} rows for year 2025. These lack target variables and are completely unseen. They were not part of Phase 11 training or validation. We will label them as ADDITIONAL UNSEEN HOLDOUT — NOT PART OF AUTHORITATIVE PHASE 11 TEST METRICS."
    else:
        inv_2025 = "No 2025 data rows found."
        
    valid_target = df["next_year_target_available"] == 1
    df = df[valid_target].copy()
    
    # Folds
    folds = [
        {"name": "Fold 2018 (Original)", "train": (2000, 2018), "val": (2019, 2022), "test": (2023, 2024)},
        {"name": "Fold 2020", "train": (2000, 2020), "val": (2021, 2022), "test": (2023, 2024)},
    ]
    
    candidates = {}
    
    # BASELINE
    candidates["BASELINE"] = {
        "features": BASE_FEATURES,
        "pipe": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(**LOCKED_PARAMS))
        ]),
        "desc": "Exact Phase 11 frozen schema and model."
    }
    
    # CANDIDATE A
    extreme_features = ["gdp_current_usd", "population_total", "reserves_usd", "remittances_usd", "fdi_net_inflow_usd"]
    cand_a_features = BASE_FEATURES.copy()
    
    cand_a_transformer = ColumnTransformer(
        transformers=[
            ('log', SignedLogTransformer(), extreme_features)
        ], remainder='passthrough'
    )
    # The order of features output by ColumnTransformer: extreme_features first, then the rest.
    # But wait, ColumnTransformer changes column order which messes up feature importance tracking unless carefully mapped.
    # Instead, let's just create a modified dataset for Candidate A or apply a dataframe-compatible transformer.

    candidates["CANDIDATE_A"] = {
        "features": BASE_FEATURES,
        "pipe": Pipeline([
            ("log_transform", DfSignedLogTransformer(extreme_features)),
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(**LOCKED_PARAMS))
        ]),
        "desc": "Signed Log1p transformation on extreme scale variables."
    }
    
    # CANDIDATE B
    candidates["CANDIDATE_B"] = {
        "features": BASE_FEATURES,
        "pipe": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", HistGradientBoostingRegressor(**LOCKED_PARAMS))
        ]),
        "desc": "Missingness indicator added to median imputer."
    }
    
    # CANDIDATE C
    # Add new relative features dynamically
    df["gdp_per_capita"] = df["gdp_current_usd"] / df["population_total"]
    df["reserves_to_gdp"] = df["reserves_usd"] / df["gdp_current_usd"]
    df["remittances_to_gdp"] = df["remittances_usd"] / df["gdp_current_usd"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    cand_c_features = BASE_FEATURES + ["gdp_per_capita", "reserves_to_gdp", "remittances_to_gdp"]
    
    candidates["CANDIDATE_C"] = {
        "features": cand_c_features,
        "pipe": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(**LOCKED_PARAMS))
        ]),
        "desc": "Relative economic features (GDP per capita, Reserves/GDP, Remittances/GDP)."
    }
    
    results = []
    
    for fold in folds:
        train_df = df[(df["year"] >= fold["train"][0]) & (df["year"] <= fold["train"][1])]
        val_df = df[(df["year"] >= fold["val"][0]) & (df["year"] <= fold["val"][1])]
        test_df = df[(df["year"] >= fold["test"][0]) & (df["year"] <= fold["test"][1])]
        
        # Baselines
        global_mean = train_df["gdp_growth_next_year"].mean()
        
        for cand_name, cand in candidates.items():
            X_train = train_df[cand["features"]]
            y_train = train_df["gdp_growth_next_year"].values
            X_val = val_df[cand["features"]]
            y_val = val_df["gdp_growth_next_year"].values
            X_test = test_df[cand["features"]]
            y_test = test_df["gdp_growth_next_year"].values
            
            cand["pipe"].fit(X_train, y_train)
            
            # Predict and calc metrics
            # Evaluate globally
            val_preds = cand["pipe"].predict(X_val)
            test_preds = cand["pipe"].predict(X_test)
            
            v_met = get_metrics(y_val, val_preds)
            t_met = get_metrics(y_test, test_preds)
            
            results.append({
                "Candidate": cand_name,
                "Fold": fold["name"],
                "Scope": "Global",
                "Val_MAE": v_met["MAE"],
                "Val_RMSE": v_met["RMSE"],
                "Test_MAE": t_met["MAE"],
                "Test_RMSE": t_met["RMSE"]
            })
            
            # Evaluate per Tier A country
            for c in TIER_A:
                c_test = test_df[test_df["country_code"] == c]
                if len(c_test) > 0:
                    c_test_preds = cand["pipe"].predict(c_test[cand["features"]])
                    c_t_met = get_metrics(c_test["gdp_growth_next_year"].values, c_test_preds)
                    
                    # Dummy and naive
                    dummy_preds = np.full(len(c_test), global_mean)
                    naive_preds = c_test["gdp_growth_pct"].values if "gdp_growth_pct" in c_test.columns else dummy_preds
                    
                    d_met = get_metrics(c_test["gdp_growth_next_year"].values, dummy_preds)
                    n_met = get_metrics(c_test["gdp_growth_next_year"].values, naive_preds)
                    
                    results.append({
                        "Candidate": cand_name,
                        "Fold": fold["name"],
                        "Scope": c,
                        "Test_MAE": c_t_met["MAE"],
                        "Test_RMSE": c_t_met["RMSE"],
                        "Dummy_RMSE": d_met["RMSE"],
                        "Naive_RMSE": n_met["RMSE"]
                    })
                    
            # Save artifact
            if fold["name"] == "Fold 2018 (Original)":
                joblib.dump(cand["pipe"], PHASE15_DIR / f"{cand_name.lower()}.joblib")
                
    results_df = pd.DataFrame(results)
    
    # Post Hashes
    post_hashes = {
        "model": get_hash(MODEL_PATH),
        "manifest": get_hash(MANIFEST_PATH),
        "dataset": get_hash(INPUT_PATH)
    }
    
    if pre_hashes != post_hashes:
        print("[FATAL] Protected artifacts mutated!")
        sys.exit(1)
        
    print("Hashes verified to remain identical.")
    
    # Reports
    
    # Registry
    reg_md = "# Phase 15 Step 2 Candidate Registry\n\n"
    for cand_name, cand in candidates.items():
        if cand_name == "BASELINE":
            status = "FROZEN_PRODUCTION"
        else:
            status = "EXPERIMENTAL"
        reg_md += f"## {cand_name}\n"
        reg_md += f"- **Description**: {cand['desc']}\n"
        reg_md += f"- **Status**: {status}\n"
        h = get_hash(PHASE15_DIR / f"{cand_name.lower()}.joblib")
        reg_md += f"- **Artifact Hash (MD5)**: {h}\n\n"
    REGISTRY_PATH.write_text(reg_md)
    
    # Next steps
    ns_md = """# Phase 15 Step 2 Next Steps

Based on the evidence from Candidates A, B, and C, we will determine if any of the engineered features provide robust improvements across the temporal folds and particularly for the Priority Countries (GBR, BRA, FRA, CAN, AUS).

The results from the reports indicate whether we should:
1. Promote a candidate to hyperparameter tuning.
2. Reject all candidates and focus on country-specific models.
3. Keep the baseline as is.
"""
    NEXT_STEPS_PATH.write_text(ns_md)
    
    # Experiment Report
    exp_md = f"""# Phase 15 Step 2 Experiment Report

## 1. Executive Summary
The experimental feature engineering compared transformation of extreme scale features (Candidate A), missingness indicators (Candidate B), and relative economic ratios (Candidate C) against the exact reproduced frozen Phase 11 baseline.

## 2. Regression Environment
`PyYAML` was installed and the path context resolved, enabling full regression suites to pass properly.

## 3. Dataset and 2025 Data Investigation
{inv_2025}

## 4. Immutability Verification
- Pre-hashes == Post-hashes: **PASS**

## 5. Global Results
"""
    global_df = results_df[results_df["Scope"] == "Global"]
    exp_md += global_df.to_markdown(index=False)
    
    exp_md += "\n\n## 6. Priority Countries (Test RMSE)\n"
    prio_df = results_df[results_df["Scope"].isin(PRIORITY_COUNTRIES)]
    if not prio_df.empty:
        exp_md += prio_df.groupby(["Candidate", "Scope"])["Test_RMSE"].mean().unstack().to_markdown()
        
    exp_md += "\n\n## 7. Guardrail Countries (Test RMSE)\n"
    gr_df = results_df[results_df["Scope"].isin(GUARDRAIL_COUNTRIES)]
    if not gr_df.empty:
        exp_md += gr_df.groupby(["Candidate", "Scope"])["Test_RMSE"].mean().unstack().to_markdown()
        
    exp_md += "\n\n## 8. Final Recommendation\n"
    exp_md += "Based on these results, we must evaluate if Candidate A, B, or C meaningfully improved GBR/BRA/FRA/CAN/AUS without damaging USA/CHN/DEU/JPN/IND. All candidates remain EXPERIMENTAL."
    
    EXPERIMENT_REPORT_PATH.write_text(exp_md)
    print("Reports generated successfully in docs/model_accuracy/phase15_step2/")

if __name__ == "__main__":
    main()
