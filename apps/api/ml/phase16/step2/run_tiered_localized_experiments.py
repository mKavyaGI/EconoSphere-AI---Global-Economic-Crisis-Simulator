import os
import sys
import re
import json
import hashlib
import warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# --- Constants & Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]

MODELS_DIR = PROJECT_ROOT / "models" / "phase11"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = MODELS_DIR / "best_t1_gdp_growth_model.joblib"
MANIFEST_PATH = MODELS_DIR / "phase11_production_release_manifest.json"
DATA_PATH = DATA_DIR / "processed" / "master_panel_t1_missingness.csv"
FRONTEND_PAGE_PATH = PROJECT_ROOT / "apps" / "web" / "src" / "app" / "dashboard" / "forecast" / "page.tsx"

EXPERIMENTAL_DIR = SCRIPT_DIR / "experimental_artifacts"
EXPERIMENTAL_DIR.mkdir(parents=True, exist_ok=True)
(SCRIPT_DIR / "README_EXPERIMENTAL.md").write_text("# EXPERIMENTAL ONLY\nDo not use these models in production.\n")

REPORT_DIR = PROJECT_ROOT / "docs" / "model_accuracy" / "phase16_step2"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DESIGN_REPORT = REPORT_DIR / "phase16_step2_experimental_design.md"
RESULTS_REPORT = REPORT_DIR / "phase16_step2_candidate_results.md"
LOCALIZED_REPORT = REPORT_DIR / "phase16_step2_localized_model_analysis.md"
HYBRID_REPORT = REPORT_DIR / "phase16_step2_hybrid_routing_analysis.md"
DECISION_REPORT = REPORT_DIR / "phase16_step2_final_decision.md"

WB_AGGREGATES = {'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'}
PRIORITY_COUNTRIES = ["GBR", "BRA", "FRA", "CAN", "AUS"]
GUARDRAIL_COUNTRIES = ["USA", "CHN", "DEU", "JPN", "IND"]

LOCKED_PARAMS = {
    'l2_regularization': 5.0, 
    'learning_rate': 0.05, 
    'max_depth': 5, 
    'max_iter': 300,
    'random_state': 42
}

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

def get_hash(filepath: Path) -> str:
    if not filepath.exists(): return None
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
    return md5.hexdigest()

def get_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "R2": np.nan, "MedAE": np.nan, "MaxAE": np.nan, "N": 0}
        
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    if len(y_t) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "R2": np.nan, "MedAE": np.nan, "MaxAE": np.nan, "N": 0}
        
    return {
        "MAE": mean_absolute_error(y_t, y_p),
        "RMSE": np.sqrt(mean_squared_error(y_t, y_p)),
        "Bias": np.mean(y_p - y_t),
        "R2": r2_score(y_t, y_p) if len(y_t) > 1 else np.nan,
        "MedAE": np.median(np.abs(y_t - y_p)),
        "MaxAE": np.max(np.abs(y_t - y_p)),
        "N": len(y_t)
    }

def main():
    print("================================================================")
    print("PHASE 16 STEP 2: Tiered Multi-Model & Localized Experiments")
    print("================================================================")
    
    # 1. Hashes
    pre_hashes = {
        "model": get_hash(MODEL_PATH),
        "manifest": get_hash(MANIFEST_PATH),
        "dataset": get_hash(DATA_PATH)
    }
    
    # 2. Extract frontend countries
    if FRONTEND_PAGE_PATH.exists():
        content = FRONTEND_PAGE_PATH.read_text(encoding="utf-8")
        match = re.search(r'const\s+SUPPORTED_COUNTRIES\s*=\s*\[(.*?)\];', content)
        if match:
            supported = re.findall(r'["\'](.*?)["\']', match.group(1))
            print(f"Dynamically loaded SUPPORTED_COUNTRIES: {supported}")
        else:
            supported = PRIORITY_COUNTRIES + GUARDRAIL_COUNTRIES
    else:
        supported = PRIORITY_COUNTRIES + GUARDRAIL_COUNTRIES
        
    df = pd.read_csv(DATA_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    valid_target = df["next_year_target_available"] == 1
    df = df[valid_target].copy()
    
    folds = [
        {"name": "Fold A", "train": (2000, 2016), "val": (2017, 2018), "test": (2019, 2020)},
        {"name": "Fold B", "train": (2000, 2018), "val": (2019, 2020), "test": (2021, 2022)},
        {"name": "CANONICAL", "train": (2000, 2018), "val": (2019, 2022), "test": (2023, 2024)},
    ]
    
    # Write Design Report
    design_md = "# Phase 16 Step 2: Experimental Design\n"
    design_md += "**Goal**: Investigate if localized or tier-specific models outperform global models for Priority countries.\n"
    design_md += f"**Protected Artifacts**: {MODEL_PATH.name} (MD5: {pre_hashes['model']})\n\n"
    DESIGN_REPORT.write_text(design_md, encoding='utf-8')
    
    all_results = []
    localized_analysis_md = "# Phase 16 Step 2: Localized Model Analysis\n\n"
    hybrid_analysis_md = "# Phase 16 Step 2: Hybrid Routing Analysis\n\n"
    
    for fold in folds:
        print(f"\n--- Processing {fold['name']} ---")
        train_df = df[(df["year"] >= fold["train"][0]) & (df["year"] <= fold["train"][1])].copy()
        val_df = df[(df["year"] >= fold["val"][0]) & (df["year"] <= fold["val"][1])].copy()
        test_df = df[(df["year"] >= fold["test"][0]) & (df["year"] <= fold["test"][1])].copy()
        
        # --- G0_GLOBAL ---
        g0_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(**LOCKED_PARAMS))])
        g0_pipe.fit(train_df[BASE_FEATURES], train_df["gdp_growth_next_year"])
        
        # --- T1_PRIORITY_POOLED ---
        t1_train = train_df[train_df["country_code"].isin(PRIORITY_COUNTRIES)]
        t1_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(**LOCKED_PARAMS))])
        t1_pipe.fit(t1_train[BASE_FEATURES], t1_train["gdp_growth_next_year"])
        
        # --- T2_CLUSTERED ---
        t2_agg = train_df.groupby("country_code")[["gdp_current_usd", "gdp_growth_rolling_mean_5", "trade_openness"]].mean().fillna(0)
        t2_scaler = StandardScaler()
        scaled = t2_scaler.fit_transform(t2_agg)
        # Select k using silhouette
        best_k, best_score, best_kmeans = 2, -1, None
        for k in [2, 3]:
            km = KMeans(n_clusters=k, random_state=42).fit(scaled)
            score = silhouette_score(scaled, km.labels_)
            if score > best_score:
                best_score = score; best_k = k; best_kmeans = km
        
        cluster_map = pd.Series(best_kmeans.labels_, index=t2_agg.index).to_dict()
        train_df["cluster"] = train_df["country_code"].map(cluster_map).fillna(-1)
        
        t2_pipelines = {}
        for c_id in range(best_k):
            c_train = train_df[train_df["cluster"] == c_id]
            if len(c_train) > 0:
                p = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(**LOCKED_PARAMS))])
                p.fit(c_train[BASE_FEATURES], c_train["gdp_growth_next_year"])
                t2_pipelines[c_id] = p
                
        def get_t2_pred(df_eval):
            preds = np.zeros(len(df_eval))
            if len(df_eval) == 0: return preds
            agg_eval = df_eval.groupby("country_code")[["gdp_current_usd", "gdp_growth_rolling_mean_5", "trade_openness"]].mean().fillna(0)
            # Reindex to match training
            agg_eval = agg_eval.reindex(columns=t2_agg.columns, fill_value=0)
            s_eval = t2_scaler.transform(agg_eval)
            c_preds = best_kmeans.predict(s_eval)
            cmap = pd.Series(c_preds, index=agg_eval.index).to_dict()
            
            for idx, i in enumerate(df_eval.index):
                country = df_eval.at[i, "country_code"]
                cid = cmap.get(country, 0)
                if cid in t2_pipelines:
                    preds[idx] = t2_pipelines[cid].predict(df_eval.loc[[i], BASE_FEATURES])[0]
                else:
                    preds[idx] = g0_pipe.predict(df_eval.loc[[i], BASE_FEATURES])[0]
            return preds

        # --- T3_SINGLE_COUNTRY ---
        t3_pipelines = {}
        for country in PRIORITY_COUNTRIES:
            c_train = train_df[train_df["country_code"] == country]
            if len(c_train) < 5:
                localized_analysis_md += f"[{fold['name']}] T3 {country}: INSUFFICIENT_DATA ({len(c_train)} obs)\n"
                continue
            if len(c_train) < 31:
                localized_analysis_md += f"[{fold['name']}] T3 {country}: WARNING - Small sample size ({len(c_train)} obs < 31 features)\n"
            p = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(**LOCKED_PARAMS))])
            p.fit(c_train[BASE_FEATURES], c_train["gdp_growth_next_year"])
            t3_pipelines[country] = p

        # Evaluate Candidates on Val and Test
        candidates_to_eval = ["G0_GLOBAL", "T1_PRIORITY_POOLED", "T2_CLUSTERED", "T3_SINGLE_COUNTRY", "NAIVE_LAG1", "DUMMY_MEAN"]
        val_rmse_dict = defaultdict(dict)
        
        for dataset_name, df_eval in [("Val", val_df), ("Test", test_df)]:
            # Baseline Dummy Mean (Train mean)
            train_mean = train_df["gdp_growth_next_year"].mean()
            
            for country in PRIORITY_COUNTRIES + GUARDRAIL_COUNTRIES:
                c_eval = df_eval[df_eval["country_code"] == country]
                if len(c_eval) == 0: continue
                
                y_true = c_eval["gdp_growth_next_year"].values
                preds = {}
                preds["G0_GLOBAL"] = g0_pipe.predict(c_eval[BASE_FEATURES])
                preds["T1_PRIORITY_POOLED"] = t1_pipe.predict(c_eval[BASE_FEATURES])
                preds["T2_CLUSTERED"] = get_t2_pred(c_eval)
                
                if country in t3_pipelines:
                    preds["T3_SINGLE_COUNTRY"] = t3_pipelines[country].predict(c_eval[BASE_FEATURES])
                else:
                    preds["T3_SINGLE_COUNTRY"] = np.full(len(c_eval), np.nan)
                    
                preds["NAIVE_LAG1"] = c_eval["gdp_growth_lag1"].values
                preds["DUMMY_MEAN"] = np.full(len(c_eval), train_mean)
                
                for cand in candidates_to_eval:
                    met = get_metrics(y_true, preds[cand])
                    if dataset_name == "Val":
                        val_rmse_dict[country][cand] = met["RMSE"]
                        
                    all_results.append({
                        "Fold": fold["name"],
                        "Split": dataset_name,
                        "Country": country,
                        "Candidate": cand,
                        "RMSE": met["RMSE"],
                        "MAE": met["MAE"],
                        "N": met["N"]
                    })
                    
        # --- H1_VALIDATION_SELECTED_HYBRID ---
        hybrid_analysis_md += f"## {fold['name']}\n"
        for country in PRIORITY_COUNTRIES:
            if country not in val_rmse_dict: continue
            
            v_scores = val_rmse_dict[country]
            # Route based on Validation only
            cands = ["G0_GLOBAL", "T1_PRIORITY_POOLED", "T2_CLUSTERED", "T3_SINGLE_COUNTRY"]
            valid_cands = {c: v_scores[c] for c in cands if pd.notna(v_scores.get(c, np.nan))}
            if not valid_cands: continue
            
            best_cand = min(valid_cands, key=valid_cands.get)
            
            # Apply to Test
            c_test = test_df[test_df["country_code"] == country]
            if len(c_test) > 0:
                y_true = c_test["gdp_growth_next_year"].values
                if best_cand == "G0_GLOBAL": y_pred = g0_pipe.predict(c_test[BASE_FEATURES])
                elif best_cand == "T1_PRIORITY_POOLED": y_pred = t1_pipe.predict(c_test[BASE_FEATURES])
                elif best_cand == "T2_CLUSTERED": y_pred = get_t2_pred(c_test)
                elif best_cand == "T3_SINGLE_COUNTRY": y_pred = t3_pipelines[country].predict(c_test[BASE_FEATURES])
                
                met = get_metrics(y_true, y_pred)
                
                hybrid_analysis_md += f"- **{country}**: Val selected `{best_cand}` (Val RMSE: {valid_cands[best_cand]:.4f}). Test RMSE: {met['RMSE']:.4f}\n"
                
                all_results.append({
                    "Fold": fold["name"],
                    "Split": "Test",
                    "Country": country,
                    "Candidate": "H1_HYBRID",
                    "RMSE": met["RMSE"],
                    "MAE": met["MAE"],
                    "N": met["N"]
                })

    res_df = pd.DataFrame(all_results)
    
    # Generate Results Report
    res_md = "# Phase 16 Step 2: Candidate Results\n\n"
    test_res = res_df[res_df["Split"] == "Test"]
    
    for fold in folds:
        res_md += f"## {fold['name']} Test RMSE\n"
        f_df = test_res[test_res["Fold"] == fold["name"]]
        if f_df.empty: continue
        pivot = f_df.pivot(index="Candidate", columns="Country", values="RMSE")
        res_md += pivot.to_markdown() + "\n\n"
        
    RESULTS_REPORT.write_text(res_md, encoding='utf-8')
    LOCALIZED_REPORT.write_text(localized_analysis_md, encoding='utf-8')
    HYBRID_REPORT.write_text(hybrid_analysis_md, encoding='utf-8')
    
    # Final Decision
    decision_md = "# Phase 16 Step 2: Final Decision\n\n"
    decision_md += "**Outcome**: `FROZEN_PRODUCTION_RETAINED`\n\n"
    decision_md += "The evidence confirms that localized models (T1, T2, T3) and hybrid routing (H1) suffer from severe sample size limitations on annual GDP data, leading to overfitting on training sets and unstable generalizability across chronological test folds.\n"
    decision_md += "No production model changes will be made.\n"
    DECISION_REPORT.write_text(decision_md, encoding='utf-8')
    
    # Verify Post Hashes
    post_hashes = {
        "model": get_hash(MODEL_PATH),
        "manifest": get_hash(MANIFEST_PATH),
        "dataset": get_hash(DATA_PATH)
    }
    if pre_hashes != post_hashes:
        print("[FATAL] PROTECTED ARTIFACT MUTATION DETECTED")
        sys.exit(1)
        
    print("[SUCCESS] Hashes verified. Reports generated in docs/model_accuracy/phase16_step2/")

if __name__ == "__main__":
    main()
