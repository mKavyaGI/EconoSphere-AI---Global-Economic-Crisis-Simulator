import json
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"

OUT_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "phase12"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SHAP_PATH = OUT_DIR / "phase12_step5_shap_global_importance.csv"
REGIME_SHAP_PATH = OUT_DIR / "phase12_step5_shap_regime_importance.csv"
VAL_PREDS_PATH = OUT_DIR / "phase12_step5_shap_validation_predictions.csv"
WF_SHAP_PATH = OUT_DIR / "phase12_step5_shap_walk_forward_importance.csv"
CTRL_VS_CAND_PATH = OUT_DIR / "phase12_step5_control_vs_candidate.csv"
EXP_FORECAST_PATH = OUT_DIR / "phase12_step5_2026_experimental_explanations.csv"
META_PATH = MODELS_DIR / "step5_shap_metadata.json"

TRAIN_YEARS = (2000, 2018)
VAL_YEARS = (2019, 2022)
TEST_YEARS = (2023, 2024)

WB_AGGREGATES = {
    'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'
}

def get_md5(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def construct_regimes(df, train_mask):
    df = df.copy()
    train_data = df[train_mask]
    
    low_thresh = train_data["gdp_growth_lag1"].quantile(0.33)
    high_thresh = train_data["gdp_growth_lag1"].quantile(0.67)
    
    def assign_growth(val):
        if pd.isna(val): return "MODERATE"
        if val <= low_thresh: return "LOW"
        if val >= high_thresh: return "HIGH"
        return "MODERATE"
        
    df["growth_regime"] = df["gdp_growth_lag1"].apply(assign_growth)
    
    vol_thresh = train_data["gdp_growth_rolling_std_3"].quantile(0.80)
    
    def assign_vol(val):
        if pd.isna(val): return "NORMAL"
        return "STRESS" if val >= vol_thresh else "NORMAL"
        
    df["stress_regime"] = df["gdp_growth_rolling_std_3"].apply(assign_vol)
    
    df["growth_regime_num"] = df["growth_regime"].map({"LOW": 0, "MODERATE": 1, "HIGH": 2})
    df["stress_regime_num"] = df["stress_regime"].map({"NORMAL": 0, "STRESS": 1})
    
    return df

def train_model(X_train, y_train):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.05, max_depth=5, max_iter=300, 
            l2_regularization=5.0, random_state=42
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe

class DeterministicMarginalExplainer:
    """
    A deterministic SHAP-compatible fallback. 
    Because the official SHAP package strictly requires numba <3.14 and fails to install on Python 3.14.6,
    we implement a deterministic marginal contribution explainer.
    This approximates SHAP values by calculating the marginal effect of feature i
    relative to a background dataset B.
    phi_i(x) = E_{b in B}[ f(x) - f(x_{i=b_i}) ]
    """
    def __init__(self, model_predict, background_data):
        self.predict = model_predict
        self.bg = background_data
        
    def __call__(self, X):
        X = np.array(X)
        N, M = X.shape
        shap_values = np.zeros((N, M))
        
        # Base prediction for x
        base_preds = self.predict(X)
        
        np.random.seed(42)
        bg_sample_size = min(100, len(self.bg))
        bg_idx = np.random.choice(len(self.bg), size=bg_sample_size, replace=False)
        bg_subset = self.bg[bg_idx]
        
        for i in range(M):
            # Tile X bg_sample_size times: Shape (N * bg_sample_size, M)
            X_tiled = np.tile(X, (bg_sample_size, 1))
            
            # Tile bg_subset column i N times: Shape (N * bg_sample_size,)
            # e.g., if bg_subset is [b1, b2], we want [b1, b1.. b2, b2..]
            b_vals = np.repeat(bg_subset[:, i], N)
            
            X_tiled[:, i] = b_vals
            
            # Predict in one batch
            preds = self.predict(X_tiled)
            
            # Reshape to (bg_sample_size, N) and average
            preds_matrix = preds.reshape(bg_sample_size, N)
            avg_replaced_preds = preds_matrix.mean(axis=0)
            
            # Marginal drop
            shap_values[:, i] = base_preds - avg_replaced_preds
            
        # Distribute remaining interaction variance proportionally
        expected_val = self.predict(bg_subset).mean()
        sum_shap = shap_values.sum(axis=1)
        diff = (base_preds - expected_val) - sum_shap
        
        shap_values += (diff[:, None] / M)
        return shap_values

def analyze_shap_global(shap_values, feature_names):
    mean_abs = np.abs(shap_values).mean(axis=0)
    median_abs = np.median(np.abs(shap_values), axis=0)
    mean_signed = shap_values.mean(axis=0)
    
    ranks = np.argsort(-mean_abs) + 1 # 1-based rank
    
    res = []
    for i, f in enumerate(feature_names):
        group = "BASE"
        if "regime" in f:
            group = f.upper()
        res.append({
            "feature": f,
            "mean_abs_shap": mean_abs[i],
            "median_abs_shap": median_abs[i],
            "mean_signed_shap": mean_signed[i],
            "rank": int(np.where(ranks == i+1)[0][0] + 1),
            "feature_group": group
        })
    df = pd.DataFrame(res).sort_values("rank")
    return df

def analyze_shap_regime(shap_values, X_df, feature_names, regime_col):
    res = []
    for regime in X_df[regime_col].unique():
        idx = X_df[regime_col] == regime
        if idx.sum() == 0: continue
        
        sv = shap_values[idx]
        mean_abs = np.abs(sv).mean(axis=0)
        mean_signed = sv.mean(axis=0)
        median_abs = np.median(np.abs(sv), axis=0)
        ranks = np.argsort(-mean_abs) + 1
        
        for i, f in enumerate(feature_names):
            res.append({
                "regime_type": regime_col.replace("_num", ""),
                "regime_value": regime,
                "sample_count": int(idx.sum()),
                "feature": f,
                "mean_abs_shap": mean_abs[i],
                "median_abs_shap": median_abs[i],
                "mean_signed_shap": mean_signed[i],
                "rank": int(np.where(ranks == i+1)[0][0] + 1)
            })
    return pd.DataFrame(res)

def main():
    print("STEP 5: SHAP Explainability & Regime Feature Audit")
    
    raw_md5 = get_md5(RAW_PATH)
    if raw_md5 != "8ac7e0b2bf09fbe89289f82d0c7cf25e":
        raise ValueError(f"Raw MD5 mismatch: {raw_md5}")
        
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    valid_target = df["next_year_target_available"] == 1
    
    train_mask = (df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1]) & valid_target
    val_mask = (df["year"] >= VAL_YEARS[0]) & (df["year"] <= VAL_YEARS[1]) & valid_target
    test_mask = (df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1]) & valid_target
    
    df = construct_regimes(df, train_mask)
    
    base_features = [
        "exchange_rate_lcu_usd", "tariff_rate_pct", "remittances_usd", "fdi_net_inflow_usd", 
        "unemployment_pct", "imports_pct_gdp", "tax_revenue_pct_gdp", "exports_pct_gdp", 
        "interest_rate_pct", "reserves_usd", "current_account_pct_gdp", "inflation_cpi_pct", 
        "population_total", "gdp_current_usd", "gdp_growth_lag1", "gdp_growth_lag2", 
        "gdp_growth_lag3", "inflation_lag1", "unemployment_lag1", "exports_lag1", 
        "imports_lag1", "gdp_growth_rolling_mean_3", "gdp_growth_rolling_std_3", 
        "gdp_growth_rolling_mean_5", "inflation_rolling_mean_3", "trade_openness", 
        "trade_balance_ratio", "log_gdp_usd", "log_population", "gdp_growth_rolling_std_5", "inflation_rolling_std_3"
    ]
    candidate_features = base_features + ["growth_regime_num", "stress_regime_num"]
    
    train = df[train_mask].copy()
    val = df[val_mask].copy()
    
    X_train = train[candidate_features]
    y_train = train["gdp_growth_next_year"].values
    X_val = val[candidate_features]
    y_val = val["gdp_growth_next_year"].values
    
    pipe = train_model(X_train, y_train)
    val_preds = pipe.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE Check: {val_rmse:.4f}")
    
    if not np.isclose(val_rmse, 8.2598, atol=0.01):
        print(f"ERROR: Expected Validation RMSE ~8.2598, got {val_rmse:.4f}")
        return
        
    imputer = pipe.named_steps["imputer"]
    model = pipe.named_steps["model"]
    
    X_train_imp = imputer.transform(X_train)
    X_val_imp = imputer.transform(X_val)
    
    print("Using Custom Deterministic Marginal Explainer (SHAP Fallback)")
    explainer = DeterministicMarginalExplainer(model.predict, X_train_imp)
    explainer_method = "DeterministicMarginalExplainer (Custom SHAP Fallback due to numba/Python 3.14 incompatibility)"
    
    sv_data = explainer(X_val_imp)
    
    df_global = analyze_shap_global(sv_data, candidate_features)
    df_global.to_csv(GLOBAL_SHAP_PATH, index=False)
    
    df_regime_g = analyze_shap_regime(sv_data, val, candidate_features, "growth_regime_num")
    df_regime_s = analyze_shap_regime(sv_data, val, candidate_features, "stress_regime_num")
    df_regime = pd.concat([df_regime_g, df_regime_s], ignore_index=True)
    df_regime.to_csv(REGIME_SHAP_PATH, index=False)
    
    val_out = val[["country_code", "year", "gdp_growth_next_year", "growth_regime", "stress_regime"]].copy()
    val_out.rename(columns={"year": "feature_year", "gdp_growth_next_year": "actual_growth"}, inplace=True)
    val_out["target_year"] = val_out["feature_year"] + 1
    val_out["predicted_growth"] = val_preds
    val_out["error"] = val_preds - val_out["actual_growth"]
    
    g_idx = candidate_features.index("growth_regime_num")
    s_idx = candidate_features.index("stress_regime_num")
    val_out["shap_growth_regime"] = sv_data[:, g_idx]
    val_out["shap_stress_regime"] = sv_data[:, s_idx]
    val_out.to_csv(VAL_PREDS_PATH, index=False)
    
    windows = list(range(2013, 2025))
    wf_res = []
    
    for target_year in windows:
        f_year = target_year - 1
        t_mask = (df["year"] < f_year) & valid_target
        v_mask = (df["year"] == f_year) & valid_target
        if v_mask.sum() == 0: continue
        
        df_w = construct_regimes(df, t_mask)
        X_t = df_w[t_mask][candidate_features]
        y_t = df_w[t_mask]["gdp_growth_next_year"].values
        X_v = df_w[v_mask][candidate_features]
        
        p = train_model(X_t, y_t)
        imp = p.named_steps["imputer"]
        mod = p.named_steps["model"]
        
        X_t_i = imp.transform(X_t)
        X_v_i = imp.transform(X_v)
        
        ex_w = DeterministicMarginalExplainer(mod.predict, X_t_i)
        sv_w = ex_w(X_v_i)
        
        df_w_g = analyze_shap_global(sv_w, candidate_features)
        df_w_g["target_year"] = target_year
        wf_res.append(df_w_g)
        
    df_wf = pd.concat(wf_res, ignore_index=True)
    df_wf.to_csv(WF_SHAP_PATH, index=False)
    
    metrics_path = PROJECT_ROOT / "data" / "processed" / "phase12_step4_experiment_metrics.csv"
    step4_meta_path = MODELS_DIR / "step4_regime_aware_metadata.json"
    
    df_m = pd.read_csv(metrics_path)
    ctrl = df_m[df_m["experiment"] == "A. Control"].iloc[0]
    cand = df_m[df_m["experiment"] == "B. Regime Features"].iloc[0]
    
    with open(step4_meta_path, "r") as f:
        s4meta = json.load(f)
        
    cvc = [{
        "Model": "Phase 11 Control",
        "Validation_RMSE": ctrl["validation_rmse"],
        "Test_RMSE": ctrl["test_rmse"],
        "WalkForward_RMSE": s4meta["control_walk_forward_RMSE"],
        "Recession_WF_RMSE": s4meta["control_recession_walk_forward_RMSE"]
    }, {
        "Model": "Candidate B",
        "Validation_RMSE": cand["validation_rmse"],
        "Test_RMSE": cand["test_rmse"],
        "WalkForward_RMSE": s4meta["candidate_walk_forward_RMSE"],
        "Recession_WF_RMSE": s4meta["candidate_recession_walk_forward_RMSE"]
    }]
    pd.DataFrame(cvc).to_csv(CTRL_VS_CAND_PATH, index=False)
    
    f2026_mask = (df["year"] == 2025)
    df_f = construct_regimes(df, df["year"] <= 2025)
    
    X_f = df_f[f2026_mask][candidate_features]
    countries = df_f[f2026_mask]["country_code"].values
    
    target_c = ["IND", "CHN", "USA", "JPN", "GBR"]
    f_res = []
    
    if len(X_f) > 0:
        train_full_mask = (df["year"] <= 2024) & valid_target
        pipe_full = train_model(df_f[train_full_mask][candidate_features], df_f[train_full_mask]["gdp_growth_next_year"])
        imp_f = pipe_full.named_steps["imputer"]
        mod_f = pipe_full.named_steps["model"]
        
        preds_f = pipe_full.predict(X_f)
        X_f_i = imp_f.transform(X_f)
        
        ex_f = DeterministicMarginalExplainer(mod_f.predict, imp_f.transform(df_f[train_full_mask][candidate_features]))
        sv_f = ex_f(X_f_i)
        
        for i, c in enumerate(countries):
            if c in target_c:
                top_idx = np.argsort(-sv_f[i])[:10]
                bot_idx = np.argsort(sv_f[i])[:10]
                top_f = [candidate_features[idx] for idx in top_idx]
                bot_f = [candidate_features[idx] for idx in bot_idx]
                
                f_res.append({
                    "country_code": c,
                    "target_year": 2026,
                    "forecast": preds_f[i],
                    "growth_regime": df_f.loc[f2026_mask].iloc[i]["growth_regime"],
                    "stress_regime": df_f.loc[f2026_mask].iloc[i]["stress_regime"],
                    "growth_regime_shap": sv_f[i, g_idx],
                    "stress_regime_shap": sv_f[i, s_idx],
                    "top_positive_contributors": ",".join(top_f),
                    "top_negative_contributors": ",".join(bot_f),
                    "label": "EXPERIMENTAL — NOT PRODUCTION"
                })
        pd.DataFrame(f_res).to_csv(EXP_FORECAST_PATH, index=False)
    
    growth_rank = df_global[df_global["feature"] == "growth_regime_num"]["rank"].values[0]
    stress_rank = df_global[df_global["feature"] == "stress_regime_num"]["rank"].values[0]
    
    meaningful = (growth_rank <= 15) or (stress_rank <= 15)
    decision = "EXPLAINABILITY CONFIRMED" if meaningful else "EXPLAINABILITY INCONCLUSIVE"
    
    meta = {
        "phase": 12,
        "step": 5,
        "model_name": "Phase 12 Step 4 Candidate B (Regime Features)",
        "model_parameters": {
            "learning_rate": 0.05,
            "max_depth": 5,
            "max_iter": 300,
            "l2_regularization": 5.0,
            "random_state": 42
        },
        "candidate_b_definition": "Phase 11 features + growth_regime_num + stress_regime_num",
        "raw_dataset_MD5": raw_md5,
        "shap_library_version": "None (Custom Fallback)",
        "explainer_type": "DeterministicMarginalExplainer",
        "fallback_method_if_used": explainer_method,
        "random_seed": 42,
        "train_period": TRAIN_YEARS,
        "validation_period": VAL_YEARS,
        "test_period": TEST_YEARS,
        "walk_forward_periods": "2013-2024",
        "candidate_b_validation_RMSE": float(cand["validation_rmse"]),
        "candidate_b_test_RMSE": float(cand["test_rmse"]),
        "candidate_b_walk_forward_RMSE": float(s4meta["candidate_walk_forward_RMSE"]),
        "candidate_b_recession_RMSE": float(s4meta["candidate_recession_walk_forward_RMSE"]),
        "growth_regime_global_rank": int(growth_rank),
        "stress_regime_global_rank": int(stress_rank),
        "final_explainability_decision": decision,
        "production_model_modified": False,
        "phase11_artifacts_modified": False,
        "phase12_previous_artifacts_modified": False
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Done. Explanability decision: {decision}")

if __name__ == "__main__":
    main()
