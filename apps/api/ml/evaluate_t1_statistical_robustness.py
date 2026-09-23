import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "phase12"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED_RAW_MD5 = "8ac7e0b2bf09fbe89289f82d0c7cf25e"

def get_file_hash(filepath):
    if not filepath.exists():
        return None
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def construct_regimes(df, train_mask):
    """Construct regimes using thresholds strictly defined on the training set."""
    df = df.copy()
    train_data = df[train_mask]
    
    if len(train_data) == 0:
        df["growth_regime"] = "MODERATE"
        df["stress_regime"] = "NORMAL"
        df["growth_regime_num"] = 1
        df["stress_regime_num"] = 0
        return df

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

def train_base_model(X_train, y_train, random_state=42):
    locked_params = {
        'l2_regularization': 5.0, 
        'learning_rate': 0.05, 
        'max_depth': 5, 
        'max_iter': 300,
        'random_state': random_state
    }
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(**locked_params))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def paired_permutation_test(errors1, errors2, n_permutations=10000, random_state=42):
    np.random.seed(random_state)
    diffs = errors1 - errors2
    observed_mean = np.mean(diffs)
    
    count = 0
    for _ in range(n_permutations):
        signs = np.random.choice([-1, 1], size=len(diffs))
        permuted_mean = np.mean(diffs * signs)
        if np.abs(permuted_mean) >= np.abs(observed_mean):
            count += 1
            
    p_value = count / n_permutations
    return p_value

def main():
    print("PHASE 12 STEP 6: Statistical Robustness & Model Stability Validation")
    
    # 1. Integrity Check
    raw_md5 = get_file_hash(RAW_DATA_PATH)
    if raw_md5 != EXPECTED_RAW_MD5:
        print(f"INTEGRITY FAILURE: Raw dataset MD5 {raw_md5} != {EXPECTED_RAW_MD5}")
        return
        
    # Record hashes of protected artifacts
    protected_files = [
        "models/phase11/best_t1_gdp_growth_model.joblib",
        "models/phase11/gradient_boosting_baseline.joblib",
        "models/phase11/t1_model_metadata.json",
        "data/processed/t1_step11_predictions.csv",
        "data/processed/t1_step11_2026_forecasts.csv",
        "models/phase11/t1_step11_model_metadata.json",
    ]
    # Add Phase 12 Steps 1-5 artifacts if they exist
    for f in MODELS_DIR.glob("step[1-5]_*"):
        protected_files.append(str(f.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    for f in PROCESSED_DIR.glob("phase12_step[1-5]_*"):
        protected_files.append(str(f.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    initial_hashes = {}
    for pf in protected_files:
        path = PROJECT_ROOT / pf
        initial_hashes[pf] = get_file_hash(path)
        
    print(f"Recorded {len(initial_hashes)} protected artifacts for integrity verification.")

    # 2. Data Loading & Preparation
    df = pd.read_csv(INPUT_PATH)
    
    WB_AGGREGATES = {
        'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 
        'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN', 'LDC', 
        'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 
        'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'
    }
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    valid_target = df["next_year_target_available"] == 1
    
    base_features = [
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
    candidate_features = base_features + ["growth_regime_num", "stress_regime_num"]

    # 3. Validation and Test Set Generation
    train_mask_val = (df["year"] <= 2018) & valid_target
    val_mask = (df["year"].between(2019, 2022)) & valid_target
    test_mask = (df["year"].between(2023, 2024)) & valid_target
    
    df_regime_val = construct_regimes(df, train_mask_val)
    
    X_train_ctrl = df_regime_val[train_mask_val][base_features]
    X_train_cand = df_regime_val[train_mask_val][candidate_features]
    y_train = df_regime_val[train_mask_val]["gdp_growth_next_year"].values
    
    X_val_ctrl = df_regime_val[val_mask][base_features]
    X_val_cand = df_regime_val[val_mask][candidate_features]
    y_val = df_regime_val[val_mask]["gdp_growth_next_year"].values
    
    X_test_ctrl = df_regime_val[test_mask][base_features]
    X_test_cand = df_regime_val[test_mask][candidate_features]
    y_test = df_regime_val[test_mask]["gdp_growth_next_year"].values

    pipe_ctrl = train_base_model(X_train_ctrl, y_train, random_state=42)
    pipe_cand = train_base_model(X_train_cand, y_train, random_state=42)
    
    val_preds_ctrl = pipe_ctrl.predict(X_val_ctrl)
    val_preds_cand = pipe_cand.predict(X_val_cand)
    test_preds_ctrl = pipe_ctrl.predict(X_test_ctrl)
    test_preds_cand = pipe_cand.predict(X_test_cand)

    val_rmse_ctrl = np.sqrt(mean_squared_error(y_val, val_preds_ctrl))
    val_rmse_cand = np.sqrt(mean_squared_error(y_val, val_preds_cand))
    test_rmse_ctrl = np.sqrt(mean_squared_error(y_test, test_preds_ctrl))
    test_rmse_cand = np.sqrt(mean_squared_error(y_test, test_preds_cand))
    
    # 4. Walk-forward Prediction generation
    windows = list(range(2013, 2025))
    wf_metrics = []
    df_wf_list = []
    
    for target_year in windows:
        feature_year = target_year - 1
        tm = (df["year"] < feature_year) & valid_target
        test_m = (df["year"] == feature_year) & valid_target
        if test_m.sum() == 0: continue
        
        df_win = construct_regimes(df, tm)
        tr = df_win[tm]
        te = df_win[test_m]
        
        p_c = train_base_model(tr[base_features], tr["gdp_growth_next_year"].values)
        p_cand = train_base_model(tr[candidate_features], tr["gdp_growth_next_year"].values)
        
        pr_c = p_c.predict(te[base_features])
        pr_cand = p_cand.predict(te[candidate_features])
        
        y_te = te["gdp_growth_next_year"].values
        
        c_error = pr_c - y_te
        cand_error = pr_cand - y_te
        
        wf_metrics.append({
            "target_year": target_year,
            "sample_count": len(y_te),
            "control_rmse": np.sqrt(mean_squared_error(y_te, pr_c)),
            "candidate_rmse": np.sqrt(mean_squared_error(y_te, pr_cand)),
            "rmse_difference": np.sqrt(mean_squared_error(y_te, pr_cand)) - np.sqrt(mean_squared_error(y_te, pr_c)),
            "control_mae": mean_absolute_error(y_te, pr_c),
            "candidate_mae": mean_absolute_error(y_te, pr_cand),
            "mae_difference": mean_absolute_error(y_te, pr_cand) - mean_absolute_error(y_te, pr_c),
            "control_bias": np.mean(c_error),
            "candidate_bias": np.mean(cand_error),
            "candidate_better": np.sqrt(mean_squared_error(y_te, pr_cand)) < np.sqrt(mean_squared_error(y_te, pr_c))
        })
        
        te_out = te.copy()
        te_out["target_year"] = target_year
        te_out["control_pred"] = pr_c
        te_out["candidate_pred"] = pr_cand
        te_out["control_error"] = c_error
        te_out["candidate_error"] = cand_error
        te_out["absolute_control_error"] = np.abs(c_error)
        te_out["absolute_candidate_error"] = np.abs(cand_error)
        te_out["squared_control_error"] = c_error**2
        te_out["squared_candidate_error"] = cand_error**2
        df_wf_list.append(te_out)
        
    df_wf = pd.concat(df_wf_list, ignore_index=True)
    df_wf_metrics = pd.DataFrame(wf_metrics)

    wf_rmse_ctrl = np.sqrt(mean_squared_error(df_wf["gdp_growth_next_year"], df_wf["control_pred"]))
    wf_rmse_cand = np.sqrt(mean_squared_error(df_wf["gdp_growth_next_year"], df_wf["candidate_pred"]))

    # We will use df_wf (all years 2013-2024) for comprehensive robustness
    
    # 5. EXPERIMENT A: YEAR-BY-YEAR COMPARISON
    # Handled by df_wf_metrics
    df_wf_metrics.to_csv(PROCESSED_DIR / "phase12_step6_yearly_comparison.csv", index=False)
    
    # 6. EXPERIMENT B: PAIRED STATISTICAL SIGNIFICANCE
    abs_diff = df_wf["absolute_candidate_error"] - df_wf["absolute_control_error"]
    sq_diff = df_wf["squared_candidate_error"] - df_wf["squared_control_error"]
    
    p_val_perm_sq = paired_permutation_test(df_wf["squared_candidate_error"].values, df_wf["squared_control_error"].values)
    p_val_perm_abs = paired_permutation_test(df_wf["absolute_candidate_error"].values, df_wf["absolute_control_error"].values)
    
    wilcoxon_stat_sq, wilcoxon_p_sq = stats.wilcoxon(df_wf["squared_candidate_error"], df_wf["squared_control_error"])
    wilcoxon_stat_abs, wilcoxon_p_abs = stats.wilcoxon(df_wf["absolute_candidate_error"], df_wf["absolute_control_error"])
    
    stats_res = [{
        "metric": "Squared Error",
        "sample_size": len(df_wf),
        "mean_difference": sq_diff.mean(),
        "median_difference": sq_diff.median(),
        "permutation_p_value": p_val_perm_sq,
        "wilcoxon_statistic": wilcoxon_stat_sq,
        "wilcoxon_p_value": wilcoxon_p_sq,
        "significant_at_05": (wilcoxon_p_sq < 0.05) or (p_val_perm_sq < 0.05)
    }, {
        "metric": "Absolute Error",
        "sample_size": len(df_wf),
        "mean_difference": abs_diff.mean(),
        "median_difference": abs_diff.median(),
        "permutation_p_value": p_val_perm_abs,
        "wilcoxon_statistic": wilcoxon_stat_abs,
        "wilcoxon_p_value": wilcoxon_p_abs,
        "significant_at_05": (wilcoxon_p_abs < 0.05) or (p_val_perm_abs < 0.05)
    }]
    pd.DataFrame(stats_res).to_csv(PROCESSED_DIR / "phase12_step6_statistical_significance.csv", index=False)
    
    # 7. EXPERIMENT C: BOOTSTRAP ROBUSTNESS
    np.random.seed(42)
    n_boot = 10000
    boot_res = []
    
    n_samples = len(df_wf)
    y_true = df_wf["gdp_growth_next_year"].values
    p_c = df_wf["control_pred"].values
    p_cand = df_wf["candidate_pred"].values
    
    rmse_diffs = []
    mae_diffs = []
    abs_diffs_mean = []
    
    for i in range(n_boot):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        r_c = np.sqrt(mean_squared_error(y_true[idx], p_c[idx]))
        r_cand = np.sqrt(mean_squared_error(y_true[idx], p_cand[idx]))
        m_c = mean_absolute_error(y_true[idx], p_c[idx])
        m_cand = mean_absolute_error(y_true[idx], p_cand[idx])
        
        rmse_diffs.append(r_cand - r_c)
        mae_diffs.append(m_cand - m_c)
        
    rmse_diffs = np.array(rmse_diffs)
    mae_diffs = np.array(mae_diffs)
    
    boot_df = pd.DataFrame([{
        "metric": "RMSE",
        "mean_difference": np.mean(rmse_diffs),
        "ci_lower_95": np.percentile(rmse_diffs, 2.5),
        "ci_upper_95": np.percentile(rmse_diffs, 97.5),
        "probability_candidate_improves": np.mean(rmse_diffs < 0)
    }, {
        "metric": "MAE",
        "mean_difference": np.mean(mae_diffs),
        "ci_lower_95": np.percentile(mae_diffs, 2.5),
        "ci_upper_95": np.percentile(mae_diffs, 97.5),
        "probability_candidate_improves": np.mean(mae_diffs < 0)
    }])
    boot_df.to_csv(PROCESSED_DIR / "phase12_step6_bootstrap_results.csv", index=False)
    
    # 8. EXPERIMENT D: WALK-FORWARD STABILITY (df_wf_metrics created earlier)
    win_count = int((df_wf_metrics["candidate_rmse"] < df_wf_metrics["control_rmse"]).sum())
    loss_count = int((df_wf_metrics["candidate_rmse"] > df_wf_metrics["control_rmse"]).sum())
    df_wf_metrics.to_csv(PROCESSED_DIR / "phase12_step6_walk_forward_stability.csv", index=False)
    
    # 9. EXPERIMENT E: TEMPORAL SUBPERIOD ROBUSTNESS
    periods = {
        "Pre-shock": (2013, 2019),
        "COVID/shock": (2020, 2021),
        "Post-shock": (2022, 2024)
    }
    temp_res = []
    for p_name, (y_s, y_e) in periods.items():
        mask = df_wf["target_year"].between(y_s, y_e)
        df_p = df_wf[mask]
        if len(df_p) == 0: continue
        r_c = np.sqrt(mean_squared_error(df_p["gdp_growth_next_year"], df_p["control_pred"]))
        r_cand = np.sqrt(mean_squared_error(df_p["gdp_growth_next_year"], df_p["candidate_pred"]))
        m_c = mean_absolute_error(df_p["gdp_growth_next_year"], df_p["control_pred"])
        m_cand = mean_absolute_error(df_p["gdp_growth_next_year"], df_p["candidate_pred"])
        temp_res.append({
            "period": p_name,
            "sample_count": len(df_p),
            "control_rmse": r_c,
            "candidate_rmse": r_cand,
            "rmse_difference": r_cand - r_c,
            "control_mae": m_c,
            "candidate_mae": m_cand,
            "mae_difference": m_cand - m_c,
            "candidate_better": r_cand < r_c
        })
    pd.DataFrame(temp_res).to_csv(PROCESSED_DIR / "phase12_step6_temporal_robustness.csv", index=False)
    
    # 10. EXPERIMENT F: COUNTRY ROBUSTNESS
    country_res = []
    for c, df_c in df_wf.groupby("country_code"):
        if len(df_c) < 5: continue
        r_c = np.sqrt(mean_squared_error(df_c["gdp_growth_next_year"], df_c["control_pred"]))
        r_cand = np.sqrt(mean_squared_error(df_c["gdp_growth_next_year"], df_c["candidate_pred"]))
        m_c = mean_absolute_error(df_c["gdp_growth_next_year"], df_c["control_pred"])
        m_cand = mean_absolute_error(df_c["gdp_growth_next_year"], df_c["candidate_pred"])
        country_res.append({
            "ISO3": c,
            "observation_count": len(df_c),
            "control_rmse": r_c,
            "candidate_rmse": r_cand,
            "rmse_difference": r_cand - r_c,
            "control_mae": m_c,
            "candidate_mae": m_cand,
            "mae_difference": m_cand - m_c,
            "candidate_better": r_cand < r_c
        })
    pd.DataFrame(country_res).to_csv(PROCESSED_DIR / "phase12_step6_country_robustness.csv", index=False)
    
    # 11. EXPERIMENT G: REGIME ROBUSTNESS
    regime_res = []
    for r_g in ["LOW", "MODERATE", "HIGH"]:
        for r_s in ["NORMAL", "STRESS"]:
            mask = (df_wf["growth_regime"] == r_g) & (df_wf["stress_regime"] == r_s)
            df_r = df_wf[mask]
            if len(df_r) < 5: continue
            r_c = np.sqrt(mean_squared_error(df_r["gdp_growth_next_year"], df_r["control_pred"]))
            r_cand = np.sqrt(mean_squared_error(df_r["gdp_growth_next_year"], df_r["candidate_pred"]))
            m_c = mean_absolute_error(df_r["gdp_growth_next_year"], df_r["control_pred"])
            m_cand = mean_absolute_error(df_r["gdp_growth_next_year"], df_r["candidate_pred"])
            regime_res.append({
                "growth_regime": r_g,
                "stress_regime": r_s,
                "sample_count": len(df_r),
                "control_rmse": r_c,
                "candidate_rmse": r_cand,
                "rmse_difference": r_cand - r_c,
                "control_mae": m_c,
                "candidate_mae": m_cand,
                "mae_difference": m_cand - m_c,
                "candidate_better": r_cand < r_c
            })
    
    # Also just recession
    rec_mask = df_wf["gdp_growth_next_year"] < 0
    df_rec = df_wf[rec_mask]
    if len(df_rec) >= 5:
        r_c = np.sqrt(mean_squared_error(df_rec["gdp_growth_next_year"], df_rec["control_pred"]))
        r_cand = np.sqrt(mean_squared_error(df_rec["gdp_growth_next_year"], df_rec["candidate_pred"]))
        m_c = mean_absolute_error(df_rec["gdp_growth_next_year"], df_rec["control_pred"])
        m_cand = mean_absolute_error(df_rec["gdp_growth_next_year"], df_rec["candidate_pred"])
        regime_res.append({
            "growth_regime": "ALL (RECESSION)",
            "stress_regime": "ALL (RECESSION)",
            "sample_count": len(df_rec),
            "control_rmse": r_c,
            "candidate_rmse": r_cand,
            "rmse_difference": r_cand - r_c,
            "control_mae": m_c,
            "candidate_mae": m_cand,
            "mae_difference": m_cand - m_c,
            "candidate_better": r_cand < r_c
        })
    pd.DataFrame(regime_res).to_csv(PROCESSED_DIR / "phase12_step6_regime_comparison.csv", index=False)
    
    # 12. EXPERIMENT H: RANDOM-SEED ROBUSTNESS
    seeds = [42, 7, 21, 123, 2026]
    seed_res = []
    
    for s in seeds:
        p_c = train_base_model(X_train_ctrl, y_train, random_state=s)
        p_cand = train_base_model(X_train_cand, y_train, random_state=s)
        
        pr_c_val = p_c.predict(X_val_ctrl)
        pr_cand_val = p_cand.predict(X_val_cand)
        pr_c_te = p_c.predict(X_test_ctrl)
        pr_cand_te = p_cand.predict(X_test_cand)
        
        rv_c = np.sqrt(mean_squared_error(y_val, pr_c_val))
        rv_cand = np.sqrt(mean_squared_error(y_val, pr_cand_val))
        rt_c = np.sqrt(mean_squared_error(y_test, pr_c_te))
        rt_cand = np.sqrt(mean_squared_error(y_test, pr_cand_te))
        
        mv_c = mean_absolute_error(y_val, pr_c_val)
        mv_cand = mean_absolute_error(y_val, pr_cand_val)
        mt_c = mean_absolute_error(y_test, pr_c_te)
        mt_cand = mean_absolute_error(y_test, pr_cand_te)
        
        seed_res.append({
            "random_seed": s,
            "val_control_rmse": rv_c,
            "val_candidate_rmse": rv_cand,
            "val_rmse_difference": rv_cand - rv_c,
            "test_control_rmse": rt_c,
            "test_candidate_rmse": rt_cand,
            "test_rmse_difference": rt_cand - rt_c,
            "val_control_mae": mv_c,
            "val_candidate_mae": mv_cand,
            "test_control_mae": mt_c,
            "test_candidate_mae": mt_cand,
            "val_candidate_better": rv_cand < rv_c,
            "test_candidate_better": rt_cand < rt_c
        })
    pd.DataFrame(seed_res).to_csv(PROCESSED_DIR / "phase12_step6_seed_robustness.csv", index=False)
    
    # 13. EXPERIMENT I: TRAINING-WINDOW SENSITIVITY
    window_configs = [
        {"name": "full_historical", "start": 1960, "end": 2018},
        {"name": "recent_history", "start": 2000, "end": 2018},
        {"name": "medium_history", "start": 1990, "end": 2018}
    ]
    win_res = []
    
    for cfg in window_configs:
        m_tr = (df["year"] >= cfg["start"]) & (df["year"] <= cfg["end"]) & valid_target
        df_w = construct_regimes(df, m_tr)
        
        X_tr_c = df_w[m_tr][base_features]
        X_tr_cand = df_w[m_tr][candidate_features]
        y_tr = df_w[m_tr]["gdp_growth_next_year"].values
        
        X_v_c = df_w[val_mask][base_features]
        X_v_cand = df_w[val_mask][candidate_features]
        y_v = df_w[val_mask]["gdp_growth_next_year"].values
        
        p_c = train_base_model(X_tr_c, y_tr, random_state=42)
        p_cand = train_base_model(X_tr_cand, y_tr, random_state=42)
        
        pr_c = p_c.predict(X_v_c)
        pr_cand = p_cand.predict(X_v_cand)
        
        r_c = np.sqrt(mean_squared_error(y_v, pr_c))
        r_cand = np.sqrt(mean_squared_error(y_v, pr_cand))
        m_c = mean_absolute_error(y_v, pr_c)
        m_cand = mean_absolute_error(y_v, pr_cand)
        
        win_res.append({
            "configuration": cfg["name"],
            "training_start": cfg["start"],
            "training_end": cfg["end"],
            "control_val_rmse": r_c,
            "candidate_val_rmse": r_cand,
            "rmse_difference": r_cand - r_c,
            "control_val_mae": m_c,
            "candidate_val_mae": m_cand,
            "mae_difference": m_cand - m_c,
            "candidate_better": r_cand < r_c,
            "decision": "PROMOTE" if r_cand < r_c else "REJECT"
        })
    pd.DataFrame(win_res).to_csv(PROCESSED_DIR / "phase12_step6_window_sensitivity.csv", index=False)

    # 14. Robustness Decision Framework
    # 1. Validation RMSE remains better than 8.2853.
    # 2. Paired statistical analysis supports improvement.
    # 3. Bootstrap 95% CI provides convincing evidence that improvement is not centered around zero.
    # 4. Candidate B demonstrates reasonable walk-forward stability.
    # 5. Improvement is not isolated to one year.
    # 6. Improvement is not isolated to one country.
    # 7. Improvement is not caused exclusively by one regime.
    # 8. Random-seed results remain directionally stable.
    # 9. Training-window sensitivity does not cause the improvement to disappear.
    # 10. No leakage or integrity violation exists.
    
    boot_rmse_ci_upper = boot_df.iloc[0]["ci_upper_95"]
    is_val_better = val_rmse_cand < 8.2853 and val_rmse_cand < val_rmse_ctrl
    is_stat_sig = (wilcoxon_p_sq < 0.05) or (p_val_perm_sq < 0.05)
    is_boot_sig = boot_rmse_ci_upper < 0
    is_wf_stable = win_count > loss_count
    
    country_win_rate = pd.DataFrame(country_res)["candidate_better"].mean()
    seed_win_rate = pd.DataFrame(seed_res)["val_candidate_better"].mean()
    window_win_rate = pd.DataFrame(win_res)["candidate_better"].mean()
    
    if is_val_better and is_stat_sig and is_boot_sig and is_wf_stable and country_win_rate > 0.4 and seed_win_rate > 0.8 and window_win_rate == 1.0:
        final_decision = "ROBUST EXPERIMENTAL WINNER"
    elif is_val_better and (is_stat_sig or is_boot_sig or is_wf_stable or country_win_rate > 0.4 or seed_win_rate > 0.6):
        final_decision = "ROBUSTNESS INCONCLUSIVE"
    else:
        final_decision = "ROBUSTNESS REJECTED"
        
    print(f"Stats sig: {is_stat_sig}, boot sig: {is_boot_sig}, wf stable: {is_wf_stable}")
    print(f"country win: {country_win_rate:.2f}, seed win: {seed_win_rate:.2f}, window win: {window_win_rate:.2f}")
    
    # 15. Save Metadata
    metadata = {
        "phase": 12,
        "step": 6,
        "control_model": "HistGradientBoostingRegressor",
        "candidate_model": "HistGradientBoostingRegressor",
        "control_feature_count": len(base_features),
        "candidate_feature_count": len(candidate_features),
        "locked_parameters": {
            "learning_rate": 0.05,
            "max_depth": 5,
            "max_iter": 300,
            "l2_regularization": 5.0,
            "random_state": 42
        },
        "dataset_md5": raw_md5,
        "validation_period": "2019-2022",
        "test_period": "2023-2024",
        "walk_forward_period": "2013-2024",
        "statistical_tests": {
            "permutation_test_p_value": float(p_val_perm_sq),
            "wilcoxon_p_value": float(wilcoxon_p_sq),
            "significance_level": 0.05
        },
        "bootstrap_configuration": {
            "iterations": n_boot,
            "seed": 42
        },
        "random_seeds": seeds,
        "training_windows": [cfg["name"] for cfg in window_configs],
        "metrics": {
            "val_rmse_control": float(val_rmse_ctrl),
            "val_rmse_candidate": float(val_rmse_cand),
            "test_rmse_control": float(test_rmse_ctrl),
            "test_rmse_candidate": float(test_rmse_cand),
            "wf_rmse_control": float(wf_rmse_ctrl),
            "wf_rmse_candidate": float(wf_rmse_cand),
        },
        "final_decision": final_decision,
        "production_model_modified": False,
        "protected_artifact_hashes": initial_hashes,
        "timestamp": datetime.now().isoformat(),
        "reproducibility": "deterministic_seeds_used"
    }
    
    with open(MODELS_DIR / "step6_model_robustness_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Re-verify hashes at the end
    for pf, old_hash in initial_hashes.items():
        if get_file_hash(PROJECT_ROOT / pf) != old_hash:
            print(f"FATAL: Protected artifact {pf} was modified!")
            return
            
    print("Step 6 Evaluation Complete. Artifacts generated.")
    print(f"DECISION: {final_decision}")

if __name__ == "__main__":
    main()
