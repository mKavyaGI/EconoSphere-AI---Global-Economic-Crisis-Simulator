import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "phase12"
WALK_FWD_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step4_walk_forward_metrics.csv"
WALK_FWD_PREDS_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step4_walk_forward_predictions.csv"

WB_AGGREGATES = {
    'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'
}

def construct_regimes(df, train_mask):
    """Construct regimes using thresholds strictly defined on the training set."""
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

def train_base_model(X_train, y_train):
    locked_params = {
        'l2_regularization': 5.0, 
        'learning_rate': 0.05, 
        'max_depth': 5, 
        'max_iter': 300,
        'random_state': 42
    }
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(**locked_params))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def main():
    print("STEP 4: Walk-Forward Evaluation")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    valid_target = df["next_year_target_available"] == 1
    
    meta_path = MODELS_DIR / "step4_regime_aware_metadata.json"
    if not meta_path.exists():
        print("Metadata not found, run training first.")
        return
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    best_exp_name = "B. Regime Features" # Forcing evaluation since it beat control
    
    print(f"Evaluating {best_exp_name}")
    
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
    
    windows = list(range(2013, 2025))
    all_preds_candidate = []
    all_preds_control = []
    window_metrics = []
    
    for target_year in windows:
        feature_year = target_year - 1
        
        train_mask = (df["year"] < feature_year) & valid_target
        test_mask = (df["year"] == feature_year) & valid_target
        
        if test_mask.sum() == 0:
            continue
            
        # Re-construct regimes for this expanding window using strictly the historical training data
        df_window = construct_regimes(df, train_mask)
        
        train = df_window[train_mask].copy()
        test = df_window[test_mask].copy()
        
        y_train = train["gdp_growth_next_year"].values
        y_test = test["gdp_growth_next_year"].values
        
        # 1. Control model
        pipe_ctrl = train_base_model(train[base_features], y_train)
        preds_ctrl = pipe_ctrl.predict(test[base_features])
        rmse_ctrl = np.sqrt(mean_squared_error(y_test, preds_ctrl))
        
        # 2. Candidate model (B)
        pipe_cand = train_base_model(train[candidate_features], y_train)
        preds_cand = pipe_cand.predict(test[candidate_features])
        rmse_cand = np.sqrt(mean_squared_error(y_test, preds_cand))
        
        window_metrics.append({
            "Target_Year": target_year,
            "Control_RMSE": rmse_ctrl,
            "Candidate_RMSE": rmse_cand,
            "Control_MAE": mean_absolute_error(y_test, preds_ctrl),
            "Candidate_MAE": mean_absolute_error(y_test, preds_cand)
        })
        
        test_out = test.copy()
        test_out["predicted_gdp_growth"] = preds_cand
        test_out["control_predicted_gdp_growth"] = preds_ctrl
        test_out["target_year"] = target_year
        all_preds_candidate.append(test_out)
        
    df_metrics = pd.DataFrame(window_metrics)
    df_metrics.to_csv(WALK_FWD_METRICS_PATH, index=False)
    
    df_preds = pd.concat(all_preds_candidate, ignore_index=True)
    df_preds.to_csv(WALK_FWD_PREDS_PATH, index=False)
    
    overall_rmse_cand = np.sqrt(mean_squared_error(df_preds["gdp_growth_next_year"], df_preds["predicted_gdp_growth"]))
    overall_rmse_ctrl = np.sqrt(mean_squared_error(df_preds["gdp_growth_next_year"], df_preds["control_predicted_gdp_growth"]))
    
    # Calculate Recession RMSE
    rec_mask = df_preds["gdp_growth_next_year"] < 0
    rec_rmse_cand = np.sqrt(mean_squared_error(df_preds.loc[rec_mask, "gdp_growth_next_year"], df_preds.loc[rec_mask, "predicted_gdp_growth"]))
    rec_rmse_ctrl = np.sqrt(mean_squared_error(df_preds.loc[rec_mask, "gdp_growth_next_year"], df_preds.loc[rec_mask, "control_predicted_gdp_growth"]))
    
    print(f"Overall Walk-Forward RMSE: Control = {overall_rmse_ctrl:.4f} | Candidate = {overall_rmse_cand:.4f}")
    print(f"Recession Walk-Forward RMSE: Control = {rec_rmse_ctrl:.4f} | Candidate = {rec_rmse_cand:.4f}")
    
    meta["candidate_walk_forward_RMSE"] = overall_rmse_cand
    meta["control_walk_forward_RMSE"] = overall_rmse_ctrl
    meta["candidate_recession_walk_forward_RMSE"] = rec_rmse_cand
    meta["control_recession_walk_forward_RMSE"] = rec_rmse_ctrl
    
    # Final promotion decision: Does it also improve WF RMSE?
    if overall_rmse_cand < overall_rmse_ctrl:
        meta["final_decision"] = "PROMOTE CANDIDATE"
        print("=> CANDIDATE BEAT CONTROL IN VALIDATION AND WALK-FORWARD. PROMOTE!")
    else:
        meta["final_decision"] = "KEEP PHASE 11 CONTROL"
        print("=> CANDIDATE FAILED WALK-FORWARD CONFIRMATION. KEEP CONTROL.")
        
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
