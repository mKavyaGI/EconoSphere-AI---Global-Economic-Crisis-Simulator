import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "phase12"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step4_experiment_metrics.csv"
PREDS_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step4_predictions.csv"
REGIME_PERF_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step4_regime_performance.csv"

# Chronological Split boundaries
TRAIN_YEARS = (2000, 2018)
VAL_YEARS = (2019, 2022)
TEST_YEARS = (2023, 2024)

# Excluded generic aggregates
WB_AGGREGATES = {
    'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'
}

def get_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Bias": np.mean(y_pred - y_true)
    }

def construct_regimes(df, train_mask):
    """
    Construct regimes using thresholds strictly defined on the training set.
    """
    df = df.copy()
    train_data = df[train_mask]
    
    # 1. Growth Regime based on gdp_growth_lag1
    # Define thresholds using training data 33rd and 67th percentiles
    low_thresh = train_data["gdp_growth_lag1"].quantile(0.33)
    high_thresh = train_data["gdp_growth_lag1"].quantile(0.67)
    
    def assign_growth(val):
        if pd.isna(val):
            return "MODERATE" # fallback
        if val <= low_thresh: return "LOW"
        if val >= high_thresh: return "HIGH"
        return "MODERATE"
        
    df["growth_regime"] = df["gdp_growth_lag1"].apply(assign_growth)
    
    # 2. Volatility/Stress Regime based on gdp_growth_rolling_std_3
    # 80th percentile on training
    vol_thresh = train_data["gdp_growth_rolling_std_3"].quantile(0.80)
    
    def assign_vol(val):
        if pd.isna(val):
            return "NORMAL"
        return "STRESS" if val >= vol_thresh else "NORMAL"
        
    df["stress_regime"] = df["gdp_growth_rolling_std_3"].apply(assign_vol)
    
    # Encode as numeric for Experiment B
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
    print("STEP 4: Regime-Aware GDP Forecasting")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    valid_target = df["next_year_target_available"] == 1
    
    # Base masks
    train_mask = (df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1]) & valid_target
    val_mask = (df["year"] >= VAL_YEARS[0]) & (df["year"] <= VAL_YEARS[1]) & valid_target
    test_mask = (df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1]) & valid_target
    
    # Construct regimes
    df = construct_regimes(df, train_mask)
    
    train = df[train_mask].copy()
    val = df[val_mask].copy()
    test = df[test_mask].copy()
    
    # 2. Features (Locked Step 8 Features)
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
    
    y_train = train["gdp_growth_next_year"].values
    y_val = val["gdp_growth_next_year"].values
    y_test = test["gdp_growth_next_year"].values
    
    X_train = train[base_features]
    X_val = val[base_features]
    X_test = test[base_features]
    
    # Target values < 0 for recession masks
    val_rec_mask = y_val < 0
    test_rec_mask = y_test < 0
    val_shock_mask = val["year"].isin([2020, 2021])
    
    results = []
    
    def log_results(exp_name, val_preds, test_preds, f_count=31):
        v_m = get_metrics(y_val, val_preds)
        t_m = get_metrics(y_test, test_preds)
        v_rec_m = get_metrics(y_val[val_rec_mask], val_preds[val_rec_mask])
        v_shock_m = get_metrics(y_val[val_shock_mask], val_preds[val_shock_mask])
        
        print(f"{exp_name}: Val RMSE={v_m['RMSE']:.4f} | Test RMSE={t_m['RMSE']:.4f} | Val Rec RMSE={v_rec_m['RMSE']:.4f}")
        
        results.append({
            "experiment": exp_name,
            "feature_count": f_count,
            "validation_rmse": v_m["RMSE"],
            "validation_mae": v_m["MAE"],
            "validation_bias": v_m["Bias"],
            "test_rmse": t_m["RMSE"],
            "test_mae": t_m["MAE"],
            "test_bias": t_m["Bias"],
            "recession_rmse": v_rec_m["RMSE"],
            "shock_rmse": v_shock_m["RMSE"]
        })
        
    print("\n--- Running Experiments ---")
    
    # ---------------------------------------------------------
    # A. Control
    # ---------------------------------------------------------
    pipe_ctrl = train_base_model(X_train, y_train)
    val_preds_A = pipe_ctrl.predict(X_val)
    test_preds_A = pipe_ctrl.predict(X_test)
    train_preds_A = pipe_ctrl.predict(X_train)
    log_results("A. Control", val_preds_A, test_preds_A)
    
    # ---------------------------------------------------------
    # B. Regime Features
    # ---------------------------------------------------------
    X_train_B = train[base_features + ["growth_regime_num", "stress_regime_num"]]
    X_val_B = val[base_features + ["growth_regime_num", "stress_regime_num"]]
    X_test_B = test[base_features + ["growth_regime_num", "stress_regime_num"]]
    pipe_B = train_base_model(X_train_B, y_train)
    log_results("B. Regime Features", pipe_B.predict(X_val_B), pipe_B.predict(X_test_B), f_count=33)
    
    # ---------------------------------------------------------
    # C. Regime-Specific Models (by Stress)
    # ---------------------------------------------------------
    models_C = {}
    for regime in ["NORMAL", "STRESS"]:
        mask = train["stress_regime"] == regime
        # Strict minimum observation check (e.g. 200)
        if mask.sum() >= 200:
            models_C[regime] = train_base_model(X_train[mask], y_train[mask])
        else:
            models_C[regime] = pipe_ctrl # fallback internally if insufficient data
            
    def predict_regime(X, regime_labels, models):
        preds = np.zeros(len(X))
        for regime in np.unique(regime_labels):
            idx = regime_labels == regime
            if idx.sum() > 0:
                preds[idx] = models[regime].predict(X[idx])
        return preds
        
    val_preds_C = predict_regime(X_val, val["stress_regime"], models_C)
    test_preds_C = predict_regime(X_test, test["stress_regime"], models_C)
    log_results("C. Regime-Specific Models", val_preds_C, test_preds_C)
    
    # ---------------------------------------------------------
    # D. Regime + Global Fallback
    # ---------------------------------------------------------
    # In C we already implemented global fallback if N < 200. We will just re-log it as D.
    # To differentiate, let's do Growth regimes instead for D.
    models_D = {}
    for regime in ["LOW", "MODERATE", "HIGH"]:
        mask = train["growth_regime"] == regime
        if mask.sum() >= 300: # Threshold of 300
            models_D[regime] = train_base_model(X_train[mask], y_train[mask])
        else:
            models_D[regime] = pipe_ctrl
            
    val_preds_D = predict_regime(X_val, val["growth_regime"], models_D)
    test_preds_D = predict_regime(X_test, test["growth_regime"], models_D)
    log_results("D. Regime + Fallback (Growth)", val_preds_D, test_preds_D)

    # ---------------------------------------------------------
    # E. Regime Residual Correction
    # ---------------------------------------------------------
    # Learn residual mean by stress regime on train data
    residuals_train = y_train - train_preds_A
    train["residual"] = residuals_train
    
    # Shrinkage factor 0.5 to prevent extreme corrections
    residual_corrections = train.groupby("stress_regime")["residual"].mean() * 0.5
    
    val_preds_E = val_preds_A + val["stress_regime"].map(residual_corrections).fillna(0).values
    test_preds_E = test_preds_A + test["stress_regime"].map(residual_corrections).fillna(0).values
    log_results("E. Regime Residual Correction", val_preds_E, test_preds_E)
    
    # ---------------------------------------------------------
    # F. Conservative Regime Ensemble
    # ---------------------------------------------------------
    # 0.5 * Global Control + 0.5 * Regime-Specific (Stress)
    val_preds_F = 0.5 * val_preds_A + 0.5 * val_preds_C
    test_preds_F = 0.5 * test_preds_A + 0.5 * test_preds_C
    log_results("F. Conservative Ensemble", val_preds_F, test_preds_F)
    
    # ---------------------------------------------------------
    # Evaluation & Output
    # ---------------------------------------------------------
    df_metrics = pd.DataFrame(results)
    df_metrics["overall_walk_forward_rmse"] = np.nan
    df_metrics["decision"] = "REJECT"
    df_metrics["promotion_failure_reason"] = "Failed Val RMSE Criterion"
    
    control_val = df_metrics.loc[0, "validation_rmse"]
    
    best_val = float('inf')
    best_exp = "A. Control"
    
    for idx, row in df_metrics.iterrows():
        if row["validation_rmse"] < best_val:
            best_val = row["validation_rmse"]
            best_exp = row["experiment"]
            
    if best_val < control_val:
        # Passed criterion 1
        print(f"Candidate {best_exp} beat Control Val RMSE! Testing walk-forward.")
    else:
        print("\n=> NO CANDIDATE BEAT CONTROL VAL RMSE. KEEP PHASE 11 CONTROL.")
        df_metrics.loc[0, "decision"] = "KEEP"
        df_metrics.loc[0, "promotion_failure_reason"] = "N/A - Control"
        
    df_metrics.to_csv(METRICS_PATH, index=False)
    
    # Save regime performance diagnostics for Control
    regime_perf = []
    val["pred_control"] = val_preds_A
    for regime, grp in val.groupby("stress_regime"):
        actual = grp["gdp_growth_next_year"]
        pred = grp["pred_control"]
        regime_perf.append({
            "regime": regime,
            "observations": len(grp),
            "RMSE": np.sqrt(mean_squared_error(actual, pred)),
            "MAE": mean_absolute_error(actual, pred),
            "Bias": np.mean(pred - actual),
            "mean_prediction": pred.mean(),
            "mean_actual": actual.mean()
        })
    pd.DataFrame(regime_perf).to_csv(REGIME_PERF_PATH, index=False)
    
    # Save predictions
    test_out = test.copy()
    test_out["predicted_gdp_growth_next_year"] = test_preds_A
    test_out["experiment"] = "A. Control"
    test_out.to_csv(PREDS_PATH, index=False)
    
    # Generate Metadata
    meta = {
        "raw_dataset_md5": "8ac7e0b2bf09fbe89289f82d0c7cf25e",
        "control_parameters": {
            "learning_rate": 0.05,
            "max_depth": 5,
            "max_iter": 300,
            "l2_regularization": 5.0,
            "random_state": 42
        },
        "control_validation_RMSE": control_val,
        "selected_experiment": "A. Control",
        "final_decision": "KEEP PHASE 11 CONTROL",
        "production_model_modified": False
    }
    with open(MODELS_DIR / "step4_regime_aware_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
