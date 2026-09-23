"""
EconoSphere AI -- Phase 11, Step 10: Missingness-Aware T+1 Forecasting
======================================================================
Script   : apps/api/ml/train_t1_missingness_aware.py
Input    : data/processed/master_panel_t1_missingness.csv
Outputs  : models/phase11/t1_missingness_aware_model_metadata.json
           data/processed/t1_missingness_aware_predictions.csv
           data/processed/t1_missingness_aware_2026_forecasts.csv
"""
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
import warnings

warnings.filterwarnings("ignore")

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "phase11"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "t1_missingness_aware_2026_forecasts.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "t1_missingness_aware_predictions.csv"

TRAIN_YEARS = (2000, 2018)
VAL_YEARS = (2019, 2022)
TEST_YEARS = (2023, 2024)

WB_AGGREGATES = {'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'}

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Bias": bias}

def main():
    print("STEP 10: Missingness-Aware T+1 Forecasting")
    
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    
    base_features = [
        "exchange_rate_lcu_usd", "tariff_rate_pct", "remittances_usd", "fdi_net_inflow_usd", 
        "unemployment_pct", "imports_pct_gdp", "tax_revenue_pct_gdp", "exports_pct_gdp", 
        "interest_rate_pct", "reserves_usd", "current_account_pct_gdp", "inflation_cpi_pct", 
        "population_total", "gdp_current_usd", "gdp_growth_lag1", "gdp_growth_lag2", 
        "gdp_growth_lag3", "inflation_lag1", "unemployment_lag1", "exports_lag1", 
        "imports_lag1", "gdp_growth_rolling_mean_3", "gdp_growth_rolling_std_3", 
        "gdp_growth_rolling_mean_5", "inflation_rolling_mean_3", "trade_openness", 
        "trade_balance_ratio", "log_gdp_usd", "log_population"
    ]
    rolling = ["gdp_growth_rolling_std_5", "inflation_rolling_std_3"]
    step8_control_features = base_features + rolling
    
    macro_indicators = [
        "inflation_cpi_pct", "unemployment_pct", "fdi_net_inflow_usd", "exports_pct_gdp", 
        "imports_pct_gdp", "exchange_rate_lcu_usd", "interest_rate_pct", "reserves_usd", 
        "current_account_pct_gdp", "remittances_usd", "tariff_rate_pct", "tax_revenue_pct_gdp"
    ]
    missing_indicators = [f"{col}_missing" for col in macro_indicators]
    agg_missingness = ["total_missing_feature_count", "total_missing_feature_ratio"]
    
    experiments = {
        "Exp A (Step 8 Control)": {"features": step8_control_features, "native_nan": False},
        "Exp B (+ Indicators)": {"features": step8_control_features + missing_indicators, "native_nan": False},
        "Exp C (+ Aggregates)": {"features": step8_control_features + missing_indicators + agg_missingness, "native_nan": False},
        "Exp D (Native NaN)": {"features": step8_control_features + missing_indicators + agg_missingness, "native_nan": True}
    }
    
    valid_target = df["next_year_target_available"] == 1
    train = df[(df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1]) & valid_target].copy()
    val = df[(df["year"] >= VAL_YEARS[0]) & (df["year"] <= VAL_YEARS[1]) & valid_target].copy()
    test = df[(df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1]) & valid_target].copy()
    inf_2025 = df[df["year"] == 2025].copy()
    
    print("\n--- Model Evaluation (HistGradientBoosting) ---")
    y_train = train["gdp_growth_next_year"].values
    y_val = val["gdp_growth_next_year"].values
    y_test = test["gdp_growth_next_year"].values
    
    exp_results = {}
    best_exp_rmse = float('inf')
    best_exp_name = ""
    best_exp_config = None
    
    for name, config in experiments.items():
        feats = config["features"]
        
        # Prepare X_train, X_val, X_test
        X_tr = train[feats].copy()
        X_va = val[feats].copy()
        X_te = test[feats].copy()
        
        # If Exp D (Native NaN), we revert imputed values back to NaN for the original macro features
        if config["native_nan"]:
            for col in macro_indicators:
                if f"{col}_missing" in feats:
                    # Where the missing indicator is 1, set the feature to NaN
                    X_tr.loc[X_tr[f"{col}_missing"] == 1, col] = np.nan
                    X_va.loc[X_va[f"{col}_missing"] == 1, col] = np.nan
                    X_te.loc[X_te[f"{col}_missing"] == 1, col] = np.nan
            
            # For Native NaN, we don't use SimpleImputer
            pipe = Pipeline([
                ("model", HistGradientBoostingRegressor(random_state=42))
            ])
        else:
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(random_state=42))
            ])
            
        pipe.fit(X_tr, y_train)
        
        val_preds = pipe.predict(X_va)
        test_preds = pipe.predict(X_te)
        
        val_m = metrics(y_val, val_preds)
        test_m = metrics(y_test, test_preds)
        
        exp_results[name] = {"val": val_m, "test": test_m}
        print(f"{name}: Val RMSE = {val_m['RMSE']:.4f} | Test RMSE = {test_m['RMSE']:.4f}")
        
        if val_m["RMSE"] < best_exp_rmse:
            best_exp_rmse = val_m["RMSE"]
            best_exp_name = name
            best_exp_config = config
            
    print(f"\nBest Configuration based on Val RMSE: {best_exp_name}")
    
    print("\n--- Hyperparameter Optimization on Best Config ---")
    feats = best_exp_config["features"]
    
    X_tr = train[feats].copy()
    X_va = val[feats].copy()
    X_te = test[feats].copy()
    X_inf = inf_2025[feats].copy()
    
    if best_exp_config["native_nan"]:
        for col in macro_indicators:
            X_tr.loc[X_tr[f"{col}_missing"] == 1, col] = np.nan
            X_va.loc[X_va[f"{col}_missing"] == 1, col] = np.nan
            X_te.loc[X_te[f"{col}_missing"] == 1, col] = np.nan
            X_inf.loc[X_inf[f"{col}_missing"] == 1, col] = np.nan
            
        base_pipe = Pipeline([
            ("model", HistGradientBoostingRegressor(random_state=42))
        ])
    else:
        base_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(random_state=42))
        ])
    
    param_grid = {
        'model__max_iter': [100, 300, 500],
        'model__max_depth': [3, 5, 8],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__l2_regularization': [0.0, 1.0, 5.0]
    }
    
    best_pipe = None
    best_tune_rmse = float('inf')
    
    from sklearn.base import clone
    for params in ParameterGrid(param_grid):
        pipe = clone(base_pipe)
        pipe.set_params(**params)
        pipe.fit(X_tr, y_train)
        preds = pipe.predict(X_va)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        if rmse < best_tune_rmse:
            best_tune_rmse = rmse
            best_pipe = pipe
            
    print(f"Tuned Val RMSE = {best_tune_rmse:.4f}")
    
    print("\n--- Final Test Evaluation ---")
    test_preds = best_pipe.predict(X_te)
    final_test_m = metrics(y_test, test_preds)
    
    print(f"Final Test RMSE: {final_test_m['RMSE']:.4f}")
    
    print("\n--- 5-Country Verification ---")
    five_countries = ["IND", "CHN", "USA", "JPN", "GBR"]
    for code in five_countries:
        for year in [2024, 2025]: 
            feature_year = year - 1
            mask = (test["country_code"] == code) & (test["year"] == feature_year)
            if mask.sum() > 0:
                actual = test.loc[mask, "gdp_growth_next_year"].values[0]
                pred = best_pipe.predict(X_te.loc[mask])[0]
                error = pred - actual
                print(f"{code} (Target {year}): Actual={actual:.2f}%, Predicted={pred:.2f}%, Error={error:.2f}%")
    
    print("\n--- 2026 Forecast Generation ---")
    preds_2026 = best_pipe.predict(X_inf)
    inf_2025["predicted_gdp_growth"] = preds_2026
    inf_2025["target_year"] = 2026
    inf_2025["model_version"] = "t1_missingness_aware"
    
    for code in five_countries:
        mask = inf_2025["country_code"] == code
        if mask.sum() > 0:
            pred = inf_2025.loc[mask, "predicted_gdp_growth"].values[0]
            print(f"{code} 2026 Forecast: {pred:.2f}%")
            
    inf_out = inf_2025[["country_code", "country_name", "year", "target_year", "predicted_gdp_growth", "model_version"]].rename(columns={"year": "feature_year"})
    inf_out.to_csv(FORECAST_PATH, index=False)
    
    test_out = test.copy()
    test_out["predicted_gdp_growth_next_year"] = test_preds
    test_out.to_csv(PREDICTIONS_PATH, index=False)
    
    from sklearn.inspection import permutation_importance
    r = permutation_importance(best_pipe, X_va, y_val, n_repeats=5, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({"Feature": feats, "Importance": r.importances_mean}).sort_values("Importance", ascending=False)
    
    results_out = {
        "best_feature_set": best_exp_name,
        "val_rmse": best_tune_rmse,
        "test_rmse": final_test_m["RMSE"],
        "experiments": exp_results,
        "importance": importance_df.head(15).to_dict(orient="records")
    }
    
    with open(MODELS_DIR / "t1_missingness_aware_model_metadata.json", "w") as f:
        json.dump(results_out, f)

if __name__ == "__main__":
    main()
