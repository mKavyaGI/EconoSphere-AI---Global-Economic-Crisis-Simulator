"""
EconoSphere AI -- Phase 11, Step 8: Advanced ML Training
======================================================================
Script   : apps/api/ml/train_t1_advanced.py
Input    : data/processed/master_panel_t1_advanced.csv (READ-ONLY)
Outputs  : models/phase11/best_advanced_model.joblib
"""
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings("ignore")

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_advanced.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "phase11"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "t1_advanced_2026_forecasts.csv"

TRAIN_YEARS = (2000, 2018)
VAL_YEARS = (2019, 2022)
TEST_YEARS = (2023, 2025)

WB_AGGREGATES = {'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'}

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Bias": bias}

def main():
    print("STEP 8: Advanced ML Training (T+1)")
    
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    
    # Feature sets
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
    momentum = ["gdp_growth_momentum_1_2", "inflation_momentum_1_2", "unemployment_momentum_1_2", "exports_momentum", "imports_momentum"]
    rolling = ["gdp_growth_rolling_std_5", "inflation_rolling_std_3"]
    country_global = ["country_historical_avg_gdp_growth", "country_historical_avg_inflation", "global_gdp_growth_lag1", "global_inflation_lag1"]
    
    experiments = {
        "Exp A (Baseline)": base_features,
        "Exp B (+Momentum)": base_features + momentum,
        "Exp C (+Rolling)": base_features + rolling,
        "Exp D (+Country/Global)": base_features + country_global,
        "Exp E (All Advanced)": base_features + momentum + rolling + country_global
    }
    
    # Splits
    valid_target = df["next_year_target_available"] == 1
    train = df[(df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1]) & valid_target]
    val = df[(df["year"] >= VAL_YEARS[0]) & (df["year"] <= VAL_YEARS[1]) & valid_target]
    test = df[(df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1]) & valid_target]
    inf_2025 = df[df["year"] == 2025]
    
    print("\n--- Feature Ablation (HistGradientBoosting) ---")
    best_exp_rmse = float('inf')
    best_exp_features = []
    best_exp_name = ""
    
    y_train = train["gdp_growth_next_year"].values
    y_val = val["gdp_growth_next_year"].values
    
    exp_results = {}
    for name, feats in experiments.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(random_state=42))
        ])
        pipe.fit(train[feats], y_train)
        preds = pipe.predict(val[feats])
        m = metrics(y_val, preds)
        exp_results[name] = m
        print(f"{name}: Val RMSE = {m['RMSE']:.4f}")
        
        if m["RMSE"] < best_exp_rmse:
            best_exp_rmse = m["RMSE"]
            best_exp_features = feats
            best_exp_name = name
            
    print(f"Best Feature Set: {best_exp_name}")
    
    print("\n--- Model Comparison on Best Feature Set ---")
    models = {
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    best_model_rmse = float('inf')
    best_model_name = ""
    for mname, model in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        pipe.fit(train[best_exp_features], y_train)
        preds = pipe.predict(val[best_exp_features])
        m = metrics(y_val, preds)
        print(f"{mname}: Val RMSE = {m['RMSE']:.4f}")
        if m["RMSE"] < best_model_rmse:
            best_model_rmse = m["RMSE"]
            best_model_name = mname
            
    print(f"Best Model Type: {best_model_name}")
    
    print("\n--- Hyperparameter Optimization (Time-Aware Val) ---")
    param_grid = {}
    if best_model_name == "HistGradientBoosting":
        param_grid = {
            'model__max_iter': [100, 300, 500],
            'model__max_depth': [3, 5, 8],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__l2_regularization': [0.0, 1.0, 5.0]
        }
    elif best_model_name in ["RandomForest", "ExtraTrees"]:
        param_grid = {
            'model__n_estimators': [100, 300],
            'model__max_depth': [5, 10, None],
            'model__min_samples_leaf': [1, 5]
        }
        
    best_pipe = None
    best_tune_rmse = float('inf')
    
    from sklearn.base import clone
    base_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", clone(models[best_model_name]))
    ])
    
    for params in ParameterGrid(param_grid):
        pipe = clone(base_pipe)
        pipe.set_params(**params)
        pipe.fit(train[best_exp_features], y_train)
        preds = pipe.predict(val[best_exp_features])
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        if rmse < best_tune_rmse:
            best_tune_rmse = rmse
            best_pipe = pipe
            
    print(f"Tuned Val RMSE = {best_tune_rmse:.4f}")
    
    print("\n--- Final Test Evaluation ---")
    y_test = test["gdp_growth_next_year"].values
    test_preds = best_pipe.predict(test[best_exp_features])
    final_test_m = metrics(y_test, test_preds)
    
    print(f"Final Test RMSE: {final_test_m['RMSE']:.4f}")
    
    print("\n--- 2026 Forecast Generation ---")
    preds_2026 = best_pipe.predict(inf_2025[best_exp_features])
    inf_2025["predicted_gdp_growth"] = preds_2026
    inf_2025["forecast_year"] = 2026
    inf_out = inf_2025[["country_code", "country_name", "year", "forecast_year", "predicted_gdp_growth"]].copy()
    inf_out.to_csv(FORECAST_PATH, index=False)
    
    print("\n--- Feature Importance ---")
    r = permutation_importance(best_pipe, val[best_exp_features], y_val, n_repeats=5, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({"Feature": best_exp_features, "Importance": r.importances_mean}).sort_values("Importance", ascending=False)
    
    # Save the full results and metadata for report generator
    results_out = {
        "best_feature_set": best_exp_name,
        "best_model_name": best_model_name,
        "val_rmse": best_tune_rmse,
        "test_rmse": final_test_m["RMSE"],
        "importance": importance_df.head(10).to_dict(orient="records"),
        "test_preds": test[["country_code", "year", "gdp_growth_next_year"]].assign(pred=test_preds).to_dict(orient="records")
    }
    
    with open(MODELS_DIR / "t1_advanced_results.json", "w") as f:
        json.dump(results_out, f)
        
    joblib.dump(best_pipe, MODELS_DIR / "best_advanced_model.joblib")

if __name__ == "__main__":
    main()
