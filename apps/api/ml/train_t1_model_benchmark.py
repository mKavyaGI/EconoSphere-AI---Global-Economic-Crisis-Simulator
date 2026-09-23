"""
EconoSphere AI -- Phase 11, Step 11: Benchmark and Uncertainty
======================================================================
Script   : apps/api/ml/train_t1_model_benchmark.py
Input    : data/processed/master_panel_t1_missingness.csv
Outputs  : models/phase11/t1_step11_model_metadata.json
           data/processed/t1_step11_predictions.csv
           data/processed/t1_step11_2026_forecasts.csv
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "phase11"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "t1_step11_2026_forecasts.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "t1_step11_predictions.csv"

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
        "R2": r2_score(y_true, y_pred),
        "Bias": np.mean(y_pred - y_true)
    }

def main():
    print("STEP 11.4: Model Benchmarking")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    valid_target = df["next_year_target_available"] == 1
    
    train = df[(df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1]) & valid_target].copy()
    val = df[(df["year"] >= VAL_YEARS[0]) & (df["year"] <= VAL_YEARS[1]) & valid_target].copy()
    test = df[(df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1]) & valid_target].copy()
    inf_2025 = df[df["year"] == 2025].copy()
    
    # 2. Features (Locked Step 8 Features)
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
    locked_features = base_features + rolling
    
    X_train = train[locked_features].copy()
    y_train = train["gdp_growth_next_year"].values
    X_val = val[locked_features].copy()
    y_val = val["gdp_growth_next_year"].values
    X_test = test[locked_features].copy()
    y_test = test["gdp_growth_next_year"].values
    X_inf = inf_2025[locked_features].copy()
    
    # 3. Models Config
    locked_params = {
        'l2_regularization': 5.0, 
        'learning_rate': 0.05, 
        'max_depth': 5, 
        'max_iter': 300,
        'random_state': 42
    }
    
    models_to_evaluate = {
        "A. HistGradientBoosting (Locked Control)": HistGradientBoostingRegressor(**locked_params),
        "B. RandomForestRegressor (Default)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "C. ExtraTreesRegressor (Default)": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "D. GradientBoostingRegressor (Default)": GradientBoostingRegressor(random_state=42)
    }
    
    print("\n--- Phase 1: Default Benchmarking ---")
    results = {}
    
    for name, model in models_to_evaluate.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        
        val_preds = pipe.predict(X_val)
        test_preds = pipe.predict(X_test)
        
        val_m = get_metrics(y_val, val_preds)
        test_m = get_metrics(y_test, test_preds)
        
        results[name] = {"val": val_m, "test": test_m, "pipe": pipe}
        print(f"{name}: Val RMSE = {val_m['RMSE']:.4f} | Test RMSE = {test_m['RMSE']:.4f}")

    print("\n--- Phase 2: Bounded Tuning ---")
    # RF bounded tuning
    rf_grid = {
        'model__n_estimators': [200],
        'model__max_depth': [5, 10, None],
        'model__min_samples_leaf': [1, 5]
    }
    rf_base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    best_rf_rmse = float('inf')
    best_rf_pipe = None
    for params in ParameterGrid(rf_grid):
        from sklearn.base import clone
        pipe = clone(rf_base)
        pipe.set_params(**params)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        if rmse < best_rf_rmse:
            best_rf_rmse = rmse
            best_rf_pipe = pipe
            
    val_m = get_metrics(y_val, best_rf_pipe.predict(X_val))
    test_m = get_metrics(y_test, best_rf_pipe.predict(X_test))
    results["E. RandomForest (Tuned)"] = {"val": val_m, "test": test_m, "pipe": best_rf_pipe}
    print(f"E. RandomForest (Tuned): Val RMSE = {val_m['RMSE']:.4f} | Test RMSE = {test_m['RMSE']:.4f}")

    # 4. Model Selection Rule
    print("\n--- Model Selection ---")
    control_val_rmse = results["A. HistGradientBoosting (Locked Control)"]["val"]["RMSE"]
    control_test_rmse = results["A. HistGradientBoosting (Locked Control)"]["test"]["RMSE"]
    
    best_val_rmse = float('inf')
    best_model_name = ""
    for name, m in results.items():
        if m["val"]["RMSE"] < best_val_rmse:
            best_val_rmse = m["val"]["RMSE"]
            best_model_name = name
            
    selected_name = "A. HistGradientBoosting (Locked Control)"
    
    if best_val_rmse < control_val_rmse:
        candidate_test_rmse = results[best_model_name]["test"]["RMSE"]
        if candidate_test_rmse < control_test_rmse:
            print(f"Candidate {best_model_name} beat Control on Val ({best_val_rmse:.4f} < {control_val_rmse:.4f}) AND Test ({candidate_test_rmse:.4f} < {control_test_rmse:.4f})")
            selected_name = best_model_name
        else:
            print(f"Candidate {best_model_name} beat Control on Val but FAILED on Test ({candidate_test_rmse:.4f} > {control_test_rmse:.4f})")
            print("Action: KEEP STEP 10 LOCKED CONTROL")
    else:
        print(f"No candidate beat Control on Val. Best was {best_val_rmse:.4f} >= {control_val_rmse:.4f}")
        print("Action: KEEP STEP 10 LOCKED CONTROL")
        
    print(f"\n=> Selected Model: {selected_name}")
    final_pipe = results[selected_name]["pipe"]
    
    # 5. Temporal Robustness Evaluation
    print("\n--- Temporal Robustness (Test Set) ---")
    test_predictions = final_pipe.predict(X_test)
    for year in [2023, 2024]:
        mask = test["year"] == year
        if mask.sum() > 0:
            yr_actual = y_test[mask]
            yr_pred = test_predictions[mask]
            yr_rmse = np.sqrt(mean_squared_error(yr_actual, yr_pred))
            yr_mae = mean_absolute_error(yr_actual, yr_pred)
            yr_r2 = r2_score(yr_actual, yr_pred)
            print(f"{year} (N={mask.sum()}): RMSE={yr_rmse:.4f}, MAE={yr_mae:.4f}, R2={yr_r2:.4f}")
            
    # 6. Prediction Uncertainty (90% Residual-Calibrated)
    print("\n--- STEP 11.9: Prediction Uncertainty ---")
    # Calibrate on validation set ONLY
    val_predictions = final_pipe.predict(X_val)
    val_abs_residuals = np.abs(y_val - val_predictions)
    q90 = np.quantile(val_abs_residuals, 0.90)
    print(f"Validation 90th percentile absolute residual: {q90:.4f}")
    
    # 7. 5-Country Verification
    print("\n--- 5-Country Verification ---")
    five_countries = ["IND", "CHN", "USA", "JPN", "GBR"]
    for code in five_countries:
        for year in [2024]: 
            mask = (test["country_code"] == code) & (test["year"] == year)
            if mask.sum() > 0:
                actual = test.loc[mask, "gdp_growth_next_year"].values[0]
                pred = final_pipe.predict(X_test.loc[mask])[0]
                error = pred - actual
                print(f"{code} (Target {year}): Actual={actual:.2f}%, Predicted={pred:.2f}%, Error={error:.2f}%")
                
    # 8. Inference / Forecast Generation
    print("\n--- 2026 Forecast Generation with Uncertainty ---")
    inf_predictions = final_pipe.predict(X_inf)
    
    inf_2025["predicted_gdp_growth"] = inf_predictions
    inf_2025["lower_bound_90"] = inf_predictions - q90
    inf_2025["upper_bound_90"] = inf_predictions + q90
    inf_2025["target_year"] = 2026
    inf_2025["model_version"] = "t1_step11_locked_control" if selected_name == "A. HistGradientBoosting (Locked Control)" else "t1_step11_benchmark"
    
    for code in five_countries:
        mask = inf_2025["country_code"] == code
        if mask.sum() > 0:
            pred = inf_2025.loc[mask, "predicted_gdp_growth"].values[0]
            lb = inf_2025.loc[mask, "lower_bound_90"].values[0]
            ub = inf_2025.loc[mask, "upper_bound_90"].values[0]
            width = ub - lb
            print(f"{code} 2026 Forecast: {pred:.2f}% | 90% Interval: [{lb:.2f}%, {ub:.2f}%] (Width: {width:.2f}%)")
            
    # Output files
    inf_out = inf_2025[["country_code", "country_name", "year", "target_year", "predicted_gdp_growth", "lower_bound_90", "upper_bound_90", "model_version"]].rename(columns={"year": "feature_year"})
    inf_out.to_csv(FORECAST_PATH, index=False)
    
    test_out = test.copy()
    test_out["predicted_gdp_growth_next_year"] = test_predictions
    test_out["lower_bound_90"] = test_predictions - q90
    test_out["upper_bound_90"] = test_predictions + q90
    test_out.to_csv(PREDICTIONS_PATH, index=False)
    
    # Save metadata
    # Ensure all Numpy types are converted to Python native types using json.dumps default
    def default_serializer(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError
        
    meta = {
        "selected_model": selected_name,
        "calibration_q90": float(q90),
        "validation_metrics": results[selected_name]["val"],
        "test_metrics": results[selected_name]["test"]
    }
    with open(MODELS_DIR / "t1_step11_model_metadata.json", "w") as f:
        json.dump(meta, f, default=default_serializer, indent=2)

if __name__ == "__main__":
    main()
