"""
EconoSphere AI -- Phase 11, Step 11: Reproduce Locked Control
======================================================================
Script   : apps/api/ml/reproduce_t1_step10_best.py
Input    : data/processed/master_panel_t1_missingness.csv
Outputs  : Terminal standard output
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"

# Chronological split boundaries
TRAIN_YEARS = (2000, 2018)
VAL_YEARS = (2019, 2022)
TEST_YEARS = (2023, 2024)

# Excluded generic aggregates
WB_AGGREGATES = {'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'}

def get_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Bias": np.mean(y_pred - y_true)
    }

def main():
    print("STEP 11.2: Exact Reproducibility Lock")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_PATH)
    
    # 2. Filtering
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    valid_target = df["next_year_target_available"] == 1
    
    train = df[(df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1]) & valid_target].copy()
    val = df[(df["year"] >= VAL_YEARS[0]) & (df["year"] <= VAL_YEARS[1]) & valid_target].copy()
    test = df[(df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1]) & valid_target].copy()
    
    # 3. Features
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
    
    # 4. Pipeline and Parameters
    # Step 10 hyperparameters exactly
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
    
    # 5. Training
    pipe.fit(X_train, y_train)
    
    # 6. Evaluation
    val_preds = pipe.predict(X_val)
    test_preds = pipe.predict(X_test)
    
    val_metrics = get_metrics(y_val, val_preds)
    test_metrics = get_metrics(y_test, test_preds)
    
    print("\n--- Model Configuration ---")
    print(f"Data Path: {INPUT_PATH.name}")
    print(f"Train Years: {TRAIN_YEARS}")
    print(f"Val Years: {VAL_YEARS}")
    print(f"Test Years: {TEST_YEARS}")
    print(f"Model Class: HistGradientBoostingRegressor")
    print(f"Imputation: SimpleImputer(strategy='median')")
    print(f"Hyperparameters: {locked_params}")
    
    print("\n--- Reproducibility Metrics ---")
    print(f"Validation RMSE : {val_metrics['RMSE']:.4f}")
    print(f"Validation MAE  : {val_metrics['MAE']:.4f}")
    print(f"Test RMSE       : {test_metrics['RMSE']:.4f}")
    print(f"Test MAE        : {test_metrics['MAE']:.4f}")
    
    # Check explicitly against the reported 3.9113
    diff = abs(test_metrics["RMSE"] - 3.9113)
    if diff < 0.0001:
        print("\n[SUCCESS] Exact reproducibility achieved. Difference < 0.0001")
    else:
        print(f"\n[WARNING] Discrepancy detected. Difference = {diff:.6f}")
        
    # Test specific country forecast
    mask = (test["country_code"] == "IND") & (test["year"] == 2024)
    if mask.sum() > 0:
        pred = pipe.predict(X_test.loc[mask])[0]
        actual = test.loc[mask, "gdp_growth_next_year"].values[0]
        print(f"IND 2024 Forecast Check: Actual={actual:.2f}%, Predicted={pred:.2f}%")

if __name__ == "__main__":
    main()
