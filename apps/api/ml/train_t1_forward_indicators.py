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
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step3_2026_forecasts.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step3_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step3_experiment_metrics.csv"
RECESSION_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step3_recession_analysis.csv"

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

def engineer_features(df):
    """Engineer forward-looking change and stress features."""
    df = df.copy()
    df.sort_values(by=["country_code", "year"], inplace=True)
    
    # Calculate lags that aren't already present
    df['interest_rate_lag1'] = df.groupby('country_code')['interest_rate_pct'].shift(1)
    df['exports_pct_gdp_lag1'] = df.groupby('country_code')['exports_pct_gdp'].shift(1)
    
    # Experiment C: MACRO CHANGES (year over year change)
    df['inflation_change'] = df['inflation_cpi_pct'] - df['inflation_lag1']
    df['unemployment_change'] = df['unemployment_pct'] - df['unemployment_lag1']
    df['interest_rate_change'] = df['interest_rate_pct'] - df['interest_rate_lag1']
    df['exports_pct_gdp_change'] = df['exports_pct_gdp'] - df['exports_pct_gdp_lag1']
    
    # Experiment D: MACRO STRESS SIGNALS
    # e.g., inflation acceleration (change in change)
    df['inflation_change_lag1'] = df.groupby('country_code')['inflation_change'].shift(1)
    df['inflation_acceleration'] = df['inflation_change'] - df['inflation_change_lag1']
    
    # unemployment deterioration: binary indicator if unemployment rose by > 0.5%
    df['unemployment_deterioration'] = (df['unemployment_change'] > 0.5).astype(float)
    
    # trade contraction: exports drop by > 2% of GDP
    df['trade_contraction'] = (df['exports_pct_gdp_change'] < -2.0).astype(float)
    
    # interest rate tightening: rates up by > 0.5%
    df['interest_rate_tightening'] = (df['interest_rate_change'] > 0.5).astype(float)
    
    return df

def main():
    print("STEP 3: Forward-Looking Economic Indicators")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    
    # Engineer new features
    df = engineer_features(df)
    
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
    
    # Macro Changes
    macro_changes = ['inflation_change', 'unemployment_change', 'interest_rate_change', 'exports_pct_gdp_change']
    
    # Macro Stress
    macro_stress = ['inflation_acceleration', 'unemployment_deterioration', 'trade_contraction', 'interest_rate_tightening']
    
    # Momentum (from Phase 12 Step 2 - if needed, but we will just use the available ones or skip. Actually, we can use gdp_growth_lag1/2/3 which are already in locked_features. To represent "Momentum" specifically, maybe gdp_growth_change)
    df['gdp_growth_change'] = df['gdp_growth_pct'] - df['gdp_growth_lag1']
    momentum = ['gdp_growth_change']
    
    feature_sets = {
        "A. Control": locked_features,
        "C. Macro Changes": locked_features + macro_changes,
        "D. Macro Stress": locked_features + macro_stress,
        "E. Forward + Momentum": locked_features + macro_changes + macro_stress + momentum,
        "F. Minimal Signal": locked_features + ['inflation_change', 'unemployment_change', 'trade_contraction']
    }
    
    # 3. Models Config
    locked_params = {
        'l2_regularization': 5.0, 
        'learning_rate': 0.05, 
        'max_depth': 5, 
        'max_iter': 300,
        'random_state': 42
    }
    
    print("\n--- Running Experiments A-F ---")
    results = []
    recession_results = []
    
    best_val_rmse = float('inf')
    best_exp_name = "A. Control"
    best_pipe = None
    
    y_train = train["gdp_growth_next_year"].values
    y_val = val["gdp_growth_next_year"].values
    y_test = test["gdp_growth_next_year"].values
    
    val_recession_mask = y_val < 0
    test_recession_mask = y_test < 0
    
    for name, features in feature_sets.items():
        # Update missing engineering features in train/val/test
        for ds in [train, val, test, inf_2025]:
            for f in features:
                if f not in ds.columns:
                    ds[f] = df[df.index.isin(ds.index)][f]
        
        X_train = train[features].copy()
        X_val = val[features].copy()
        X_test = test[features].copy()
        
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(**locked_params))
        ])
        pipe.fit(X_train, y_train)
        
        val_preds = pipe.predict(X_val)
        test_preds = pipe.predict(X_test)
        
        val_m = get_metrics(y_val, val_preds)
        test_m = get_metrics(y_test, test_preds)
        
        # Recession metrics
        val_rec_preds = val_preds[val_recession_mask]
        val_rec_actual = y_val[val_recession_mask]
        val_rec_m = get_metrics(val_rec_actual, val_rec_preds)
        missed_recessions = np.sum((val_rec_actual < 0) & (val_rec_preds >= 0))
        
        print(f"{name}: Val RMSE = {val_m['RMSE']:.4f} | Test RMSE = {test_m['RMSE']:.4f} | Val Rec RMSE = {val_rec_m['RMSE']:.4f}")
        
        results.append({
            "Experiment": name,
            "Features": len(features),
            "Val_RMSE": val_m["RMSE"],
            "Test_RMSE": test_m["RMSE"]
        })
        
        recession_results.append({
            "Experiment": name,
            "Recession_RMSE": val_rec_m["RMSE"],
            "Recession_Bias": val_rec_m["Bias"],
            "Missed_Recessions": missed_recessions
        })
        
        if val_m["RMSE"] < best_val_rmse:
            best_val_rmse = val_m["RMSE"]
            best_exp_name = name
            best_pipe = pipe
            
    pd.DataFrame(results).to_csv(METRICS_PATH, index=False)
    pd.DataFrame(recession_results).to_csv(RECESSION_PATH, index=False)
    
    print("\n--- Model Selection Rule ---")
    control_val = results[0]["Val_RMSE"]
    control_test = results[0]["Test_RMSE"]
    
    if best_val_rmse < control_val:
        candidate_test = next(r["Test_RMSE"] for r in results if r["Experiment"] == best_exp_name)
        print(f"Candidate {best_exp_name} beat Control on Val ({best_val_rmse:.4f} < {control_val:.4f})")
    else:
        print(f"No candidate beat Control on Val. Best was {best_val_rmse:.4f} >= {control_val:.4f}")
        
    # Generate predictions using best pipe (even if control)
    test_out = test.copy()
    test_out["predicted_gdp_growth_next_year"] = best_pipe.predict(test[feature_sets[best_exp_name]])
    test_out.to_csv(PREDICTIONS_PATH, index=False)
    
    # Forecasts for 2026
    X_inf = inf_2025[feature_sets[best_exp_name]].copy()
    inf_predictions = best_pipe.predict(X_inf)
    inf_2025["predicted_gdp_growth"] = inf_predictions
    inf_2025["target_year"] = 2026
    inf_2025["model_version"] = f"t1_step3_{best_exp_name}"
    
    inf_out = inf_2025[["country_code", "country_name", "year", "target_year", "predicted_gdp_growth", "model_version"]].rename(columns={"year": "feature_year"})
    inf_out.to_csv(FORECAST_PATH, index=False)
    
    # Save Metadata
    meta = {
        "best_experiment": best_exp_name,
        "features_used": feature_sets[best_exp_name]
    }
    with open(MODELS_DIR / "step3_forward_indicators_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
