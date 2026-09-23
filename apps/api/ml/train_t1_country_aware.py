"""
EconoSphere AI -- Phase 11, Step 9: Country-Aware T+1 Forecasting
======================================================================
Script   : apps/api/ml/train_t1_country_aware.py
Input    : data/processed/master_panel_t1_advanced.csv (READ-ONLY)
Outputs  : models/phase11/t1_country_aware_model_metadata.json
           data/processed/t1_country_aware_predictions.csv
           data/processed/t1_country_aware_2026_forecasts.csv
"""
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
import warnings

warnings.filterwarnings("ignore")

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_advanced.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "phase11"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "t1_country_aware_2026_forecasts.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "t1_country_aware_predictions.csv"

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
    print("STEP 9: Country-Aware T+1 Forecasting")
    
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    
    # ---------------------------------------------------------
    # Generate Structural Country Features (Train data ONLY)
    # ---------------------------------------------------------
    # We must only use 2000-2018 data to calculate these
    train_mask = (df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1])
    train_df = df[train_mask]
    
    # We already have country_historical_avg_gdp_growth in the dataset, but to be strictly safe and add others:
    country_stats = train_df.groupby("country_code").agg({
        "gdp_growth_pct": ["mean", "std"],
        "inflation_cpi_pct": ["mean", "std"]
    })
    country_stats.columns = [
        "country_hist_avg_gdp_growth", 
        "country_hist_std_gdp_growth",
        "country_hist_avg_inflation",
        "country_hist_std_inflation"
    ]
    country_stats = country_stats.reset_index()
    
    # Global fallbacks for unknown countries (from train data only)
    global_avg_gdp = train_df["gdp_growth_pct"].mean()
    global_std_gdp = train_df["gdp_growth_pct"].std()
    global_avg_inf = train_df["inflation_cpi_pct"].mean()
    global_std_inf = train_df["inflation_cpi_pct"].std()
    
    df = df.merge(country_stats, on="country_code", how="left")
    df["country_hist_avg_gdp_growth"] = df["country_hist_avg_gdp_growth"].fillna(global_avg_gdp)
    df["country_hist_std_gdp_growth"] = df["country_hist_std_gdp_growth"].fillna(global_std_gdp)
    df["country_hist_avg_inflation"] = df["country_hist_avg_inflation"].fillna(global_avg_inf)
    df["country_hist_std_inflation"] = df["country_hist_std_inflation"].fillna(global_std_inf)

    # ---------------------------------------------------------
    # Feature sets
    # ---------------------------------------------------------
    # Exp A uses exact Step 8 features (which the prompt says best addition was rolling volatility)
    # We will use base + momentum + rolling + global (as per Step 8 advanced script, or just replicate exactly)
    # Let's assume Step 8 included:
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
    step8_features = base_features + ["gdp_growth_momentum_1_2", "inflation_momentum_1_2", "unemployment_momentum_1_2", "exports_momentum", "imports_momentum", "gdp_growth_rolling_std_5", "inflation_rolling_std_3", "global_gdp_growth_lag1", "global_inflation_lag1"]
    
    # We define numerical vs categorical features for identity
    
    experiments = {
        "Exp A": {"num": step8_features, "cat": []},
        "Exp B": {"num": step8_features + ["country_hist_avg_gdp_growth"], "cat": []},
        "Exp C": {"num": step8_features + ["country_hist_avg_gdp_growth", "country_hist_std_gdp_growth", "country_hist_avg_inflation", "country_hist_std_inflation"], "cat": []},
        "Exp D": {"num": step8_features, "cat": ["country_code"]},
        "Exp E": {"num": step8_features + ["country_hist_avg_gdp_growth", "country_hist_std_gdp_growth", "country_hist_avg_inflation", "country_hist_std_inflation"], "cat": ["country_code"]}
    }
    
    # Splits
    valid_target = df["next_year_target_available"] == 1
    train = df[(df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1]) & valid_target]
    val = df[(df["year"] >= VAL_YEARS[0]) & (df["year"] <= VAL_YEARS[1]) & valid_target]
    test = df[(df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1]) & valid_target]
    inf_2025 = df[df["year"] == 2025]
    
    print("\n--- Model Evaluation (HistGradientBoosting) ---")
    
    y_train = train["gdp_growth_next_year"].values
    y_val = val["gdp_growth_next_year"].values
    y_test = test["gdp_growth_next_year"].values
    
    exp_results = {}
    best_exp_rmse = float('inf')
    best_exp_name = ""
    best_exp_config = None
    
    for name, config in experiments.items():
        # Build Preprocessing Pipeline
        numeric_features = config["num"]
        categorical_features = config["cat"]
        
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        transformers = [("num", numeric_transformer, numeric_features)]
        
        if len(categorical_features) > 0:
            categorical_transformer = Pipeline(steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat", categorical_transformer, categorical_features))
            
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(random_state=42))
        ])
        
        X_train = train[numeric_features + categorical_features]
        X_val = val[numeric_features + categorical_features]
        X_test = test[numeric_features + categorical_features]
        
        pipe.fit(X_train, y_train)
        
        val_preds = pipe.predict(X_val)
        test_preds = pipe.predict(X_test)
        
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
    numeric_features = best_exp_config["num"]
    categorical_features = best_exp_config["cat"]
    
    transformers = [("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features)]
    if len(categorical_features) > 0:
        transformers.append(("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    
    base_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingRegressor(random_state=42))
    ])
    
    param_grid = {
        'model__max_iter': [100, 300],
        'model__max_depth': [3, 5, 8],
        'model__learning_rate': [0.05, 0.1],
        'model__l2_regularization': [0.0, 1.0, 5.0]
    }
    
    X_train = train[numeric_features + categorical_features]
    X_val = val[numeric_features + categorical_features]
    X_test = test[numeric_features + categorical_features]
    
    best_pipe = None
    best_tune_rmse = float('inf')
    
    from sklearn.base import clone
    for params in ParameterGrid(param_grid):
        pipe = clone(base_pipe)
        pipe.set_params(**params)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        if rmse < best_tune_rmse:
            best_tune_rmse = rmse
            best_pipe = pipe
            
    print(f"Tuned Val RMSE = {best_tune_rmse:.4f}")
    
    print("\n--- Final Test Evaluation ---")
    test_preds = best_pipe.predict(X_test)
    final_test_m = metrics(y_test, test_preds)
    
    print(f"Final Test RMSE: {final_test_m['RMSE']:.4f}")
    
    print("\n--- 5-Country Verification ---")
    five_countries = ["IND", "CHN", "USA", "JPN", "GBR"]
    for code in five_countries:
        for year in [2024, 2025]: # 2024, 2025 targets are test
            # The test features year is 2023, 2024. Wait, the actual 'year' in df is the feature year. 
            # If target year is 2024, feature year is 2023.
            feature_year = year - 1
            mask = (test["country_code"] == code) & (test["year"] == feature_year)
            if mask.sum() > 0:
                actual = test.loc[mask, "gdp_growth_next_year"].values[0]
                pred = best_pipe.predict(test.loc[mask, numeric_features + categorical_features])[0]
                error = pred - actual
                print(f"{code} (Target {year}): Actual={actual:.2f}%, Predicted={pred:.2f}%, Error={error:.2f}%")
    
    print("\n--- 2026 Forecast Generation ---")
    inf_X = inf_2025[numeric_features + categorical_features]
    preds_2026 = best_pipe.predict(inf_X)
    inf_2025["predicted_gdp_growth"] = preds_2026
    inf_2025["target_year"] = 2026
    inf_2025["model_version"] = "t1_country_aware"
    
    for code in five_countries:
        mask = inf_2025["country_code"] == code
        if mask.sum() > 0:
            pred = inf_2025.loc[mask, "predicted_gdp_growth"].values[0]
            print(f"{code} 2026 Forecast: {pred:.2f}%")
            
    inf_out = inf_2025[["country_code", "country_name", "year", "target_year", "predicted_gdp_growth", "model_version"]].rename(columns={"year": "feature_year"})
    inf_out.to_csv(FORECAST_PATH, index=False)
    
    # Save test predictions
    test_out = test.copy()
    test_out["predicted_gdp_growth_next_year"] = test_preds
    test_out.to_csv(PREDICTIONS_PATH, index=False)
    
    # Compute permutation importance (only numerical features for simplicity, or handle OHE explicitly)
    # Scikit-learn handles it on the raw dataframe if passed into the pipeline
    from sklearn.inspection import permutation_importance
    r = permutation_importance(best_pipe, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({"Feature": X_val.columns, "Importance": r.importances_mean}).sort_values("Importance", ascending=False)
    
    # Save the full results and metadata
    results_out = {
        "best_feature_set": best_exp_name,
        "val_rmse": best_tune_rmse,
        "test_rmse": final_test_m["RMSE"],
        "experiments": exp_results,
        "importance": importance_df.head(10).to_dict(orient="records")
    }
    
    with open(MODELS_DIR / "t1_country_aware_model_metadata.json", "w") as f:
        json.dump(results_out, f)

if __name__ == "__main__":
    main()
