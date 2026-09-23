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
WALK_FWD_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step3_walk_forward_metrics.csv"
WALK_FWD_PREDS_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step3_walk_forward_predictions.csv"
IMPORTANCE_PATH = PROJECT_ROOT / "data" / "processed" / "phase12_step3_feature_importance.csv"

WB_AGGREGATES = {
    'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'
}

def engineer_features(df):
    df = df.copy()
    df.sort_values(by=["country_code", "year"], inplace=True)
    df['interest_rate_lag1'] = df.groupby('country_code')['interest_rate_pct'].shift(1)
    df['exports_pct_gdp_lag1'] = df.groupby('country_code')['exports_pct_gdp'].shift(1)
    df['inflation_change'] = df['inflation_cpi_pct'] - df['inflation_lag1']
    df['unemployment_change'] = df['unemployment_pct'] - df['unemployment_lag1']
    df['interest_rate_change'] = df['interest_rate_pct'] - df['interest_rate_lag1']
    df['exports_pct_gdp_change'] = df['exports_pct_gdp'] - df['exports_pct_gdp_lag1']
    df['inflation_change_lag1'] = df.groupby('country_code')['inflation_change'].shift(1)
    df['inflation_acceleration'] = df['inflation_change'] - df['inflation_change_lag1']
    df['unemployment_deterioration'] = (df['unemployment_change'] > 0.5).astype(float)
    df['trade_contraction'] = (df['exports_pct_gdp_change'] < -2.0).astype(float)
    df['interest_rate_tightening'] = (df['interest_rate_change'] > 0.5).astype(float)
    df['gdp_growth_change'] = df['gdp_growth_pct'] - df['gdp_growth_lag1']
    return df

def main():
    print("STEP 3: Walk-Forward Evaluation")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    df = engineer_features(df)
    valid_target = df["next_year_target_available"] == 1
    
    meta_path = MODELS_DIR / "step3_forward_indicators_metadata.json"
    if not meta_path.exists():
        print("Metadata not found, run training first.")
        return
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    best_exp_name = meta["best_experiment"]
    features = meta["features_used"]
    
    print(f"Evaluating {best_exp_name} with {len(features)} features")
    
    locked_params = {
        'l2_regularization': 5.0, 
        'learning_rate': 0.05, 
        'max_depth': 5, 
        'max_iter': 300,
        'random_state': 42
    }
    
    windows = list(range(2013, 2025))
    all_preds = []
    window_metrics = []
    
    importances = []
    
    for target_year in windows:
        feature_year = target_year - 1
        
        train = df[(df["year"] < feature_year) & valid_target].copy()
        test = df[(df["year"] == feature_year) & valid_target].copy()
        
        if len(test) == 0:
            continue
            
        X_train = train[features]
        y_train = train["gdp_growth_next_year"].values
        X_test = test[features]
        y_test = test["gdp_growth_next_year"].values
        
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(**locked_params))
        ])
        
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        window_metrics.append({
            "Target_Year": target_year,
            "RMSE": rmse,
            "MAE": mean_absolute_error(y_test, preds)
        })
        
        test_out = test.copy()
        test_out["predicted_gdp_growth"] = preds
        test_out["target_year"] = target_year
        all_preds.append(test_out)
        
        # Calculate permutation importance on test set (or train if preferred, but test is standard for holdout)
        # However, for walk-forward, we can accumulate
        
    df_metrics = pd.DataFrame(window_metrics)
    df_metrics.to_csv(WALK_FWD_METRICS_PATH, index=False)
    
    df_preds = pd.concat(all_preds, ignore_index=True)
    df_preds.to_csv(WALK_FWD_PREDS_PATH, index=False)
    
    overall_rmse = np.sqrt(mean_squared_error(df_preds["gdp_growth_next_year"], df_preds["predicted_gdp_growth"]))
    print(f"Overall Walk-Forward RMSE: {overall_rmse:.4f}")
    
    # Recession metrics over walk-forward
    recession_mask = df_preds["gdp_growth_next_year"] < 0
    rec_rmse = np.sqrt(mean_squared_error(df_preds.loc[recession_mask, "gdp_growth_next_year"], df_preds.loc[recession_mask, "predicted_gdp_growth"]))
    print(f"Recession Walk-Forward RMSE: {rec_rmse:.4f}")

if __name__ == "__main__":
    main()
