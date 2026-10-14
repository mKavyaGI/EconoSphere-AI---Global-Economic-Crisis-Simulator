"""
EconoSphere AI -- Accuracy Optimization Benchmark Experiment
======================================================================
Tests 5 concrete improvements against the Locked Baseline:
  1. Feature Cleaning (Drop raw dollar/LCU nominals, add global & momentum features)
  2. Loss Function Optimization (Huber loss to tame extreme outliers)
  3. Hyperparameter Regularization (Reduce overfitting from depth=5 to depth=3/4)
  4. Target Formulation (Residual / Deviation from country historical mean)
  5. Multi-Model Ensembling (A5a: HistGradientBoosting + RandomForest + Ridge)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"

TRAIN_YEARS = (2000, 2018)
VAL_YEARS = (2019, 2022)
TEST_YEARS = (2023, 2024)

WB_AGGREGATES = {
    'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 'EMU', 'EUU', 
    'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN', 'LDC', 
    'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PRE', 'PSS', 
    'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WLD'
}

def get_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true))
    }

def print_result(name, val_m, test_m):
    print(f"\n[{name}]")
    print(f"  Val  -> RMSE: {val_m['RMSE']:.4f} | MAE: {val_m['MAE']:.4f} | R2: {val_m['R2']:.4f} | Bias: {val_m['Bias']:+.4f}")
    print(f"  Test -> RMSE: {test_m['RMSE']:.4f} | MAE: {test_m['MAE']:.4f} | R2: {test_m['R2']:.4f} | Bias: {test_m['Bias']:+.4f}")

def main():
    df = pd.read_csv(INPUT_PATH)
    df = df[~df["country_code"].isin(WB_AGGREGATES)].copy()
    valid_target = df["next_year_target_available"] == 1
    
    train = df[(df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1]) & valid_target].copy()
    val = df[(df["year"] >= VAL_YEARS[0]) & (df["year"] <= VAL_YEARS[1]) & valid_target].copy()
    test = df[(df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1]) & valid_target].copy()
    
    # -------------------------------------------------------------------------
    # 1. BASELINE CONFIGURATION (Step 10 Locked Control)
    # -------------------------------------------------------------------------
    baseline_features = [
        "exchange_rate_lcu_usd", "tariff_rate_pct", "remittances_usd", "fdi_net_inflow_usd", 
        "unemployment_pct", "imports_pct_gdp", "tax_revenue_pct_gdp", "exports_pct_gdp", 
        "interest_rate_pct", "reserves_usd", "current_account_pct_gdp", "inflation_cpi_pct", 
        "population_total", "gdp_current_usd", "gdp_growth_lag1", "gdp_growth_lag2", 
        "gdp_growth_lag3", "inflation_lag1", "unemployment_lag1", "exports_lag1", 
        "imports_lag1", "gdp_growth_rolling_mean_3", "gdp_growth_rolling_std_3", 
        "gdp_growth_rolling_mean_5", "inflation_rolling_mean_3", "trade_openness", 
        "trade_balance_ratio", "log_gdp_usd", "log_population",
        "gdp_growth_rolling_std_5", "inflation_rolling_std_3"
    ]
    
    y_train = train["gdp_growth_next_year"].values
    y_val = val["gdp_growth_next_year"].values
    y_test = test["gdp_growth_next_year"].values
    
    pipe_baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.05, max_depth=5, max_iter=300, l2_regularization=5.0, random_state=42
        ))
    ])
    pipe_baseline.fit(train[baseline_features], y_train)
    m_base_val = get_metrics(y_val, pipe_baseline.predict(val[baseline_features]))
    m_base_test = get_metrics(y_test, pipe_baseline.predict(test[baseline_features]))
    print_result("0. Baseline (Locked Control)", m_base_val, m_base_test)

    # -------------------------------------------------------------------------
    # 2. CANDIDATE A: Cleaned Features (Drop raw scale non-stationaries, add global/momentum)
    # -------------------------------------------------------------------------
    # Dropped: exchange_rate_lcu_usd (arbitrary denomination), gdp_current_usd (Z=19 OOD), 
    #          population_total, reserves_usd, remittances_usd, fdi_net_inflow_usd (raw dollars)
    # Added: global_gdp_growth_lag1, global_inflation_lag1, gdp_growth_momentum_1_2,
    #        country_historical_avg_gdp_growth, total_missing_feature_ratio
    clean_features = [
        "tariff_rate_pct", "unemployment_pct", "imports_pct_gdp", "tax_revenue_pct_gdp", 
        "exports_pct_gdp", "interest_rate_pct", "current_account_pct_gdp", "inflation_cpi_pct", 
        "gdp_growth_lag1", "gdp_growth_lag2", "gdp_growth_lag3", "inflation_lag1", 
        "unemployment_lag1", "exports_lag1", "imports_lag1", 
        "gdp_growth_rolling_mean_3", "gdp_growth_rolling_std_3", 
        "gdp_growth_rolling_mean_5", "gdp_growth_rolling_std_5",
        "inflation_rolling_mean_3", "inflation_rolling_std_3", 
        "trade_openness", "trade_balance_ratio", "log_gdp_usd", "log_population",
        "gdp_growth_momentum_1_2", "inflation_momentum_1_2",
        "country_historical_avg_gdp_growth", "global_gdp_growth_lag1", "global_inflation_lag1",
        "total_missing_feature_ratio"
    ]
    pipe_clean = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.05, max_depth=4, max_iter=200, l2_regularization=10.0, random_state=42
        ))
    ])
    pipe_clean.fit(train[clean_features], y_train)
    m_clean_val = get_metrics(y_val, pipe_clean.predict(val[clean_features]))
    m_clean_test = get_metrics(y_test, pipe_clean.predict(test[clean_features]))
    print_result("1. Cleaned Features + Depth=4 Regularization", m_clean_val, m_clean_test)

    # -------------------------------------------------------------------------
    # 3. CANDIDATE B: Robust Absolute Error Loss (Taming macro outliers)
    # -------------------------------------------------------------------------
    pipe_robust = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.05, max_depth=4, max_iter=200, 
            l2_regularization=10.0, min_samples_leaf=25, random_state=42
        ))
    ])
    pipe_robust.fit(train[clean_features], y_train)
    m_huber_val = get_metrics(y_val, pipe_robust.predict(val[clean_features]))
    m_huber_test = get_metrics(y_test, pipe_robust.predict(test[clean_features]))
    print_result("2. Cleaned Features + Absolute Error Loss (L1 Robust)", m_huber_val, m_huber_test)

    # -------------------------------------------------------------------------
    # 4. CANDIDATE C: Target Deviation Formulation (Predict deviation from 5-yr rolling mean)
    # -------------------------------------------------------------------------
    # y* = gdp_growth_next_year - trend
    trend_train = train["gdp_growth_rolling_mean_5"].fillna(train["country_historical_avg_gdp_growth"]).fillna(train["gdp_growth_lag1"]).fillna(3.5).values
    y_dev_train = y_train - trend_train
    
    pipe_dev = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.04, max_depth=3, max_iter=150, 
            l2_regularization=15.0, min_samples_leaf=30, random_state=42
        ))
    ])
    pipe_dev.fit(train[clean_features], y_dev_train)
    
    val_trend = val["gdp_growth_rolling_mean_5"].fillna(val["country_historical_avg_gdp_growth"]).fillna(val["gdp_growth_lag1"]).fillna(3.5).values
    test_trend = test["gdp_growth_rolling_mean_5"].fillna(test["country_historical_avg_gdp_growth"]).fillna(test["gdp_growth_lag1"]).fillna(3.5).values
    
    pred_dev_val = pipe_dev.predict(val[clean_features]) + val_trend
    pred_dev_test = pipe_dev.predict(test[clean_features]) + test_trend
    
    m_dev_val = get_metrics(y_val, pred_dev_val)
    m_dev_test = get_metrics(y_test, pred_dev_test)
    print_result("3. Target Deviation Forecasting (y* = y - trend)", m_dev_val, m_dev_test)

    # -------------------------------------------------------------------------
    # 5. CANDIDATE D: Multi-Model Ensemble (A5a+: HGB + Tuned RF + Ridge)
    # -------------------------------------------------------------------------
    # 1. HGB
    p_hgb_val = pipe_baseline.predict(val[baseline_features])
    p_hgb_test = pipe_baseline.predict(test[baseline_features])
    
    # 2. Random Forest
    pipe_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1))
    ])
    pipe_rf.fit(train[clean_features], y_train)
    p_rf_val = pipe_rf.predict(val[clean_features])
    p_rf_test = pipe_rf.predict(test[clean_features])
    
    # 3. Ridge (linear baseline)
    pipe_ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=100.0, random_state=42))
    ])
    pipe_ridge.fit(train[clean_features], y_train)
    p_ridge_val = pipe_ridge.predict(val[clean_features])
    p_ridge_test = pipe_ridge.predict(test[clean_features])
    
    # Blended ensemble (equal-weight or tuned weights)
    p_ens_val = 0.40 * p_hgb_val + 0.35 * p_rf_val + 0.25 * p_ridge_val
    p_ens_test = 0.40 * p_hgb_test + 0.35 * p_rf_test + 0.25 * p_ridge_test
    
    m_ens_val = get_metrics(y_val, p_ens_val)
    m_ens_test = get_metrics(y_test, p_ens_test)
    print_result("4. Multi-Model Ensemble (HGB-Huber + RF + Ridge)", m_ens_val, m_ens_test)

    # -------------------------------------------------------------------------
    # 6. CANDIDATE E: Target Deviation Multi-Model Ensemble (HGB + RF + Ridge on y*)
    # -------------------------------------------------------------------------
    # 1. HGB on deviation
    p_dev_hgb_val = pipe_dev.predict(val[clean_features])
    p_dev_hgb_test = pipe_dev.predict(test[clean_features])
    
    # 2. RF on deviation
    pipe_rf_dev = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_leaf=25, random_state=42, n_jobs=-1))
    ])
    pipe_rf_dev.fit(train[clean_features], y_dev_train)
    p_dev_rf_val = pipe_rf_dev.predict(val[clean_features])
    p_dev_rf_test = pipe_rf_dev.predict(test[clean_features])
    
    # 3. Ridge on deviation
    pipe_ridge_dev = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=100.0, random_state=42))
    ])
    pipe_ridge_dev.fit(train[clean_features], y_dev_train)
    p_dev_ridge_val = pipe_ridge_dev.predict(val[clean_features])
    p_dev_ridge_test = pipe_ridge_dev.predict(test[clean_features])
    
    # Blend deviation predictions, then add trend back
    p_dev_ens_val_raw = 0.40 * p_dev_hgb_val + 0.35 * p_dev_rf_val + 0.25 * p_dev_ridge_val
    p_dev_ens_test_raw = 0.40 * p_dev_hgb_test + 0.35 * p_dev_rf_test + 0.25 * p_dev_ridge_test
    
    p_dev_ens_val = p_dev_ens_val_raw + val_trend
    p_dev_ens_test = p_dev_ens_test_raw + test_trend
    
    m_dev_ens_val = get_metrics(y_val, p_dev_ens_val)
    m_dev_ens_test = get_metrics(y_test, p_dev_ens_test)
    print_result("5. Deviation Ensemble (HGB + RF + Ridge on y*)", m_dev_ens_val, m_dev_ens_test)

    # Summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY COMPARISON:")
    print("="*80)
    rows = [
        ("0. Baseline Locked Control", m_base_val, m_base_test),
        ("1. Cleaned Features (No raw scale)", m_clean_val, m_clean_test),
        ("2. Robust Loss (Outlier Dampening)", m_huber_val, m_huber_test),
        ("3. Target Deviation (HGB on y*)", m_dev_val, m_dev_test),
        ("4. Raw Level Ensemble (HGB+RF+Ridge)", m_ens_val, m_ens_test),
        ("5. Target Deviation Ensemble (HGB+RF+Ridge on y*)", m_dev_ens_val, m_dev_ens_test),
    ]
    print(f"{'Model Candidate':<48} | {'Val RMSE':<9} | {'Val MAE':<9} | {'Test RMSE':<10} | {'Test MAE':<9} | {'Test R2':<8}")
    print("-" * 105)
    for name, v, t in rows:
        print(f"{name:<48} | {v['RMSE']:<9.4f} | {v['MAE']:<9.4f} | {t['RMSE']:<10.4f} | {t['MAE']:<9.4f} | {t['R2']:<8.4f}")

    # Major Country Comparison for Test Year 2024
    test_eval = test.copy()
    test_eval["pred_baseline"] = m_base_test_preds = pipe_baseline.predict(test[baseline_features])
    test_eval["pred_ens_dev"] = p_dev_ens_test
    test_eval["err_baseline"] = np.abs(test_eval["pred_baseline"] - test_eval["gdp_growth_next_year"])
    test_eval["err_ens_dev"] = np.abs(test_eval["pred_ens_dev"] - test_eval["gdp_growth_next_year"])

    sample_countries = ['USA', 'CHN', 'IND', 'DEU', 'JPN', 'GBR', 'FRA', 'BRA', 'CAN', 'IDN']
    t24 = test_eval[test_eval["year"] == 2023].set_index("country_code")
    common = [c for c in sample_countries if c in t24.index]

    print("\n" + "="*80)
    print("MAJOR ECONOMIES: 2024 FORECAST ACCURACY (Actual vs Baseline vs Target Deviation Ensemble)")
    print("="*80)
    print(f"{'Country':<8} | {'Actual':<8} | {'Baseline':<9} | {'Base Error':<11} | {'Dev Ensemble':<13} | {'Ensemble Error':<14} | {'Winner':<8}")
    print("-" * 85)
    for c in common:
        row = t24.loc[c]
        act = row["gdp_growth_next_year"]
        b_p = row["pred_baseline"]
        b_e = row["err_baseline"]
        e_p = row["pred_ens_dev"]
        e_e = row["err_ens_dev"]
        winner = "ENSEMBLE" if e_e < b_e else "BASELINE"
        print(f"{c:<8} | {act:<8.2f}% | {b_p:<9.2f}% | {b_e:<11.2f}% | {e_p:<13.2f}% | {e_e:<14.2f}% | {winner:<8}")

if __name__ == "__main__":
    main()
