import re
from pathlib import Path

f_path = Path("apps/api/ml/train_t1_baseline.py")
content = f_path.read_text(encoding="utf-8")

# 1. Update header comments
content = content.replace("Step 5: Baseline ML Training & Evaluation", "Step 7: Baseline T+1 ML Training, Evaluation and Prediction")
content = content.replace("Script   : apps/api/ml/train_baseline.py", "Script   : apps/api/ml/train_t1_baseline.py")
content = content.replace("Input    : data/processed/master_panel_processed.csv", "Input    : data/processed/master_panel_forecasting.csv")
content = content.replace("docs/phase11/baseline_ml_training_report.md", "docs/phase11/baseline_t1_training_report.md")
content = content.replace("python apps/api/ml/train_baseline.py", "python apps/api/ml/train_t1_baseline.py")

# 2. Add DummyRegressor import
content = content.replace(
    "from sklearn.linear_model import LinearRegression, Ridge",
    "from sklearn.dummy import DummyRegressor\nfrom sklearn.linear_model import LinearRegression, Ridge"
)

# 3. Update paths and constants
content = content.replace(
    'PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_processed.csv"',
    'PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_forecasting.csv"'
)
content = content.replace(
    'REPORT_PATH    = REPORT_DIR / "baseline_ml_training_report.md"',
    'REPORT_PATH    = REPORT_DIR / "baseline_t1_training_report.md"'
)
content = content.replace(
    'METADATA_PATH  = MODELS_DIR / "model_metadata.json"',
    'METADATA_PATH  = MODELS_DIR / "t1_model_metadata.json"\nPREDICTIONS_CSV  = PROJECT_ROOT / "data" / "processed" / "t1_baseline_predictions.csv"\nFORECASTS_CSV    = PROJECT_ROOT / "data" / "processed" / "t1_2026_forecasts.csv"'
)

content = content.replace('TARGET_COL   = "gdp_growth_pct"', 'TARGET_COL   = "gdp_growth_next_year"')

# 4. Feature logic documentation
content = content.replace(
    '# gdp_growth_pct is the TARGET -- must never appear here.',
    '# gdp_growth_pct is deliberately excluded to prevent same-year end-of-year information leakage,\n# meaning the model relies strictly on lagged and other historically safe features.\n# gdp_growth_next_year is the TARGET -- must never appear here.\n# next_year_target_available is excluded.'
)

# 5. Add Dummy pipeline
content = content.replace(
    '"Linear Regression": Pipeline([',
    '"Dummy (Mean)": Pipeline([\n            ("prep",  standard_preprocessor),\n            ("model", DummyRegressor(strategy="mean")),\n        ]),\n        "Linear Regression": Pipeline(['
)
content = content.replace(
    '"Linear Regression": {\n                "class": "sklearn.linear_model.Ridge",',
    '"Dummy (Mean)": {\n                "class": "sklearn.dummy.DummyRegressor",\n                "params": {"strategy": "mean"},\n                "val_metrics": all_metrics.get("Dummy (Mean)", {}).get("val"),\n                "test_metrics": all_metrics.get("Dummy (Mean)", {}).get("test"),\n                "artifact": str(saved_models.get("Dummy (Mean)", "")),\n            },\n            "Linear Regression": {\n                "class": "sklearn.linear_model.Ridge",'
)
content = content.replace(
    '"Linear Regression":    "linear_regression_baseline.joblib",',
    '"Dummy (Mean)":         "t1_dummy_baseline.joblib",\n        "Linear Regression":    "t1_linear_regression_baseline.joblib",\n        "Random Forest":        "t1_random_forest_baseline.joblib",\n        "HistGradientBoosting": "t1_gradient_boosting_baseline.joblib",'
)
content = content.replace(
    'best_path = MODELS_DIR / "best_gdp_growth_model.joblib"',
    'best_path = MODELS_DIR / "best_t1_gdp_growth_model.joblib"'
)

# 6. Change model metadata exclusion list
content = content.replace(
    '"excluded_columns": ["govt_debt_pct_gdp", "gdp_growth_pct (target)"],',
    '"excluded_columns": ["govt_debt_pct_gdp", "gdp_growth_pct", "gdp_growth_next_year (target)", "next_year_target_available"],'
)

# 7. Update Inference step (2025 rows) in main
content = content.replace(
    '# --- Validation comparison table ---',
    '# --- Validation comparison table ---'
)

inference_code = """
# ===========================================================================
# STEP 10 -- 2026 FORECAST INFERENCE
# ===========================================================================
def generate_2026_forecasts(best_pipe, test_df, original_df, best_name):
    log("STEP 10 -- Generating 2026 forecasts from 2025 inference rows")
    
    # 2025 rows
    df_2025 = original_df[original_df["year"] == 2025].copy()
    
    # Exclude aggregate codes
    df_2025 = df_2025[~df_2025["country_code"].isin(WB_AGGREGATE_CODES)].copy()
    
    if df_2025.empty:
        log("  No 2025 inference rows found.")
        return
        
    X_inf = df_2025[FEATURE_COLUMNS]
    
    preds_2026 = best_pipe.predict(X_inf)
    
    forecasts = pd.DataFrame({
        "country_code": df_2025["country_code"].values,
        "country_name": df_2025["country_name"].values,
        "feature_year": 2025,
        "target_year": 2026,
        "forecast_gdp_growth": preds_2026
    })
    
    forecasts.to_csv(FORECASTS_CSV, index=False)
    log(f"  2026 forecasts saved to {FORECASTS_CSV}")
    
    codes = ["IND", "CHN", "USA", "JPN", "GBR"]
    for c in codes:
        row = forecasts[forecasts["country_code"] == c]
        if not row.empty:
            val = row["forecast_gdp_growth"].iloc[0]
            log(f"  {c} 2026 forecast: {val:.3f}%")
            
    # Also save the actual vs predicted for the test set
    preds_test = best_pipe.predict(test_df[FEATURE_COLUMNS])
    test_results = pd.DataFrame({
        "country_code": test_df["country_code"].values,
        "country_name": test_df["country_name"].values,
        "feature_year": test_df["year"].values.astype(int),
        "target_year": test_df["year"].values.astype(int) + 1,
        "actual_gdp_growth": test_df[TARGET_COL].values,
        "predicted_gdp_growth": preds_test,
        "error": preds_test - test_df[TARGET_COL].values,
        "absolute_error": np.abs(preds_test - test_df[TARGET_COL].values)
    })
    test_results.to_csv(PREDICTIONS_CSV, index=False)
    log(f"  Test predictions saved to {PREDICTIONS_CSV}")
    
    # Detailed display for target countries
    for c in codes:
        sub = test_results[test_results["country_code"] == c]
        if not sub.empty:
            log(f"  {c} Test predictions (2023-2024 features predicting 2024-2025):")
            for _, r in sub.iterrows():
                log(f"    Target {r['target_year']}: Act={r['actual_gdp_growth']:.3f}, Pred={r['predicted_gdp_growth']:.3f}, Err={r['error']:.3f}")

"""

# Insert inference code before MAIN
content = content.replace(
    '# ===========================================================================\n# MAIN\n# ===========================================================================',
    inference_code + '\n# ===========================================================================\n# MAIN\n# ==========================================================================='
)

# Call inference code in MAIN
import re
content = re.sub(
    r"save_metadata\(\s*best_name,\s*all_metrics,\s*n_train,\s*n_val,\s*n_test,\s*n_countries,\s*saved_models\s*\)",
    "save_metadata(best_name, all_metrics, n_train, n_val, n_test, n_countries, saved_models)\n    generate_2026_forecasts(trained_pipes[best_name], test_df, df_countries, best_name)",
    content
)

# Write modified content
f_path.write_text(content, encoding="utf-8")
