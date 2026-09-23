import hashlib
import pandas as pd
import json
from pathlib import Path
import os

PROJECT_ROOT = Path("D:/Development/EconoSphere AI")
RAW_PATH = PROJECT_ROOT / "data/raw/master_panel.csv"
PROC_PATH = PROJECT_ROOT / "data/processed/master_panel_processed.csv"
FC_PATH = PROJECT_ROOT / "data/processed/master_panel_forecasting.csv"
PRED_PATH = PROJECT_ROOT / "data/processed/t1_baseline_predictions.csv"
F26_PATH = PROJECT_ROOT / "data/processed/t1_2026_forecasts.csv"
META_PATH = PROJECT_ROOT / "models/phase11/t1_model_metadata.json"
TRAIN_SCRIPT = PROJECT_ROOT / "apps/api/ml/train_t1_baseline.py"

results = {}

# 1. RAW DATA
try:
    with open(RAW_PATH, 'rb') as f:
        raw_md5 = hashlib.md5(f.read()).hexdigest()
    df_raw = pd.read_csv(RAW_PATH)
    results["raw"] = {"md5": raw_md5, "rows": len(df_raw), "cols": len(df_raw.columns)}
except Exception as e:
    results["raw"] = str(e)

# 2. PROCESSED DATA
try:
    df_proc = pd.read_csv(PROC_PATH)
    results["proc"] = {
        "rows": int(len(df_proc)), 
        "cols": int(len(df_proc.columns)),
        "dup_id_year": int(df_proc.duplicated(subset=['country_code', 'year']).sum()),
        "inx_exists": bool('INX' in df_proc['country_code'].values),
        "nru_exists": bool('NRU' in df_proc['country_code'].values),
        "gdp_growth_pct_exists": bool('gdp_growth_pct' in df_proc.columns)
    }
except Exception as e:
    results["proc"] = str(e)

# 3. FORECASTING DATA
try:
    df_fc = pd.read_csv(FC_PATH)
    results["fc"] = {
        "rows": int(len(df_fc)),
        "cols": int(len(df_fc.columns)),
        "gdp_next_exists": bool('gdp_growth_next_year' in df_fc.columns),
        "avail_exists": bool('next_year_target_available' in df_fc.columns)
    }
    
    # Check T+1 alignment programmatically
    df_fc_sorted = df_fc.sort_values(['country_code', 'year']).reset_index(drop=True)
    df_fc_sorted['shifted_actual'] = df_fc_sorted.groupby('country_code')['gdp_growth_pct'].shift(-1)
    
    # Compare where next_year_target_available is 1
    valid_mask = df_fc_sorted['next_year_target_available'] == 1
    mismatches = df_fc_sorted[valid_mask][
        ~np.isclose(df_fc_sorted[valid_mask]['gdp_growth_next_year'], df_fc_sorted[valid_mask]['shifted_actual'], equal_nan=True)
    ]
    results["fc"]["mismatches"] = int(len(mismatches))
except Exception as e:
    results["fc"] = str(e)

# 4. 2025 INFERENCE
try:
    df_2025 = df_fc[df_fc['year'] == 2025]
    results["inf_2025"] = {
        "rows": int(len(df_2025)),
        "target_nans": int(df_2025['gdp_growth_next_year'].isna().sum()),
        "avail_sum": int(df_2025['next_year_target_available'].sum())
    }
except Exception as e:
    results["inf_2025"] = str(e)

# 5, 6, 7. TRAIN SCRIPT ANALYSIS
try:
    with open(TRAIN_SCRIPT, 'r') as f:
        script = f.read()
    results["train_script"] = {
        "target_col": "TARGET_COL   = \"gdp_growth_next_year\"" in script,
        "feature_cols_has_target": "gdp_growth_next_year" in script.split("FEATURE_COLUMNS")[1].split("]")[0],
        "feature_cols_has_pct": "gdp_growth_pct" in script.split("FEATURE_COLUMNS")[1].split("]")[0],
        "train_years": "TRAIN_YEARS = (2000, 2018)" in script,
        "val_years": "VAL_YEARS   = (2019, 2022)" in script,
        "test_years": "TEST_YEARS  = (2023, 2025)" in script, # Note test is actually predicting 2024-2026, let's verify exact
        "scaler": "StandardScaler()" in script
    }
except Exception as e:
    results["train_script"] = str(e)

# 9. TEST SET & 10. FIVE TARGET COUNTRIES
try:
    df_pred = pd.read_csv(PRED_PATH)
    results["test_pred"] = {
        "rows": len(df_pred),
        "feature_years": df_pred['feature_year'].unique().tolist(),
        "target_years": df_pred['target_year'].unique().tolist(),
        "uses_pct": 'gdp_growth_pct' in df_pred.columns
    }
    targets = ["IND", "CHN", "USA", "JPN", "GBR"]
    country_res = {}
    for c in targets:
        sub = df_pred[df_pred['country_code'] == c]
        country_res[c] = {}
        for y in [2024, 2025]:
            row = sub[sub['target_year'] == y]
            if not row.empty:
                country_res[c][y] = {
                    "act": float(row['actual_gdp_growth'].iloc[0]),
                    "pred": float(row['predicted_gdp_growth'].iloc[0]),
                    "err": float(row['error'].iloc[0])
                }
    results["targets_test"] = country_res
except Exception as e:
    results["test_pred"] = str(e)

# 11. 2026 INFERENCE
try:
    df_f26 = pd.read_csv(F26_PATH)
    results["f26"] = {
        "rows": len(df_f26),
        "feature_years": df_f26['feature_year'].unique().tolist(),
        "target_years": df_f26['target_year'].unique().tolist(),
        "actual_exists": 'actual_gdp_growth' in df_f26.columns
    }
    f26_res = {}
    for c in targets:
        row = df_f26[df_f26['country_code'] == c]
        if not row.empty:
            f26_res[c] = float(row['forecast_gdp_growth'].iloc[0])
    results["targets_f26"] = f26_res
except Exception as e:
    results["f26"] = str(e)

# 12. METADATA
try:
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    results["meta"] = {
        "target": meta.get("target"),
        "train_period": meta.get("train_period"),
        "val_period": meta.get("validation_period"),
        "test_period": meta.get("test_period"),
        "best": meta.get("best_model_name") # wait, does meta store best model? 
    }
    # find best model by checking RMSE
    val_rmses = {}
    for k, v in meta.get("models", {}).items():
        if "val_metrics" in v:
            val_rmses[k] = v["val_metrics"]["RMSE"]
    best_m = min(val_rmses, key=val_rmses.get)
    results["meta"]["best_derived"] = best_m
    results["meta"]["best_rmse"] = val_rmses[best_m]
except Exception as e:
    results["meta"] = str(e)

with open("audit_results.json", "w") as f:
    json.dump(results, f, indent=2)
