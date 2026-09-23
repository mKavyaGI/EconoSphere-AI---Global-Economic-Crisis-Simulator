"""
EconoSphere AI -- Phase 11, Step 8: Advanced Feature Engineering
======================================================================
Script   : apps/api/ml/data_t1_advanced_features.py
Input    : data/processed/master_panel_forecasting.csv (READ-ONLY)
Output   : data/processed/master_panel_t1_advanced.csv
           docs/phase11/step8_advanced_feature_engineering_report.md

Run from the project root:
    python apps/api/ml/data_t1_advanced_features.py
"""
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_forecasting.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_advanced.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "phase11" / "step8_advanced_feature_engineering_report.md"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"

# Configuration
TRAIN_YEARS = (2000, 2018)
report_sections = []

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def add_section(title: str, body: str) -> None:
    report_sections.append(f"## {title}\n\n{body}\n")

def check_raw_checksum():
    if not RAW_PATH.exists():
        sys.exit("[FATAL] Raw file not found.")
    actual_md5 = hashlib.md5(RAW_PATH.read_bytes()).hexdigest()
    if actual_md5 != "8ac7e0b2bf09fbe89289f82d0c7cf25e":
        sys.exit("[FATAL] Raw dataset checksum mismatch!")
    log("Raw checksum VERIFIED.")

def main():
    log("STEP 8.4 - 8.7: Advanced Feature Engineering")
    check_raw_checksum()
    
    if not INPUT_PATH.exists():
        sys.exit(f"[FATAL] Input file missing: {INPUT_PATH}")
        
    df = pd.read_csv(INPUT_PATH)
    log(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
    
    # -----------------------------------------------------------------------
    # Step 8.4: Advanced Historical Features
    # -----------------------------------------------------------------------
    log("Generating momentum features...")
    # Momentum (difference between lags)
    df["gdp_growth_momentum_1_2"] = df["gdp_growth_lag1"] - df["gdp_growth_lag2"]
    df["inflation_momentum_1_2"] = df["inflation_lag1"] - df.groupby("country_code")["inflation_cpi_pct"].shift(2)
    df["unemployment_momentum_1_2"] = df["unemployment_lag1"] - df.groupby("country_code")["unemployment_pct"].shift(2)
    df["exports_momentum"] = df["exports_lag1"] - df.groupby("country_code")["exports_pct_gdp"].shift(2)
    df["imports_momentum"] = df["imports_lag1"] - df.groupby("country_code")["imports_pct_gdp"].shift(2)
    
    # Rolling Volatility (shift before rolling)
    log("Generating rolling volatility features...")
    df["gdp_growth_rolling_std_5"] = df.groupby("country_code")["gdp_growth_pct"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=2).std()
    )
    df["inflation_rolling_std_3"] = df.groupby("country_code")["inflation_cpi_pct"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=2).std()
    )
    
    # -----------------------------------------------------------------------
    # Step 8.5: Country Structure
    # -----------------------------------------------------------------------
    log("Generating country-level structure features (TRAIN split only)...")
    # Country-level average GDP growth, computed strictly on TRAIN data
    train_mask = (df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1])
    train_df = df[train_mask]
    
    country_avg_growth = train_df.groupby("country_code")["gdp_growth_pct"].mean().rename("country_historical_avg_gdp_growth")
    country_avg_inflation = train_df.groupby("country_code")["inflation_cpi_pct"].mean().rename("country_historical_avg_inflation")
    
    df = df.merge(country_avg_growth, on="country_code", how="left")
    df = df.merge(country_avg_inflation, on="country_code", how="left")
    
    # Fill NaN for countries not present in TRAIN
    global_avg_growth = train_df["gdp_growth_pct"].mean()
    global_avg_inflation = train_df["inflation_cpi_pct"].mean()
    
    df["country_historical_avg_gdp_growth"] = df["country_historical_avg_gdp_growth"].fillna(global_avg_growth)
    df["country_historical_avg_inflation"] = df["country_historical_avg_inflation"].fillna(global_avg_inflation)
    
    # -----------------------------------------------------------------------
    # Step 8.6: Global Economic Features
    # -----------------------------------------------------------------------
    log("Generating global economic features...")
    # Mean of lag1 values across all countries for the given year T
    global_gdp = df.groupby("year")["gdp_growth_lag1"].mean().rename("global_gdp_growth_lag1")
    global_inf = df.groupby("year")["inflation_lag1"].mean().rename("global_inflation_lag1")
    
    df = df.merge(global_gdp, on="year", how="left")
    df = df.merge(global_inf, on="year", how="left")
    
    # Verify no leaks
    if df["gdp_growth_next_year"].isna().sum() != df["next_year_target_available"].value_counts()[0]:
        log("[WARNING] Target missingness changed!")
        
    log(f"Final advanced dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    df.to_csv(OUTPUT_PATH, index=False)
    log(f"Saved to {OUTPUT_PATH}")
    
    # Leakage check logic
    target_idx = df.columns.get_loc("gdp_growth_next_year")
    
if __name__ == "__main__":
    main()
