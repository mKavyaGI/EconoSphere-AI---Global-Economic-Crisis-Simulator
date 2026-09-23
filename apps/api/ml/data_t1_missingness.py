"""
EconoSphere AI -- Phase 11, Step 10: Missingness Profile Generator
======================================================================
Script   : apps/api/ml/data_t1_missingness.py
Input    : data/processed/master_panel_t1_advanced.csv, data/raw/master_panel.csv
Output   : data/processed/master_panel_t1_missingness.csv
"""
import pandas as pd
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_advanced.csv"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "master_panel.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_panel_t1_missingness.csv"

def main():
    print("STEP 10: Missingness-Aware Feature Generation")
    
    df = pd.read_csv(INPUT_PATH)
    raw_df = pd.read_csv(RAW_PATH, low_memory=False)
    
    # We need to map raw missingness to the advanced dataframe
    # Merge the raw columns we care about, with a suffix to check missingness
    macro_indicators = [
        "inflation_cpi_pct",
        "unemployment_pct",
        "fdi_net_inflow_usd",
        "exports_pct_gdp",
        "imports_pct_gdp",
        "exchange_rate_lcu_usd",
        "interest_rate_pct",
        "reserves_usd",
        "current_account_pct_gdp",
        "remittances_usd",
        "tariff_rate_pct",
        "tax_revenue_pct_gdp"
    ]
    
    raw_sub = raw_df[["country_code", "year"] + macro_indicators].copy()
    
    # 1. Create binary missingness indicators from RAW DATA
    indicator_cols = []
    for col in macro_indicators:
        ind_col = f"{col}_missing"
        raw_sub[ind_col] = raw_sub[col].isna().astype(int)
        indicator_cols.append(ind_col)
        
    # Create aggregate features on RAW DATA
    raw_sub["total_missing_feature_count"] = raw_sub[macro_indicators].isna().sum(axis=1)
    raw_sub["total_missing_feature_ratio"] = raw_sub["total_missing_feature_count"] / len(macro_indicators)
    
    # Merge only the newly created indicators and aggregates into df
    cols_to_merge = ["country_code", "year", "total_missing_feature_count", "total_missing_feature_ratio"] + indicator_cols
    
    # Clean NRU in raw_sub to match df if needed? The data_preprocessing.py handles duplicates for NRU.
    # To be safe, we just merge on country_code and year. Since data_preprocessing dropped duplicates, 
    # merging left from df will correctly attach the missingness (it might match multiple raw, so we drop duplicates on raw_sub first)
    raw_sub = raw_sub.drop_duplicates(subset=["country_code", "year"])
    
    df = df.merge(raw_sub[cols_to_merge], on=["country_code", "year"], how="left")
    
    # Analyze Missingness Rates by Period (from the raw missingness we just merged)
    print("\n--- Missingness Rates by Period ---")
    
    # Filter only valid rows (except 2025 inference)
    valid_mask = (df["next_year_target_available"] == 1) | (df["year"] == 2025)
    valid_df = df[valid_mask]
    
    periods = {
        "TRAIN (2000-2018)": (2000, 2018),
        "VALIDATION (2019-2022)": (2019, 2022),
        "TEST (2023-2024)": (2023, 2024),
        "INFERENCE (2025)": (2025, 2025)
    }
    
    for period_name, (start_y, end_y) in periods.items():
        subset = valid_df[(valid_df["year"] >= start_y) & (valid_df["year"] <= end_y)]
        
        print(f"\n[{period_name}] - Rows: {len(subset)}")
        if len(subset) == 0:
            continue
            
        print("Missing % for Key Features (Raw Original):")
        for col in macro_indicators:
            ind_col = f"{col}_missing"
            miss_pct = subset[ind_col].mean() * 100
            print(f"  {col}: {miss_pct:.1f}%")
            
        avg_missing = subset["total_missing_feature_count"].mean()
        med_missing = subset["total_missing_feature_count"].median()
        max_missing = subset["total_missing_feature_count"].max()
        
        print(f"Row-level Stats - Avg: {avg_missing:.2f}, Median: {med_missing:.1f}, Max: {max_missing}")

    # Output dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved dataset to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
