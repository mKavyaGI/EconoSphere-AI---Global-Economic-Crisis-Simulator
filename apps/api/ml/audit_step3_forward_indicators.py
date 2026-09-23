import pandas as pd
import numpy as np
import os
import json

def run_audit():
    print("Starting Phase 12 Step 3 Data Availability Audit...")
    
    # 1. Load Data
    raw_path = "data/raw/master_panel.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    # 2. Define Candidate Variables
    candidates = [
        "inflation_cpi_pct",
        "interest_rate_pct",
        "unemployment_pct",
        "exports_pct_gdp",
        "imports_pct_gdp",
        "fdi_net_inflow_usd",
        "govt_debt_pct_gdp",
        "exchange_rate_lcu_usd",
        "reserves_usd"
    ]
    
    # 3. Analyze Availability
    audit_results = []
    
    for col in candidates:
        if col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            
            # Establish temporal safety assumption
            if col in ["exchange_rate_lcu_usd", "interest_rate_pct", "reserves_usd"]:
                safety = "A"
                reason = "High-frequency data; available immediately at end of year t."
            elif col in ["inflation_cpi_pct", "unemployment_pct", "exports_pct_gdp", "imports_pct_gdp"]:
                safety = "A"
                reason = "Macro data published with short lag; reliable estimates available early year t+1."
            elif col in ["fdi_net_inflow_usd", "govt_debt_pct_gdp"]:
                safety = "B"
                reason = "Often published with significant lag and subject to heavy revisions."
            else:
                safety = "C"
                reason = "Unknown temporal safety."
                
            audit_results.append({
                "Indicator": col,
                "Status": "Available",
                "Missing_Pct": round(missing_pct, 2),
                "Category": safety,
                "Temporal_Assumption": reason
            })
        else:
            audit_results.append({
                "Indicator": col,
                "Status": "Missing",
                "Missing_Pct": 100.0,
                "Category": "C",
                "Temporal_Assumption": "Not in dataset."
            })
            
    # Convert to DataFrame
    audit_df = pd.DataFrame(audit_results)
    
    # Save output
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase12_step3_indicator_audit.csv")
    audit_df.to_csv(out_path, index=False)
    
    print(f"Audit complete. Results saved to {out_path}")
    print("\nAudit Summary:")
    print(audit_df)

if __name__ == "__main__":
    run_audit()
