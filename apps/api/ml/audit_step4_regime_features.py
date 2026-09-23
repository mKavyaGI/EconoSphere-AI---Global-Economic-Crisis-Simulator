import pandas as pd
import numpy as np
import os

def run_audit():
    print("Starting Phase 12 Step 4 Regime Feature Audit...")
    
    raw_path = "data/raw/master_panel.csv"
    df = pd.read_csv(raw_path)
    
    # We will use features constructed during Phase 11/12 that are known safe.
    # We document the rationale for their safety as regime classifiers.
    
    audit_results = [
        {
            "Feature": "gdp_growth_lag1",
            "Source": "gdp_growth_pct shifted 1 yr",
            "Temporal_Availability": "End of Year t",
            "Safe_for_Regime": True,
            "Target_Leakage": False,
            "Rationale": "Historical GDP growth for year t is known prior to forecasting target year t+1."
        },
        {
            "Feature": "gdp_growth_rolling_mean_3",
            "Source": "gdp_growth_pct rolling 3 yr",
            "Temporal_Availability": "End of Year t",
            "Safe_for_Regime": True,
            "Target_Leakage": False,
            "Rationale": "Rolling mean strictly over past 3 years (up to year t)."
        },
        {
            "Feature": "gdp_growth_rolling_std_3",
            "Source": "gdp_growth_pct rolling 3 yr std",
            "Temporal_Availability": "End of Year t",
            "Safe_for_Regime": True,
            "Target_Leakage": False,
            "Rationale": "Measures historical GDP volatility up to year t. Excellent for detecting 'Shock/Stress' regimes."
        },
        {
            "Feature": "unemployment_deterioration",
            "Source": "unemployment_pct change (t vs t-1)",
            "Temporal_Availability": "Early Year t+1",
            "Safe_for_Regime": True,
            "Target_Leakage": False,
            "Rationale": "Unemployment data for year t is safe to use as a stress indicator for target year t+1."
        },
        {
            "Feature": "inflation_change",
            "Source": "inflation_cpi_pct change (t vs t-1)",
            "Temporal_Availability": "Early Year t+1",
            "Safe_for_Regime": True,
            "Target_Leakage": False,
            "Rationale": "Inflation change captures macro stress prior to the target year."
        },
        {
            "Feature": "gdp_growth_next_year",
            "Source": "gdp_growth_pct shifted -1 yr",
            "Temporal_Availability": "End of Year t+1",
            "Safe_for_Regime": False,
            "Target_Leakage": True,
            "Rationale": "This is the target variable itself. Using it to define the regime is forbidden leakage."
        }
    ]
    
    audit_df = pd.DataFrame(audit_results)
    
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase12_step4_regime_feature_audit.csv")
    audit_df.to_csv(out_path, index=False)
    
    print(f"Audit complete. Results saved to {out_path}")
    print("\nAudit Summary:")
    print(audit_df.to_string())

if __name__ == "__main__":
    run_audit()
