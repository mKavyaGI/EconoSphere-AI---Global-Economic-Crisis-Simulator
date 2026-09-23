import os
import sys
import pandas as pd
import numpy as np
import io
import warnings
from pathlib import Path

try:
    import pandas_datareader.wb as wb
except ImportError:
    wb = None

try:
    import yfinance as yf
except ImportError:
    yf = None

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTAL_DATA_DIR = SCRIPT_DIR / "experimental_data"
EXPERIMENTAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = SCRIPT_DIR.parents[3] / "docs" / "model_accuracy" / "phase17_step2"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 10 Frontend countries
WB_COUNTRIES = ["US", "CN", "DE", "JP", "IN", "GB", "BR", "FR", "CA", "AU"]

def fetch_wdi_cpi(countries, start=2000, end=2024):
    """Fetch WDI Inflation/CPI (Annual) as a proof-of-concept for uncredentialed WDI."""
    if wb is None:
        return pd.DataFrame()
    try:
        df = wb.download(indicator="FP.CPI.TOTL.ZG", country=countries, start=start, end=end)
        return df
    except Exception as e:
        print(f"WDI fetch failed: {e}")
        return pd.DataFrame()

import urllib.request
import urllib.error

def fetch_bis_policy_rates():
    """Fetch BIS Central Bank Policy Rates (Monthly). BIS has a public SDMX API."""
    url = "https://data.bis.org/api/v1/data/BIS,WS_CBPOL,1.0/M.D.GB+US+JP+EU+BR+CA+AU+IN+CN.T?format=csv"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            csv_data = response.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        return df
    except Exception as e:
        print(f"BIS fetch failed: {e}")
        return pd.DataFrame()

def fetch_yahoo_fx(tickers, start="2000-01-01", end="2024-12-31"):
    """Fetch Yahoo Finance FX data."""
    if yf is None:
        return pd.DataFrame()
    try:
        # download quietly
        df = yf.download(tickers, start=start, end=end, progress=False)
        return df
    except Exception as e:
        print(f"Yahoo Finance fetch failed: {e}")
        return pd.DataFrame()

def main():
    print("================================================================")
    print("PHASE 17 STEP 2: Credential-Free ETL Execution")
    print("================================================================")
    
    feasibility_md = "# Phase 17 Step 2: Credential-Free ETL Feasibility Audit\n\n"
    
    # 1. World Bank CPI
    print("Attempting to fetch WDI CPI...")
    wdi_df = fetch_wdi_cpi(WB_COUNTRIES)
    if not wdi_df.empty:
        wdi_path = EXPERIMENTAL_DATA_DIR / "wdi_cpi_annual.csv"
        wdi_df.to_csv(wdi_path)
        print(f"  -> SUCCESS: Saved to {wdi_path} ({len(wdi_df)} rows)")
        feasibility_md += "## World Bank WDI (CPI)\n- **Status**: SUCCESS\n- **Details**: Successfully downloaded without API keys via `pandas_datareader.wb`. It provides reliable annual macro coverage.\n\n"
    else:
        print("  -> FAILED")
        feasibility_md += "## World Bank WDI (CPI)\n- **Status**: FAILED\n- **Details**: Could not download data. Possibly due to network restrictions or missing dependency.\n\n"
        
    # 2. BIS Policy Rates
    print("Attempting to fetch BIS Policy Rates...")
    bis_df = fetch_bis_policy_rates()
    if not bis_df.empty:
        bis_path = EXPERIMENTAL_DATA_DIR / "bis_policy_rates.csv"
        bis_df.to_csv(bis_path)
        print(f"  -> SUCCESS: Saved to {bis_path} ({len(bis_df)} rows)")
        feasibility_md += "## BIS Policy Rates\n- **Status**: SUCCESS\n- **Details**: Successfully accessed the public SDMX API for Central Bank policy rates.\n\n"
    else:
        print("  -> FAILED")
        feasibility_md += "## BIS Policy Rates\n- **Status**: FAILED\n- **Details**: Failed to hit the BIS SDMX endpoint.\n\n"
        
    # 3. Yahoo Finance FX
    print("Attempting to fetch Yahoo Finance FX...")
    tickers = ["GBPUSD=X", "BRLUSD=X", "EURUSD=X", "CADUSD=X", "AUDUSD=X"]
    yf_df = fetch_yahoo_fx(tickers)
    if not yf_df.empty:
        yf_path = EXPERIMENTAL_DATA_DIR / "yahoo_fx.csv"
        yf_df.to_csv(yf_path)
        print(f"  -> SUCCESS: Saved to {yf_path} ({len(yf_df)} rows)")
        feasibility_md += "## Yahoo Finance (FX)\n- **Status**: SUCCESS\n- **Details**: Successfully downloaded high-frequency daily FX rates via `yfinance` without credentials.\n\n"
    else:
        print("  -> FAILED")
        feasibility_md += "## Yahoo Finance (FX)\n- **Status**: FAILED\n- **Details**: Rate-limited or blocked.\n\n"
        
    report_path = REPORT_DIR / "phase17_step2_etl_feasibility.md"
    report_path.write_text(feasibility_md, encoding='utf-8')
    print(f"\n[SUCCESS] Phase 17 Step 2 ETL checks completed. Report saved to {report_path.name}")

if __name__ == "__main__":
    main()
