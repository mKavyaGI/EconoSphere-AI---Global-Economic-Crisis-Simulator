import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# The tests will mock the external calls to avoid CI flakiness
from apps.api.ml.phase17.step2.credential_free_etl import (
    fetch_wdi_cpi,
    fetch_bis_policy_rates,
    fetch_yahoo_fx,
    WB_COUNTRIES
)

try:
    import pandas_datareader
    HAS_PANDAS_DATAREADER = True
except ImportError:
    HAS_PANDAS_DATAREADER = False

try:
    import yfinance
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

@pytest.mark.skipif(not HAS_PANDAS_DATAREADER, reason="pandas_datareader not installed")
def test_fetch_wdi_cpi_mocked():
    # Mock pandas_datareader.wb.download
    with patch("pandas_datareader.wb.download") as mock_wb:
        # Create a mock dataframe
        mock_df = pd.DataFrame({
            "FP.CPI.TOTL.ZG": [2.5, 3.1],
        }, index=pd.MultiIndex.from_tuples([("United States", "2020"), ("United States", "2021")], names=["country", "year"]))
        mock_wb.return_value = mock_df
        
        df = fetch_wdi_cpi(["USA"], start=2020, end=2021)
        
        assert mock_wb.called
        assert len(df) == 2
        assert "country" in df.index.names
        assert "year" in df.index.names
        assert "FP.CPI.TOTL.ZG" in df.columns

def test_fetch_bis_policy_rates_graceful_fail():
    # If requests fails, it should return an empty dataframe or handle gracefully
    with patch("urllib.request.urlopen") as mock_get:
        mock_get.side_effect = Exception("404 Not Found")
        
        df = fetch_bis_policy_rates()
        assert df.empty

@pytest.mark.skipif(not HAS_YFINANCE, reason="yfinance not installed")
def test_fetch_yahoo_fx_graceful_fail():
    with patch("yfinance.download") as mock_yf:
        mock_yf.side_effect = Exception("Rate limited")
        df = fetch_yahoo_fx(["GBPUSD=X"], start="2020-01-01", end="2020-01-31")
        assert df.empty
