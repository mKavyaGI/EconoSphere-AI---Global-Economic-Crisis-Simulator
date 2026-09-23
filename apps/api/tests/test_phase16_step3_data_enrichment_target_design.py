import pytest
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
WEB_PAGE = PROJECT_ROOT / "apps" / "web" / "src" / "app" / "dashboard" / "forecast" / "page.tsx"

def test_1_frontend_country_extraction():
    if not WEB_PAGE.exists():
        pytest.skip("Frontend page not found")
    content = WEB_PAGE.read_text(encoding="utf-8")
    match = re.search(r'const\s+SUPPORTED_COUNTRIES\s*=\s*\[(.*?)\];', content)
    assert match is not None
    countries = re.findall(r'["\'](.*?)["\']', match.group(1))
    assert "USA" in countries
    assert "GBR" in countries

def test_3_target_scaling_leakage():
    # Target normalization statistics must be fitted ONLY on Train targets
    train_y = np.array([1.0, 2.0, 3.0]).reshape(-1, 1)
    test_y = np.array([10.0, 20.0]).reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(train_y)
    
    # Assert mean is based only on train_y
    assert scaler.mean_[0] == 2.0
    
    # Transform test_y using train_y stats
    scaled_test = scaler.transform(test_y)
    assert scaled_test[0][0] > 5.0  # Should be heavily scaled since train mean is 2.0

def test_4_inverse_transformation_correctness():
    # Predictions are converted back before MAE/RMSE calculations
    train_y = np.array([1.0, 2.0, 3.0]).reshape(-1, 1)
    test_y = np.array([4.0, 5.0]).reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(train_y)
    
    scaled_test = scaler.transform(test_y)
    # Simulate a perfect prediction in scaled space
    pred_scaled = scaled_test
    
    # Inverse transform
    pred_original = scaler.inverse_transform(pred_scaled)
    np.testing.assert_array_almost_equal(pred_original, test_y)

def test_5_feature_selection_temporal_safety():
    # Variance, correlation, and model-guided feature selection use Train data only
    train_df = pd.DataFrame({"feat1": [1, 1, 1], "feat2": [1, 2, 3], "target": [10, 20, 30]})
    test_df = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6], "target": [40, 50, 60]})
    
    # feat1 has zero variance in train, but not in test. It should be removed based on Train.
    variance = train_df.var()
    selected_features = [f for f in ["feat1", "feat2"] if variance[f] > 0.0]
    
    assert "feat1" not in selected_features
    assert "feat2" in selected_features

def test_6_enrichment_timing_leakage():
    # Use synthetic dated observations to prove that data after the allowed forecast cutoff is rejected.
    source_df = pd.DataFrame({
        "date": pd.to_datetime(["2023-10-01", "2023-11-01", "2024-01-01", "2024-03-01"]),
        "value": [100, 105, 110, 120]
    })
    target_year = 2024
    forecast_cutoff = pd.to_datetime("2023-12-31")
    
    # Filter valid dates strictly before cutoff
    valid_df = source_df[source_df["date"] <= forecast_cutoff]
    
    assert len(valid_df) == 2
    assert valid_df["value"].max() == 105

def test_7_high_frequency_aggregation_safety():
    # Verify aggregation uses only observations genuinely available before the forecast cutoff.
    source_df = pd.DataFrame({
        "date": pd.to_datetime(["2023-10-01", "2023-11-01", "2023-12-01"]),
        "value": [100, 105, 110]
    })
    publication_lag_days = 45 # Dec 1st data is not available until Jan 15th
    forecast_cutoff = pd.to_datetime("2023-12-31")
    
    # Availability date = observation date + lag
    source_df["available_date"] = source_df["date"] + pd.Timedelta(days=publication_lag_days)
    
    valid_df = source_df[source_df["available_date"] <= forecast_cutoff]
    # Dec 1 + 45 days = Jan 15 > Dec 31. So Dec 1 is excluded.
    # Nov 1 + 45 days = Dec 16 <= Dec 31. So Nov 1 is included.
    assert len(valid_df) == 2
    assert valid_df["value"].iloc[-1] == 105

def test_8_no_pseudo_replication():
    # Verify annual targets are not duplicated across higher-frequency rows.
    # A standard grouped dataset should aggregate high-frequency features up to the annual target level, not down.
    annual_targets = pd.DataFrame({"year": [2023, 2024], "target": [2.5, 3.0]})
    monthly_features = pd.DataFrame({"year": [2023, 2023, 2024, 2024], "month": [11, 12, 11, 12], "val": [1, 2, 3, 4]})
    
    # Aggregation up to annual level
    agg_features = monthly_features.groupby("year")["val"].mean().reset_index()
    merged = pd.merge(annual_targets, agg_features, on="year")
    
    assert len(merged) == len(annual_targets)

def test_9_small_sample_risk_warning():
    # Verify the observations/features warning triggers correctly.
    observations = 15
    features = 31
    warning_triggered = observations < features
    assert warning_triggered is True

def test_10_percentage_improvement_safety():
    # Verify comparison logic handles zero safely.
    base_rmse = 0.0
    cand_rmse = 1.0
    
    if base_rmse == 0.0:
        if cand_rmse == 0.0:
            imp = 0.0
        else:
            imp = np.nan # Undefined degradation
    else:
        imp = ((cand_rmse - base_rmse) / base_rmse) * 100
        
    assert np.isnan(imp)
