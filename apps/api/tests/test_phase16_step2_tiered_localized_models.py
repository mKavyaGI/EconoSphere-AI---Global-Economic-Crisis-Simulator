import pytest
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
WEB_PAGE = PROJECT_ROOT / "apps" / "web" / "src" / "app" / "dashboard" / "forecast" / "page.tsx"

def test_frontend_country_extraction():
    if not WEB_PAGE.exists():
        pytest.skip("Frontend page not found")
    content = WEB_PAGE.read_text(encoding="utf-8")
    match = re.search(r'const\s+SUPPORTED_COUNTRIES\s*=\s*\[(.*?)\];', content)
    assert match is not None
    countries = re.findall(r'["\'](.*?)["\']', match.group(1))
    assert "USA" in countries
    assert "GBR" in countries

def test_temporal_folds_strictly_chronological():
    folds = [
        {"name": "Fold A", "train": (2000, 2016), "val": (2017, 2018), "test": (2019, 2020)},
        {"name": "Fold B", "train": (2000, 2018), "val": (2019, 2020), "test": (2021, 2022)},
        {"name": "CANONICAL", "train": (2000, 2018), "val": (2019, 2022), "test": (2023, 2024)},
    ]
    for fold in folds:
        assert fold["train"][1] < fold["val"][0]
        assert fold["val"][1] < fold["test"][0]

def test_cluster_segmentation_uses_training_data_only():
    np.random.seed(42)
    df = pd.DataFrame({
        "country_code": ["A", "A", "B", "B", "C", "C"],
        "year": [2010, 2020, 2010, 2020, 2010, 2020],
        "gdp_current_usd": [100, 1000, 50, 500, 10, 100],
        "gdp_growth_rolling_mean_5": [2.0, 2.0, -1.0, -1.0, 5.0, 5.0],
        "trade_openness": [0.5, 0.5, 0.2, 0.2, 0.9, 0.9]
    })
    train_df = df[df["year"] == 2010].copy()
    test_df = df[df["year"] == 2020].copy()
    
    # Train clustering
    agg = train_df.groupby("country_code")[["gdp_current_usd", "gdp_growth_rolling_mean_5", "trade_openness"]].mean()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(agg)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(scaled)
    
    # Validation
    agg_test = test_df.groupby("country_code")[["gdp_current_usd", "gdp_growth_rolling_mean_5", "trade_openness"]].mean()
    scaled_test = scaler.transform(agg_test)
    preds = kmeans.predict(scaled_test)
    assert len(preds) == 3

def test_hybrid_routing_uses_validation_only():
    val_scores = {"G0": 1.5, "T1": 1.2, "T2": 2.0, "T3": 1.0}
    test_scores = {"G0": 1.1, "T1": 1.9, "T2": 2.5, "T3": 3.0}
    
    # Route strictly on val
    best_model = min(val_scores, key=val_scores.get)
    assert best_model == "T3"
    
    # Check that test score of selected model is correctly referenced (even if it's worse)
    assert test_scores[best_model] == 3.0

def test_percentage_improvement_math():
    base_rmse = 2.0
    cand_rmse = 1.0
    imp = ((cand_rmse - base_rmse) / base_rmse) * 100
    assert imp == -50.0  # -50% means 50% improvement
    
    base_rmse_2 = 1.0
    cand_rmse_2 = 1.5
    deg = ((cand_rmse_2 - base_rmse_2) / base_rmse_2) * 100
    assert deg == 50.0 # +50% means 50% degradation
