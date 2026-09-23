import pytest
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

# Mock required constants
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
    assert len(countries) > 0
    assert "USA" in countries
    assert "GBR" in countries

def test_temporal_folds_chronological():
    folds = [
        {"name": "CANONICAL", "train": (2000, 2018), "val": (2019, 2022), "test": (2023, 2024)},
        {"name": "Fold A", "train": (2000, 2016), "val": (2017, 2018), "test": (2019, 2020)},
        {"name": "Fold B", "train": (2000, 2018), "val": (2019, 2020), "test": (2021, 2022)},
        {"name": "Fold C", "train": (2000, 2020), "val": (2021, 2022), "test": (2023, 2024)},
    ]
    for fold in folds:
        assert fold["train"][1] < fold["val"][0], f"Train overlaps Val in {fold['name']}"
        assert fold["val"][1] < fold["test"][0], f"Val overlaps Test in {fold['name']}"

def test_no_future_leakage_in_grouping():
    # Simulate country segmentation logic
    np.random.seed(42)
    df = pd.DataFrame({
        "country_code": ["A", "A", "B", "B", "C", "C"],
        "year": [2010, 2020, 2010, 2020, 2010, 2020],
        "gdp_current_usd": [100, 1000, 50, 500, 10, 100]
    })
    train_df = df[df["year"] == 2010]
    test_df = df[df["year"] == 2020]
    
    median_gdp = train_df.groupby("country_code")["gdp_current_usd"].median()
    q = median_gdp.quantile([0.33, 0.67])
    
    def assign(x):
        if x < q.iloc[0]: return 0
        if x < q.iloc[1]: return 1
        return 2
        
    segment_map = median_gdp.apply(assign).to_dict()
    df["segment"] = df["country_code"].map(segment_map)
    
    assert df[df["country_code"] == "C"]["segment"].iloc[0] == 0
    assert df[df["country_code"] == "B"]["segment"].iloc[0] == 1
    assert df[df["country_code"] == "A"]["segment"].iloc[0] == 2

def test_percentage_improvement_calculation():
    base_rmse = 2.0
    cand_rmse = 1.5
    improvement = ((base_rmse - cand_rmse) / base_rmse) * 100
    assert improvement == 25.0
    
    cand_rmse_worse = 2.5
    degradation = ((base_rmse - cand_rmse_worse) / base_rmse) * 100
    assert degradation == -25.0
