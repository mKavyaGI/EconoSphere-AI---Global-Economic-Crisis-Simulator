"""
Comprehensive Unit Tests for Phase 10: AI Forecasting & Decision Intelligence.
"""
import pytest
import asyncio
from datetime import datetime, timezone
from app.ai.config import ai_config
from app.ai.schemas import ForecastHorizon, RiskTier
from app.ai.ingestion.validation import DataValidator
from app.ai.ingestion.adapters.connectors import ConnectorFactory
from app.ai.feature_engineering.lag_generators import LagGenerator
from app.ai.feature_engineering.pipeline import FeaturePipeline
from app.ai.models.factory import ModelFactory
from app.ai.explainability.engine import ExplainabilityEngine
from app.ai.risk_assessment.engine import RiskAssessmentEngine
from app.ai.recommendations.engine import RecommendationEngine
from app.ai.anomaly_detection.detector import AnomalyDetectorEngine
from app.ai.registry.manager import model_registry
from app.ai.registry.metrics import ModelEvaluator

@pytest.mark.asyncio
async def test_data_validator_outlier_clipping():
    """Test DataValidator statistical outlier clipping and linear missing value imputation."""
    raw = [
        {"year": 2020, "value": 2.5, "series_id": "NY.GDP", "iso3": "USA"},
        {"year": 2021, "value": None, "series_id": "NY.GDP", "iso3": "USA"}, # Missing item to impute
        {"year": 2022, "value": 2.9, "series_id": "NY.GDP", "iso3": "USA"},
        {"year": 2023, "value": 150.0, "series_id": "NY.GDP", "iso3": "USA"}, # Extreme outlier (>5 sigma)
        {"year": 2024, "value": 2.6, "series_id": "NY.GDP", "iso3": "USA"},
    ]
    cleaned = DataValidator.clean_series_records(raw, z_threshold=2.5)
    assert len(cleaned) == 5
    # Check interpolation worked for 2021
    val_2021 = next(r["value"] for r in cleaned if r["year"] == 2021)
    assert val_2021 == 2.5  # Forward filled from 2020
    # Check outlier was clipped for 2023
    val_2023 = next(r["value"] for r in cleaned if r["year"] == 2023)
    assert val_2023 < 50.0

@pytest.mark.asyncio
async def test_institutional_connector_factory():
    """Test reusable dataset ingestion connectors for World Bank, IMF, UN Comtrade, OECD, and FRED."""
    for inst in ["WORLD_BANK", "IMF", "COMTRADE", "OECD", "FRED"]:
        conn = ConnectorFactory.get_connector(inst)
        assert conn is not None
        series = await conn.fetch_series("NY.GDP.MKTP.KD.ZG", "CAN", start_year=2015, end_year=2020)
        assert len(series) == 6
        assert all("value" in r for r in series)

@pytest.mark.asyncio
async def test_lag_generator_and_feature_pipeline():
    """Test lag mathematical variables and complete Feature Pipeline graph topology synthesis."""
    series = [2.1, 2.3, 2.4, 2.2, 2.6, 2.8, 3.1, 3.0, 2.9, 2.7, 2.5, 2.4, 2.6]
    lags = LagGenerator.generate_lags(series, [1, 3, 6, 12])
    assert "lag_1" in lags
    assert lags["lag_1"] == 2.4
    
    pipeline = FeaturePipeline()
    historical_series = {"GDP Growth": series, "Inflation Rate": [2.0, 2.2, 2.5]}
    vector = await pipeline.build_feature_vector("IND", historical_series)
    
    assert "gdp_growth_current" in vector
    assert "derived_misery_index" in vector
    assert "graph_betweenness_centrality" in vector

@pytest.mark.asyncio
async def test_model_factory_and_universal_prediction_contract():
    """Verify all algorithm architectures produce compliant prediction schemas satisfying Requirement 3."""
    pipeline = FeaturePipeline()
    features = await pipeline.build_feature_vector("JPN", {"GDP Growth": [1.5, 1.8], "Inflation Rate": [0.8, 1.2]})
    
    for model_type in ["XGBOOST", "TRANSFORMER", "GNN", "LIGHTGBM", "VAR", "RL"]:
        model = ModelFactory.get_model(model_type)
        pred = await model.predict("JPN", "GDP Growth", ForecastHorizon.YEAR_1, features)
        
        # Enforce Requirement 3 Mandatory Fields
        assert pred.prediction_value is not None
        assert pred.confidence_interval is not None
        assert pred.confidence_interval.lower_bound_p10 <= pred.confidence_interval.median_p50 <= pred.confidence_interval.upper_bound_p90
        assert 0.0 <= pred.confidence_score <= 1.0
        assert isinstance(pred.timestamp, datetime)
        assert isinstance(pred.model_version, str) and len(pred.model_version) > 0
        assert isinstance(pred.feature_importance, dict) and len(pred.feature_importance) > 0
        assert isinstance(pred.human_readable_explanation, str) and len(pred.human_readable_explanation) > 10

@pytest.mark.asyncio
async def test_risk_assessment_engine():
    """Verify sovereign structural Risk Engine 0-100 scorecard calculation across 7 dimensions."""
    engine = RiskAssessmentEngine()
    ind_values = {
        "GDP Growth": -1.5,
        "Inflation Rate": 9.5,
        "Unemployment Rate": 7.8,
        "Government Debt-to-GDP": 92.0,
        "Interest Rate": 6.5
    }
    scorecard = await engine.evaluate_country_risk("GBR", ind_values)
    
    assert scorecard.iso3 == "GBR"
    assert 0.0 <= scorecard.overall_risk_score <= 100.0
    assert len(scorecard.dimensions) == 7
    assert "Inflation Risk" in scorecard.dimensions
    assert "Supply Chain Risk" in scorecard.dimensions
    assert scorecard.dimensions["Inflation Risk"].tier in [RiskTier.HIGH, RiskTier.CRITICAL]

@pytest.mark.asyncio
async def test_policy_recommendations_and_simulated_tradeoffs():
    """Verify AI Recommendation Engine prescribes intervention directives with explicit secondary tradeoffs."""
    risk_engine = RiskAssessmentEngine()
    scorecard = await risk_engine.evaluate_country_risk("DEU", {"Inflation Rate": 8.5, "GDP Growth": 1.2})
    
    rec_engine = RecommendationEngine()
    recs = await rec_engine.generate_recommendations("DEU", scorecard)
    
    assert len(recs) >= 1
    for r in recs:
        assert r.target_iso3 == "DEU"
        assert len(r.action_title) > 0
        assert len(r.expected_benefit) > 0
        assert len(r.affected_countries) > 0
        assert isinstance(r.simulated_tradeoffs, list)

@pytest.mark.asyncio
async def test_anomaly_detection_engine():
    """Verify anomaly detector alerts on severe statistical Z-score breaks."""
    detector = AnomalyDetectorEngine()
    global_map = {
        "USA": {"Inflation Rate": 3.1, "GDP Growth": 2.4},
        "DEU": {"Inflation Rate": 2.8, "GDP Growth": 1.5},
        "ARG": {"Inflation Rate": 145.0, "GDP Growth": -6.2}, # Extreme inflationary anomaly
    }
    alerts = await detector.scan_for_anomalies(global_map)
    
    arg_alert = next((a for a in alerts if "ARG" in a.affected_countries), None)
    assert arg_alert is not None
    assert "INFLATION" in arg_alert.detected_anomaly.upper()
    assert arg_alert.severity.value == "CRITICAL"
    assert len(arg_alert.possible_causes) >= 2

def test_model_registry_and_evaluation_metrics():
    """Verify Model Evaluator KPI calculations and Model Registry inference logging lineage."""
    actuals = [2.5, 3.0, 2.8, 3.2, 2.9]
    preds = [2.45, 2.95, 2.85, 3.15, 2.9]
    kpis = ModelEvaluator.calculate_metrics(actuals, preds)
    
    assert kpis["rmse"] < 0.2
    assert kpis["r2_score"] > 0.80
    
    model_registry.log_inference("FRA", "GDP Growth", "xgb-macro-v1.0.0-PROD", 2.65, 0.94)
    history = model_registry.get_inference_history()
    assert len(history) > 0
    assert history[0]["iso3"] == "FRA"
