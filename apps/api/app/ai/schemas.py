"""
Universal schemas and API data transfer objects for the AI Forecasting & Decision Intelligence module.

Enforces strict compliance with Requirement 3: Every prediction must include prediction value,
confidence interval, confidence score, timestamp, model version, feature importance, and human-readable explanation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ForecastHorizon(str, Enum):
    MONTHS_3 = "3_months"
    MONTHS_6 = "6_months"
    YEAR_1 = "1_year"
    YEARS_3 = "3_years"
    YEARS_5 = "5_years"

class RiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AnomalySeverity(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    CRITICAL = "CRITICAL"

class PolicyImpactDirection(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

# --- Explainability & Attribution Schemas ---

class FeatureAttribution(BaseModel):
    feature_name: str = Field(..., description="Name of the econometric, lag, or graph topology feature")
    importance_score: float = Field(..., description="Normalized Shapley value or attention weight")
    direction: str = Field(..., description="POSITIVE or NEGATIVE contribution towards the predicted delta")
    human_label: Optional[str] = Field(None, description="User-friendly description of the indicator")

class PredictionExplainability(BaseModel):
    top_influencing_features: List[FeatureAttribution] = Field(..., description="Ordered list of primary prediction drivers")
    human_readable_explanation: str = Field(..., description="Executive Natural Language Synthesis explaining why this forecast occurred")
    model_confidence_rationale: Optional[str] = Field(None, description="Explanation of uncertainty factors")

# --- Forecasting Schemas ---

class ConfidenceInterval(BaseModel):
    lower_bound_p10: float = Field(..., description="10th percentile pessimistic/optimistic lower boundary")
    median_p50: float = Field(..., description="50th percentile median expectation")
    upper_bound_p90: float = Field(..., description="90th percentile pessimistic/optimistic upper boundary")

class IndicatorPrediction(BaseModel):
    """
    Standardized predictive output contract satisfying Requirement 3.
    Every inference result must instantiate this schema regardless of the underlying ML architecture.
    """
    indicator_name: str = Field(..., description="Target macroeconomic indicator (e.g., GDP Growth, Inflation Rate)")
    iso3: str = Field(..., description="Target ISO3 country code or GLOBAL")
    horizon: ForecastHorizon = Field(..., description="Temporal prediction range")
    prediction_value: float = Field(..., description="Point predictive value (median expectation)")
    confidence_interval: ConfidenceInterval = Field(..., description="Uncertainty quantification bounds (P10-P90)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model certainty metric between 0.0 and 1.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="ISO-8601 timestamp of model inference")
    model_version: str = Field(..., description="Semantic version tag of the generating ML architecture in Model Registry")
    feature_importance: Dict[str, float] = Field(..., description="Raw numerical attribution map of predictor features")
    human_readable_explanation: str = Field(..., description="Natural language executive summary justifying the prediction")

class CountryForecastResponse(BaseModel):
    iso3: str = Field(..., description="ISO3 code of the target sovereign entity")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    active_model_version: str = Field(...)
    predictions: List[IndicatorPrediction] = Field(..., description="List of multi-horizon forecasts across all 8 core indicators")
    overall_economic_trajectory: str = Field(..., description="e.g., EXPANDING, CONTRACTING, STABILIZING, STAGFLATIONARY")

# --- AI Recommendation Schemas ---

class SimulatedTradeoff(BaseModel):
    indicator: str = Field(..., description="Affected macroeconomic KPI")
    horizon: ForecastHorizon = Field(..., description="Time frame of projected tradeoff")
    delta_percentage: float = Field(..., description="Numerical percentage shift expected from policy implementation")
    impact_direction: PolicyImpactDirection = Field(...)
    rationale: str = Field(..., description="Economic mechanism driving this secondary tradeoff")

class PolicyRecommendation(BaseModel):
    recommendation_id: str = Field(..., description="Unique alphanumeric identifier for tracking policy action")
    action_title: str = Field(..., description="Concise prescriptive directive (e.g., 'Diversify Critical Mineral Imports')")
    policy_type: str = Field(..., description="Category tag: MONETARY_FISCAL, TRADE_TARIFF, SUPPLY_RESILIENCE")
    target_iso3: str = Field(...)
    reason: str = Field(..., description="Data & graph-driven diagnostic reason triggering this policy suggestion")
    expected_benefit: str = Field(..., description="Quantified structural improvement anticipated from execution")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="AI confidence in policy efficacy")
    risk_level: RiskTier = Field(..., description="Risk level associated with implementing this recommendation")
    affected_countries: List[str] = Field(..., description="ISO3 codes of international trade partners directly impacted")
    affected_sectors: List[str] = Field(..., description="Industrial sectors experiencing primary policy ripple effects")
    simulated_tradeoffs: List[SimulatedTradeoff] = Field(default_factory=list, description="Explicit simulated consequences")

# --- Risk Engine Schemas ---

class RiskDimensionScore(BaseModel):
    dimension_name: str = Field(..., description="Name of risk factor (e.g., Supply Chain Risk, Inflation Risk)")
    score: float = Field(..., ge=0.0, le=100.0, description="Normalized score on 0.0 (safest) to 100.0 (most hazardous) scale")
    tier: RiskTier = Field(...)
    primary_contributors: List[str] = Field(default_factory=list, description="Specific metrics driving this risk evaluation")

class CountryRiskScorecard(BaseModel):
    iso3: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_risk_score: float = Field(..., ge=0.0, le=100.0, description="Weighted rollup of all 7 structural risk dimensions")
    overall_risk_tier: RiskTier = Field(...)
    dimensions: Dict[str, RiskDimensionScore] = Field(..., description="Dictionary mapping dimension names to score objects")

# --- Anomaly Detection Schemas ---

class AnomalyAlert(BaseModel):
    anomaly_id: str = Field(..., description="Unique identifier for detected structural anomaly")
    detected_anomaly: str = Field(..., description="Type label (e.g., RAPID_INFLATION, SUDDEN_GDP_COLLAPSE, TRADE_DISRUPTION)")
    severity: AnomalySeverity = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    affected_countries: List[str] = Field(..., description="Entities directly involved in the abnormal deviation")
    possible_causes: List[str] = Field(..., description="Neo4j graph root-cause path tracing and macroeconomic attribution")
    anomaly_score_z: float = Field(..., description="Z-score or Isolation Forest deviation indicator")

# --- Model Registry Schemas ---

class ModelMetadata(BaseModel):
    model_id: str = Field(..., description="Unique model registry key (e.g., 'xgb-gdp-v1.0')")
    model_type: str = Field(..., description="Architecture class (XGBoost, LightGBM, LSTM, Transformer, GNN, RL)")
    version: str = Field(..., description="Semantic version tag")
    trained_at: datetime = Field(default_factory=datetime.utcnow)
    evaluation_metrics: Dict[str, float] = Field(default_factory=dict, description="RMSE, MAE, R², MAPE from k-fold validation")
    deployment_status: str = Field(default="STAGING", description="STAGING, SHADOW, PRODUCTION, DEPRECATED")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    feature_list: List[str] = Field(default_factory=list)
