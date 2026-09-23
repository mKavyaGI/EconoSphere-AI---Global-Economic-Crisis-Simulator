"""
EconoSphere AI — Phase 11 Step 13 Forecast Schema
===================================================
Pydantic v2 response models for the locked GDP T+1 forecast endpoint.

These schemas represent the official production forecast from:
  - Model: HistGradientBoostingRegressor (Step 10, locked)
  - Test RMSE: 3.9113
  - Uncertainty: Global Q90 residual-calibrated interval (Step 11/12, locked)
  - Q90: 11.4507 percentage points

IMPORTANT: These schemas describe MODEL FORECASTS, not actual observed GDP values.
The forecast_status field always equals "forecast" for forward-looking predictions.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class ForecastModelInfo(BaseModel):
    """Provenance information about the locked point-forecast model."""
    version: str = Field(description="Official model version identifier")
    algorithm: str = Field(description="ML algorithm class name")
    test_rmse: float = Field(description="Locked test RMSE (never re-estimated from test set)")
    train_period: str = Field(description="Chronological training window")
    validation_period: str = Field(description="Chronological validation window used for uncertainty calibration")
    test_period: str = Field(description="Chronological held-out test window")


class ForecastUncertaintyInfo(BaseModel):
    """Provenance information about the uncertainty interval method."""
    method: str = Field(description="Uncertainty estimation method identifier")
    method_description: str = Field(description="Human-readable description of the interval method")
    q90: float = Field(description="90th percentile of absolute out-of-sample validation residuals")
    coverage_target: float = Field(description="Nominal coverage target (0.90 = 90%)")
    step12_decision: str = Field(description="Official Step 12 uncertainty experiment outcome")


class GDPForecastResponse(BaseModel):
    """
    Official T+1 GDP growth forecast for a single country.

    Statistical disclaimer:
      - predicted_gdp_growth is the model's single best point estimate.
      - lower_bound_90 / upper_bound_90 form an empirical 90% prediction interval
        calibrated from historical out-of-sample residuals. This is NOT a guarantee.
      - forecast_status='forecast' indicates this is a model-generated projection,
        not an observed economic statistic.
    """
    country_code: str = Field(description="ISO 3166-1 alpha-3 country code")
    country_name: str = Field(description="Full country name")
    feature_year: int = Field(description="Year of input economic indicators used for prediction")
    forecast_year: int = Field(description="Year being forecast (target year)")
    predicted_gdp_growth: float = Field(description="Point estimate of GDP growth rate (percentage points)")
    lower_bound_90: float = Field(description="Lower bound of 90% prediction interval (percentage points)")
    upper_bound_90: float = Field(description="Upper bound of 90% prediction interval (percentage points)")
    interval_width: float = Field(description="Total width of the 90% prediction interval (percentage points)")
    forecast_status: str = Field(
        default="forecast",
        description="Always 'forecast' — indicates model-generated projection, not observed data"
    )
    model: ForecastModelInfo
    uncertainty: ForecastUncertaintyInfo


class ForecastMetadataResponse(BaseModel):
    """
    Complete model and artifact provenance metadata for the locked Phase 11 pipeline.
    Used for transparency, reproducibility, and academic/research reporting.
    """
    model_version: str
    algorithm: str
    test_rmse: float
    train_period: str
    validation_period: str
    test_period: str
    feature_year: int
    forecast_year: int
    uncertainty_method: str
    q90: float
    coverage_target: float
    step12_decision: str
    artifact_file: str = Field(description="Logical artifact name (no filesystem path exposed)")
    available_countries: List[str] = Field(description="ISO3 codes with available forecasts")
    n_countries: int


class ForecastListResponse(BaseModel):
    """Collection of all available country forecasts plus metadata."""
    forecasts: List[GDPForecastResponse]
    metadata: ForecastMetadataResponse
    disclaimer: str = Field(
        default=(
            "These are model-generated GDP growth forecasts, not observed economic statistics. "
            "The 90% prediction interval is calibrated from historical out-of-sample residuals "
            "and does not guarantee that actual GDP growth will fall within the stated range."
        )
    )

class InferenceRequest(BaseModel):
    """
    Explicit schema for the Phase 11 production inference endpoint.
    Only allows exactly the 31 known feature fields.
    """
    country: str = Field(description="Country identifier")
    year: int = Field(description="Target year for prediction")
    
    exchange_rate_lcu_usd: float
    tariff_rate_pct: float
    remittances_usd: float
    fdi_net_inflow_usd: float
    unemployment_pct: float
    imports_pct_gdp: float
    tax_revenue_pct_gdp: float
    exports_pct_gdp: float
    interest_rate_pct: float
    reserves_usd: float
    current_account_pct_gdp: float
    inflation_cpi_pct: float
    population_total: float
    gdp_current_usd: float
    gdp_growth_lag1: float
    gdp_growth_lag2: float
    gdp_growth_lag3: float
    inflation_lag1: float
    unemployment_lag1: float
    exports_lag1: float
    imports_lag1: float
    gdp_growth_rolling_mean_3: float
    gdp_growth_rolling_std_3: float
    gdp_growth_rolling_mean_5: float
    inflation_rolling_mean_3: float
    trade_openness: float
    trade_balance_ratio: float
    log_gdp_usd: float
    log_population: float
    gdp_growth_rolling_std_5: float
    inflation_rolling_std_3: float

    model_config = {
        "extra": "forbid"
    }

class InferenceResponse(BaseModel):
    """
    Structured production prediction response.
    """
    country: str
    year: int
    predicted_gdp_growth: float
    model_version: str
    production_status: str

class ModelIdentityResponse(BaseModel):
    """
    Lightweight identity endpoint response.
    """
    model_version: str
    status: str
    model_class: str
    feature_count: int
    validation_rmse: float
    test_rmse: float
    release_fingerprint: str

