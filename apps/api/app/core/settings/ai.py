from pydantic_settings import BaseSettings
from typing import List

class AISettings(BaseSettings):
    """
    Centralized configuration settings for the AI Forecasting & Decision Intelligence module.
    Ensures zero hardcoded hyperparameters or system paths across ML pipelines.
    """
    # General AI Configuration
    ai_model_registry_path: str = "data/model_registry"
    ai_default_model_version: str = "1.0.0-PROD"
    ai_enable_explainability: bool = True
    ai_cache_ttl_seconds: int = 3600  # 1 Hour Redis caching for expensive forecasts
    
    # Forecasting Horizons (Days equivalent for modeling: 3M, 6M, 1Y, 3Y, 5Y)
    ai_forecast_horizons_days: List[int] = [90, 180, 365, 1095, 1825]
    ai_confidence_level_default: float = 0.95
    ai_monte_carlo_simulations: int = 500
    
    # Risk Assessment Engine Thresholds (0.0 - 100.0 Scale)
    ai_risk_threshold_low: float = 25.0
    ai_risk_threshold_medium: float = 50.0
    ai_risk_threshold_high: float = 75.0
    
    # Anomaly Detection & Z-score sensitivity
    ai_anomaly_zscore_threshold: float = 3.2
    ai_isolation_forest_contamination: float = 0.05
    
    # External Institutional Data Connectors Configuration
    ai_world_bank_api_base_url: str = "https://api.worldbank.org/v2"
    ai_fred_api_base_url: str = "https://api.stlouisfed.org/fred/series/observations"
    ai_fred_api_key: str = ""  # Optional API key override from environment
    ai_imf_sdmx_base_url: str = "https://www.imf.org/external/datamapper/api/v1"
    ai_oecd_api_base_url: str = "https://stats.oecd.org/SDMX-JSON/data"
    ai_comtrade_api_base_url: str = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
    
    # Model Hyperparameters defaults (avoiding hardcoded parameters in training)
    ai_xgboost_default_learning_rate: float = 0.05
    ai_xgboost_default_max_depth: int = 6
    ai_xgboost_default_n_estimators: int = 150
    ai_lstm_hidden_size: int = 64
    ai_transformer_num_heads: int = 4
    ai_gnn_embedding_dim: int = 32
