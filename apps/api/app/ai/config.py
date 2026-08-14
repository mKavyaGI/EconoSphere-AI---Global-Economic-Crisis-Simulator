"""
Centralized configuration access for the AI Forecasting & Decision Intelligence Module.

All machine learning hyperparameter defaults, threshold classifications, lag sizes, and external connector
URLs are managed here via global application settings to ensure zero hardcoding across model wrappers.
"""
from app.core.config import settings
from typing import List, Dict, Any

class AIConfig:
    """
    Singleton access helper providing pre-formatted configuration schemas for AI pipelines.
    """
    @property
    def registry_path(self) -> str:
        return settings.ai_model_registry_path

    @property
    def default_version(self) -> str:
        return settings.ai_default_model_version

    @property
    def cache_ttl(self) -> int:
        return settings.ai_cache_ttl_seconds

    @property
    def anomaly_zscore_threshold(self) -> float:
        """Z-score standard deviation threshold for statistical anomaly classification."""
        return settings.ai_anomaly_zscore_threshold

    # Alias matching detector.py access pattern (C-2 Fix)
    @property
    def ai_anomaly_zscore_threshold(self) -> float:
        return settings.ai_anomaly_zscore_threshold

    @property
    def horizon_map(self) -> Dict[str, int]:
        """Maps canonical horizon labels to exact target calendar days."""
        return {
            "3_months": 90,
            "6_months": 180,
            "1_year": 365,
            "3_years": 1095,
            "5_years": 1825
        }

    @property
    def supported_indicators(self) -> List[str]:
        """Core macroeconomic indicators forecasted and tracked by the AI engine."""
        return [
            "GDP Growth",
            "Inflation Rate",
            "Unemployment Rate",
            "Interest Rate",
            "Exchange Rate",
            "Trade Volume",
            "Government Debt-to-GDP",
            "Currency Strength Index"
        ]

    @property
    def risk_dimensions(self) -> List[str]:
        """Seven distinct structural risk dimensions calculated by the Risk Engine."""
        return [
            "Country Risk Score",
            "Trade Risk Score",
            "Economic Stability Score",
            "Supply Chain Risk",
            "Financial Risk",
            "Inflation Risk",
            "Currency Risk"
        ]

    def classify_risk_tier(self, score: float) -> str:
        """Classifies a continuous risk score (0.0 to 100.0) into standardized ordinal tiers."""
        if score <= settings.ai_risk_threshold_low:
            return "LOW"
        elif score <= settings.ai_risk_threshold_medium:
            return "MEDIUM"
        elif score <= settings.ai_risk_threshold_high:
            return "HIGH"
        else:
            return "CRITICAL"

    def get_model_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Retrieves default tuned hyperparameter dictionaries for given model architectures."""
        params = {
            "xgboost": {
                "learning_rate": settings.ai_xgboost_default_learning_rate,
                "max_depth": settings.ai_xgboost_default_max_depth,
                "n_estimators": settings.ai_xgboost_default_n_estimators,
            },
            "lightgbm": {
                "learning_rate": 0.05,
                "num_leaves": 31,
                "n_estimators": 150,
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
            },
            "lstm": {
                "hidden_size": settings.ai_lstm_hidden_size,
                "num_layers": 2,
                "dropout": 0.1,
            },
            "transformer": {
                "num_heads": settings.ai_transformer_num_heads,
                "hidden_dim": 128,
                "num_layers": 3,
            },
            "gnn": {
                "embedding_dim": settings.ai_gnn_embedding_dim,
                "num_hops": 3,
            }
        }
        return params.get(model_type.lower(), {})

ai_config = AIConfig()
