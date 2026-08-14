"""
Core Macroeconomic Forecasting Engine orchestrating time series feature engineering, model inference,
and XAI explanation synthesis across all 8 indicators and 5 temporal horizons.
"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.ai.models.factory import ModelFactory
from app.ai.feature_engineering.pipeline import FeaturePipeline
from app.ai.explainability.engine import ExplainabilityEngine
from app.ai.schemas import (
    CountryForecastResponse,
    IndicatorPrediction,
    ForecastHorizon
)
from app.ai.config import ai_config

class MacroForecastingEngine:
    """
    Production multi-horizon AI forecasting orchestrator.
    Integrates feature extraction, DI-enabled model inference, and SHAP explainability.
    """
    def __init__(self, model_type: str = "XGBOOST"):
        self.model = ModelFactory.get_model(model_type)
        self.feature_pipeline = FeaturePipeline()
        self.xai_engine = ExplainabilityEngine()

    async def generate_country_forecast(
        self,
        iso3: str,
        historical_series: Dict[str, List[float]],
        graph_metrics: Optional[Dict[str, float]] = None,
        simulation_shocks: Optional[Dict[str, float]] = None,
        horizons: Optional[List[ForecastHorizon]] = None
    ) -> CountryForecastResponse:
        """
        Executes end-to-end multi-horizon macroeconomic projections for a target sovereign entity.
        """
        if not horizons:
            horizons = [
                ForecastHorizon.MONTHS_3,
                ForecastHorizon.MONTHS_6,
                ForecastHorizon.YEAR_1,
                ForecastHorizon.YEARS_3,
                ForecastHorizon.YEARS_5
            ]

        # Step 1: Synthesize normalized historical observations, lags, graph topology, and shocks
        features = await self.feature_pipeline.build_feature_vector(
            iso3=iso3,
            historical_series=historical_series,
            graph_metrics=graph_metrics,
            simulation_shocks=simulation_shocks
        )

        predictions: List[IndicatorPrediction] = []
        indicators = ai_config.supported_indicators

        # Step 2: Perform iterative inference across all indicators and horizons
        for indicator in indicators:
            for hz in horizons:
                pred = await self.model.predict(
                    iso3=iso3,
                    indicator=indicator,
                    horizon=hz,
                    features=features
                )
                
                # Step 3: Enrich with Natural Language XAI synthesis
                explanation = self.xai_engine.explain_prediction(
                    indicator=indicator,
                    point_prediction=pred.prediction_value,
                    feature_importance_map=pred.feature_importance,
                    iso3=iso3
                )
                pred.human_readable_explanation = explanation
                predictions.append(pred)

        # Step 4: Evaluate macro trajectory classification
        trajectory = self._assess_trajectory(predictions)

        return CountryForecastResponse(
            iso3=iso3,
            generated_at=datetime.utcnow(),
            active_model_version=self.model.get_metadata().version,
            predictions=predictions,
            overall_economic_trajectory=trajectory
        )

    def _assess_trajectory(self, predictions: List[IndicatorPrediction]) -> str:
        """Determines aggregate structural trajectory based on 1-Year GDP and Inflation expectations."""
        gdp_1y = next((p.prediction_value for p in predictions if p.indicator_name == "GDP Growth" and p.horizon == ForecastHorizon.YEAR_1), 2.5)
        cpi_1y = next((p.prediction_value for p in predictions if p.indicator_name == "Inflation Rate" and p.horizon == ForecastHorizon.YEAR_1), 2.8)
        
        if gdp_1y < 0.5 and cpi_1y > 5.0:
            return "STAGFLATIONARY_WARNING"
        elif gdp_1y < 0.0:
            return "CONTRACTING_RECESSIONARY"
        elif gdp_1y > 3.5:
            return "EXPANDING_ROBUST"
        else:
            return "STABILIZING_MODERATE"
