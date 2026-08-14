"""
Abstract Base Classes (ABCs) and Dependency Injection interfaces for the AI Layer.

Ensures strict decoupling between underlying ML implementations (XGBoost, LightGBM, Random Forest, LSTM, Transformers,
GNNs, RL) and application service/API layers. Models can be upgraded or hot-swapped without touching REST endpoints.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.ai.schemas import (
    IndicatorPrediction,
    PolicyRecommendation,
    CountryRiskScorecard,
    AnomalyAlert,
    ModelMetadata,
    ForecastHorizon
)

class AbstractModel(ABC):
    """
    Interface contract for any physical predictive or simulation algorithm.
    All econometric, ensemble, deep sequence, and graph neural network wrappers implement this base class.
    """
    @abstractmethod
    async def load(self, model_uri: str) -> bool:
        """Loads weights, parameters, or artifacts from the Model Registry."""
        pass

    @abstractmethod
    async def predict(
        self,
        iso3: str,
        indicator: str,
        horizon: ForecastHorizon,
        features: Dict[str, Any]
    ) -> IndicatorPrediction:
        """Runs inference to generate a compliant multi-field prediction object."""
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Returns training parameters, validation metrics, and versioning tags."""
        pass

class AbstractFeatureExtractor(ABC):
    """
    Interface for composable feature engineering and lag pipeline transformers.
    """
    @abstractmethod
    async def build_feature_vector(
        self,
        iso3: str,
        historical_series: Dict[str, List[float]],
        graph_metrics: Optional[Dict[str, float]] = None,
        simulation_shocks: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Synthesizes raw observations, Neo4j graph centralities, and simulation shocks into model-ready features."""
        pass

class AbstractRecommendationEngine(ABC):
    """
    Interface for prescriptive decision intelligence and policy optimization.
    """
    @abstractmethod
    async def generate_recommendations(
        self,
        target_iso3: str,
        current_risk_scorecard: CountryRiskScorecard,
        simulation_run_id: Optional[str] = None
    ) -> List[PolicyRecommendation]:
        """Evaluates policy trade-offs and generates actionable intervention directives."""
        pass

class AbstractRiskEngine(ABC):
    """
    Interface for the multi-dimensional macroeconomic Risk Assessment Engine.
    """
    @abstractmethod
    async def evaluate_country_risk(
        self,
        iso3: str,
        indicator_values: Dict[str, float],
        graph_metrics: Optional[Dict[str, float]] = None
    ) -> CountryRiskScorecard:
        """Computes continuous 0-100 scores and classifications across all 7 structural risk dimensions."""
        pass

class AbstractAnomalyDetector(ABC):
    """
    Interface for real-time statistical and unsupervised macroeconomic anomaly detection.
    """
    @abstractmethod
    async def scan_for_anomalies(
        self,
        global_indicators: Dict[str, Dict[str, float]],
        recent_shocks: Optional[List[Dict[str, Any]]] = None
    ) -> List[AnomalyAlert]:
        """Scans current multi-country observations to alert on severe economic disruptions."""
        pass

class AbstractExplainabilityEngine(ABC):
    """
    Interface for Explainable AI (XAI) feature attribution and Natural Language Synthesis.
    """
    @abstractmethod
    def explain_prediction(
        self,
        indicator: str,
        point_prediction: float,
        feature_importance_map: Dict[str, float],
        iso3: str
    ) -> str:
        """Synthesizes numerical SHAP/attention attribution weights into executive human-readable justifications."""
        pass

class AbstractDataConnector(ABC):
    """
    Interface for reusable institutional dataset ingestion adapters (World Bank, IMF, UN Comtrade, OECD, FRED).
    """
    @abstractmethod
    async def fetch_series(
        self,
        series_id: str,
        country_code: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves and normalizes empirical observation series from external institutional APIs or SDMX feeds."""
        pass

    @abstractmethod
    async def validate_and_clean(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Executes outlier clipping, type casting, and missing interval interpolation."""
        pass
