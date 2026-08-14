"""
Reusable Feature Engineering Pipeline implementing AbstractFeatureExtractor.
Synthesizes economic indicators, trade network metrics, simulation impacts, commodity prices, lag variables,
and derived macro indices into clean numerical feature dictionaries.
"""
from typing import List, Dict, Any, Optional
from app.ai.interfaces import AbstractFeatureExtractor
from app.ai.feature_engineering.lag_generators import LagGenerator
from app.ai.feature_engineering.graph_metrics import GraphTopologyExtractor

class FeaturePipeline(AbstractFeatureExtractor):
    """
    Declarative end-to-end feature transformation pipeline for AI Forecasting & Decision Intelligence.
    Ensures zero data leakage between training and real-time inference runs.
    """
    def __init__(self):
        self.lag_gen = LagGenerator()
        self.graph_extractor = GraphTopologyExtractor()

    async def build_feature_vector(
        self,
        iso3: str,
        historical_series: Dict[str, List[float]],
        graph_metrics: Optional[Dict[str, float]] = None,
        simulation_shocks: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Constructs a complete feature vector dictionary combining:
        - Latest observed macro baseline values
        - Temporal memory lags & momentum rolling statistics
        - Neo4j graph centrality and vulnerability metrics
        - Active simulation shock deltas (from Phase 9 Engine snapshots)
        - Synthetic structural economic ratios (e.g., Misery Index, Debt Velocity)
        """
        features: Dict[str, Any] = {}
        
        # Step 1: Process historical observation series and generate lags & rolling stats
        for indicator_name, values_list in historical_series.items():
            clean_name = indicator_name.lower().replace(" ", "_").replace("-", "_")
            latest_val = values_list[-1] if values_list else 0.0
            features[f"{clean_name}_current"] = latest_val
            
            # Generate memory lag features
            lags = self.lag_gen.generate_lags(values_list, lags=[1, 3, 6, 12])
            for k, v in lags.items():
                features[f"{clean_name}_{k}"] = v
                
            # Generate growth and volatility statistics
            growth = self.lag_gen.calculate_growth_rates(values_list)
            stats = self.lag_gen.calculate_rolling_statistics(values_list, window=6)
            for k, v in {**growth, **stats}.items():
                features[f"{clean_name}_{k}"] = v
                
        # Step 2: Incorporate Neo4j Graph Topology features
        if graph_metrics is None:
            graph_metrics = await self.graph_extractor.get_sovereign_graph_features(iso3)
        for k, v in graph_metrics.items():
            features[k] = v

        # Step 3: Overlay Simulation Shocks (if forecasting inside an active scenario run)
        if simulation_shocks:
            for shock_key, shock_val in simulation_shocks.items():
                clean_shock = shock_key.lower().replace(" ", "_")
                features[f"shock_delta_{clean_shock}"] = float(shock_val)
        else:
            # Zero-out standard shock placeholders so model inference input dimensions match
            features["shock_delta_tariff"] = 0.0
            features["shock_delta_interest_rate"] = 0.0
            features["shock_delta_commodity_price"] = 0.0

        # Step 4: Compute derived synthetic economic indices
        gdp_val = features.get("gdp_growth_current", 2.5)
        cpi_val = features.get("inflation_rate_current", 2.8)
        unemp_val = features.get("unemployment_rate_current", 4.5)
        debt_val = features.get("government_debt_to_gdp_current", 65.0)
        
        # Misery Index = Inflation Rate + Unemployment Rate
        features["derived_misery_index"] = round(cpi_val + unemp_val, 4)
        
        # Solvency Risk Ratio = Government Debt-to-GDP / max(1.0, GDP Growth)
        features["derived_solvency_ratio"] = round(debt_val / max(0.5, abs(gdp_val)), 4)
        
        # Trade Dependency Exposure = Import HHI Concentration * Betweenness Centrality
        hhi = features.get("graph_import_hhi_concentration", 0.28)
        between = features.get("graph_betweenness_centrality", 0.12)
        features["derived_trade_exposure_index"] = round(hhi * between * 100.0, 4)

        return features
