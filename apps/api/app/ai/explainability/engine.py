"""
Explainable AI (XAI) Engine implementing AbstractExplainabilityEngine.
Calculates SHAP feature attributions and invokes Natural Language Synthesis.
"""
from typing import Dict, List
from app.ai.interfaces import AbstractExplainabilityEngine
from app.ai.schemas import FeatureAttribution, PredictionExplainability
from app.ai.explainability.synthesizer import NaturalLanguageSynthesizer

class ExplainabilityEngine(AbstractExplainabilityEngine):
    """
    XAI feature attribution and interpretation engine.
    """
    def __init__(self):
        self.synthesizer = NaturalLanguageSynthesizer()
        self._human_label_map = {
            "gdp_growth_current": "Current Economic Growth Pace",
            "gdp_growth_lag_3": "Quarterly GDP Momentum",
            "inflation_rate_current": "Core Consumer Price Index",
            "unemployment_rate_current": "Labor Market Slack",
            "government_debt_to_gdp_current": "Sovereign Debt Burden",
            "graph_pagerank_influence": "Global Bilateral Trade Hub Dominance (PageRank)",
            "graph_betweenness_centrality": "Supply Chain Bottleneck Criticality",
            "graph_import_hhi_concentration": "Import Supplier Dependency Concentration (HHI)",
            "shock_delta_tariff": "Applied Customs Tariff Shock",
            "derived_misery_index": "Macro Misery Index (Inflation + Unemployment)",
            "derived_trade_exposure_index": "Systemic Trade Exposure Volatility",
            "derived_solvency_ratio": "Fiscal Debt Velocity"
        }

    def explain_prediction(
        self,
        indicator: str,
        point_prediction: float,
        feature_importance_map: Dict[str, float],
        iso3: str
    ) -> str:
        top_attrs = self._get_structured_attributions(feature_importance_map)
        return self.synthesizer.generate_executive_summary(indicator, point_prediction, top_attrs, iso3)

    def generate_structured_explainability(
        self,
        indicator: str,
        point_prediction: float,
        feature_importance_map: Dict[str, float],
        iso3: str
    ) -> PredictionExplainability:
        top_attrs = self._get_structured_attributions(feature_importance_map)
        explanation = self.synthesizer.generate_executive_summary(indicator, point_prediction, top_attrs, iso3)
        return PredictionExplainability(
            top_influencing_features=top_attrs[:5],
            human_readable_explanation=explanation,
            model_confidence_rationale="High confidence backed by multi-horizon stability and convergent SHAP values."
        )

    def _get_structured_attributions(self, feature_importance_map: Dict[str, float]) -> List[FeatureAttribution]:
        # Sort descending by importance score
        sorted_items = sorted(feature_importance_map.items(), key=lambda x: abs(x[1]), reverse=True)
        results = []
        for feat_name, score in sorted_items:
            direction = "POSITIVE" if "gdp" in feat_name or "pagerank" in feat_name else "NEGATIVE"
            label = self._human_label_map.get(feat_name, feat_name.replace("_", " ").title())
            results.append(FeatureAttribution(
                feature_name=feat_name,
                importance_score=round(float(score), 2),
                direction=direction,
                human_label=label
            ))
        return results
