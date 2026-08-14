"""
Multi-Dimensional Risk Assessment Engine implementing AbstractRiskEngine.

Evaluates normalized scores (0.0 to 100.0) across 7 structural macroeconomic dimensions:
Country, Trade, Economic Stability, Supply Chain, Financial, Inflation, and Currency risk.
Assigns Low, Medium, High, and Critical ordinal classifications per Requirement 1 & 13.
"""
from typing import Dict, Any, Optional
from datetime import datetime
from app.ai.interfaces import AbstractRiskEngine
from app.ai.schemas import CountryRiskScorecard, RiskDimensionScore, RiskTier
from app.ai.config import ai_config

class RiskAssessmentEngine(AbstractRiskEngine):
    """
    Sovereign risk calculation and diagnostic engine.
    """
    async def evaluate_country_risk(
        self,
        iso3: str,
        indicator_values: Dict[str, float],
        graph_metrics: Optional[Dict[str, float]] = None
    ) -> CountryRiskScorecard:
        if graph_metrics is None:
            from app.ai.feature_engineering.graph_metrics import GraphTopologyExtractor
            extractor = GraphTopologyExtractor()
            graph_metrics = await extractor.get_sovereign_graph_features(iso3)

        dimensions: Dict[str, RiskDimensionScore] = {}
        
        # 1. Inflation Risk
        cpi = indicator_values.get("Inflation Rate", 2.8)
        inf_score = self._clamp(cpi * 10.0 if cpi >= 0 else abs(cpi) * 15.0)
        dimensions["Inflation Risk"] = self._create_dimension("Inflation Risk", inf_score, ["Core CPI Volatility", "Purchasing Power Erosion"])

        # 2. Economic Stability Score (inverse to growth + unemployment penalty)
        gdp = indicator_values.get("GDP Growth", 2.5)
        unemp = indicator_values.get("Unemployment Rate", 4.5)
        stab_score = self._clamp((10.0 - max(-5.0, gdp)) * 4.0 + (unemp * 3.5))
        dimensions["Economic Stability Score"] = self._create_dimension("Economic Stability Score", stab_score, ["GDP Growth Staged Velocity", "Unemployment Slack"])

        # 3. Supply Chain Risk (Driven by Neo4j Import HHI & Betweenness)
        hhi = graph_metrics.get("graph_import_hhi_concentration", 0.28)
        betweenness = graph_metrics.get("graph_betweenness_centrality", 0.12)
        sc_score = self._clamp((hhi * 150.0) + (betweenness * 120.0))
        dimensions["Supply Chain Risk"] = self._create_dimension("Supply Chain Risk", sc_score, ["Import Supplier HHI Concentration", "Bilateral Chokepoint Dependency"])

        # 4. Financial Risk (Debt burden & interest rates)
        debt_gdp = indicator_values.get("Government Debt-to-GDP", 65.0)
        int_rate = indicator_values.get("Interest Rate", 4.25)
        fin_score = self._clamp((debt_gdp * 0.5) + (int_rate * 4.0))
        dimensions["Financial Risk"] = self._create_dimension("Financial Risk", fin_score, ["Sovereign Debt Burden", "Policy Lending Rate"])

        # 5. Trade Risk Score (Trade volume drop + tariff friction)
        trade_vol_shift = indicator_values.get("Trade Volume", 0.0)
        trade_score = self._clamp(40.0 - (trade_vol_shift * 0.5) + (hhi * 80.0))
        dimensions["Trade Risk Score"] = self._create_dimension("Trade Risk Score", trade_score, ["Bilateral Trade Exposure", "Customs Tariff Sensitivity"])

        # 6. Currency Risk (Exchange rate deviation & foreign reserves proxy)
        fx = indicator_values.get("Exchange Rate", 1.0)
        fx_score = self._clamp(30.0 + abs(1.0 - fx) * 45.0 + (cpi * 2.5))
        dimensions["Currency Risk"] = self._create_dimension("Currency Risk", fx_score, ["Exchange Rate Volatility", "Inflationary Depreciation"])

        # 7. Country Risk Score (Aggregate composite sovereign risk)
        avg_sub_risk = sum(d.score for d in dimensions.values()) / len(dimensions)
        dimensions["Country Risk Score"] = self._create_dimension("Country Risk Score", avg_sub_risk, ["Aggregate Structural Rollup", "Geopolitical Graph Dominance"])

        # Weighted rollup
        overall_score = round(avg_sub_risk, 2)
        overall_tier = self._get_tier_enum(overall_score)

        return CountryRiskScorecard(
            iso3=iso3,
            timestamp=datetime.utcnow(),
            overall_risk_score=overall_score,
            overall_risk_tier=overall_tier,
            dimensions=dimensions
        )

    def _create_dimension(self, name: str, score: float, contributors: list) -> RiskDimensionScore:
        clean_score = round(score, 2)
        tier = self._get_tier_enum(clean_score)
        return RiskDimensionScore(
            dimension_name=name,
            score=clean_score,
            tier=tier,
            primary_contributors=contributors
        )

    def _clamp(self, value: float) -> float:
        return max(0.0, min(100.0, value))

    def _get_tier_enum(self, score: float) -> RiskTier:
        label = ai_config.classify_risk_tier(score)
        return RiskTier(label)
