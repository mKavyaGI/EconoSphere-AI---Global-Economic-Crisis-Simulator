"""
Natural Language Synthesizer (NLG) converting numerical Shapley attribution matrices
into clear, executive-ready diagnostic explanations.
"""
from typing import Dict, List
from app.ai.schemas import FeatureAttribution

class NaturalLanguageSynthesizer:
    """
    Synthesizes executive human-readable explanations justifying predictive model outputs.
    """
    @staticmethod
    def generate_executive_summary(
        indicator: str,
        point_prediction: float,
        top_features: List[FeatureAttribution],
        iso3: str
    ) -> str:
        if not top_features:
            return f"Model forecast for {indicator} in {iso3} is projected at {point_prediction} based on aggregate macro trends."
            
        primary = top_features[0]
        secondary = top_features[1] if len(top_features) > 1 else None
        
        # Build narrative syntax based on direction
        dir_verb = "upward pressure from" if primary.direction == "POSITIVE" else "downward drag due to"
        desc_1 = primary.human_label or primary.feature_name.replace("_", " ")
        
        narrative = (
            f"The projected {indicator} trajectory of {point_prediction} for {iso3} is primarily driven by {dir_verb} "
            f"{desc_1} ({primary.importance_score}% relative impact)."
        )
        
        if secondary:
            sec_dir = "reinforced by" if secondary.direction == primary.direction else "partially counterbalanced by"
            desc_2 = secondary.human_label or secondary.feature_name.replace("_", " ")
            narrative += f" This is {sec_dir} structural shifts in {desc_2} ({secondary.importance_score}% impact)."
            
        return narrative
