from typing import List, Dict, Any
from app.models.scenario import ScenarioEvent
from app.engine.interfaces import ShockVector

class EventProcessor:
    """Parses ScenarioEvents into quantitative initial ShockVectors."""
    
    def process(self, events: List[ScenarioEvent]) -> ShockVector:
        """
        Translates a list of ScenarioEvents into a unified ShockVector.
        Example Output:
        {
            "USA": {"tariff": 0.25, "export_reduction": -0.1},
            "CHN": {"export_reduction": -0.25}
        }
        """
        initial_shocks: ShockVector = {}
        
        for event in events:
            # We assume event.affectedNodes contains {"USA": {"impact": 0.1}, ...}
            # or parameters contain specific macro impacts
            
            affected_nodes = event.affectedNodes or {}
            intensity = event.shockIntensity or 1.0
            
            for country_iso, impacts in affected_nodes.items():
                if country_iso not in initial_shocks:
                    initial_shocks[country_iso] = {}
                
                # Apply the intensity multiplier to the baseline impact
                for metric, value in impacts.items():
                    current_val = initial_shocks[country_iso].get(metric, 0.0)
                    # Accumulate shocks additively for simplicity
                    initial_shocks[country_iso][metric] = current_val + (value * intensity)
                    
        return initial_shocks
