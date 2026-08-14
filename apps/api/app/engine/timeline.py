from typing import List, Dict, Any, Generator
from app.engine.interfaces import GlobalState, ShockVector

class TimelineGenerator:
    """Manages progression of time across configured horizons."""
    
    def __init__(self, time_horizons: List[int]):
        # e.g., [1, 7, 30, 180, 365, 1095]
        self.horizons = sorted(time_horizons)
        
    def generate_horizons(self) -> Generator[int, None, None]:
        """Yields each horizon in days."""
        for days in self.horizons:
            yield days
            
    def apply_time_lag(self, shocks: ShockVector, days: int) -> ShockVector:
        """
        Scales shocks based on the time horizon. Some shocks are immediate (Day 1),
        others peak later (Month 6).
        """
        import copy
        scaled_shocks = copy.deepcopy(shocks)
        
        for country, impacts in scaled_shocks.items():
            for metric, value in impacts.items():
                # Example: Inflation takes time to build up
                if "inflation" in metric:
                    # Peaks around 180 days
                    if days < 30:
                        impacts[metric] = value * 0.2
                    elif days < 180:
                        impacts[metric] = value * 0.8
                    else:
                        impacts[metric] = value * 1.0
                
        return scaled_shocks
