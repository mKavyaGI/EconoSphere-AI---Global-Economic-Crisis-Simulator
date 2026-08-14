from typing import List, Dict, Any, Optional
from app.engine.interfaces import SimulationStrategy, GlobalState, ShockVector


class StrategyManager:
    """Orchestrates multiple domain strategies to calculate cumulative shocks."""
    
    def __init__(self):
        self.strategies: List[SimulationStrategy] = []

    def register_strategy(self, strategy: SimulationStrategy):
        self.strategies.append(strategy)

    def apply_all(self, state: GlobalState, initial_shocks: ShockVector, config: dict, dependency_map: Optional[Dict[str, Any]] = None) -> ShockVector:
        """
        Runs all registered strategies and combines their resulting shocks.
        """
        cumulative_shocks: ShockVector = {}
        
        # Deep copy initial shocks into cumulative to act as base
        for country, impacts in initial_shocks.items():
            cumulative_shocks[country] = dict(impacts)

        for strategy in self.strategies:
            # Each strategy returns a dictionary of new shocks it generated
            strategy_shocks = strategy.apply_shock(state, cumulative_shocks, config, dependency_map=dependency_map)
            
            # Merge strategy shocks into cumulative shocks
            for country, impacts in strategy_shocks.items():
                if country not in cumulative_shocks:
                    cumulative_shocks[country] = {}
                
                for metric, val in impacts.items():
                    current_val = cumulative_shocks[country].get(metric, 0.0)
                    cumulative_shocks[country][metric] = current_val + val
                    
        return cumulative_shocks
