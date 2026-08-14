from typing import Dict, Any, Optional, Set, List
import structlog
from app.engine.interfaces import SimulationStrategy, GlobalState, ShockVector

logger = structlog.get_logger()


class TradeStrategy(SimulationStrategy):
    """
    Propagates trade-related shocks across the dependency graph.
    E.g., if Country A's exports drop, Country B's imports drop, leading to local inflation.
    
    Propagation Mathematical Formula:
        secondary_shock = primary_shock * edge_weight * (1 - decay_factor)
        
    Where:
        - primary_shock: The initial export_reduction value at the source node.
        - edge_weight: Proportional dependency parameter from graph edge properties (default 0.5 if unspecified).
        - decay_factor: Configured attenuation coefficient per hop (default 0.1).
        
    To ensure quantitative rigor and computational safety:
        1. Deterministic Execution: Target nodes are evaluated in strictly sorted order by ISO3 code.
        2. Cycle & Loop Protection: A visited set tracks traversed edges/nodes and halts propagation if magnitude drops below threshold.
    """

    def apply_shock(
        self, 
        state: GlobalState, 
        initial_shocks: ShockVector, 
        config: dict,
        dependency_map: Optional[Dict[str, Any]] = None
    ) -> ShockVector:
        decay = float(config.get("decay_factor", 0.1))
        threshold = float(config.get("propagation_threshold", 0.01))
        
        new_shocks: ShockVector = {}
        visited_edges: Set[tuple] = set()
        
        # Sort initial shocks by country code to guarantee determinism
        for country in sorted(initial_shocks.keys()):
            impacts = initial_shocks[country]
            if "export_reduction" not in impacts:
                continue
                
            primary_shock = impacts["export_reduction"]
            
            # If we have valid graph dependencies for this node, propagate over real edges
            if dependency_map and country in dependency_map and dependency_map[country].get("edges"):
                edges: List[dict] = dependency_map[country]["edges"]
                
                # Sort edges by target ISO3 code for guaranteed deterministic execution order
                sorted_edges = sorted(edges, key=lambda e: str(e.get("target", "")))
                
                for edge in sorted_edges:
                    target_iso = str(edge.get("target"))
                    # Check edge visitation and prevent self loops or immediate backward loop reflection
                    if not target_iso or target_iso == country or (country, target_iso) in visited_edges:
                        continue
                        
                    props = edge.get("properties", {})
                    # Derive weight from edge properties (e.g., volume or dependency ratio), fallback to 0.5
                    edge_weight = float(props.get("weight", 0.5))
                    
                    # Mathematical Equation: secondary_shock = primary_shock * edge_weight * (1 - decay_factor)
                    secondary_shock = abs(primary_shock) * edge_weight * (1.0 - decay)
                    
                    if secondary_shock > threshold:
                        if target_iso not in new_shocks:
                            new_shocks[target_iso] = {}
                        current_val = new_shocks[target_iso].get("inflation_increase", 0.0)
                        new_shocks[target_iso]["inflation_increase"] = current_val + secondary_shock
                        
                        visited_edges.add((country, target_iso))
                        visited_edges.add((target_iso, country)) # Prevent backward circular feedback in same step
            else:
                # Fallback mechanism when graph edges are empty or unavailable (maintains baseline test compatibility)
                shock_magnitude = abs(primary_shock) * (1.0 - decay)
                if shock_magnitude > threshold:
                    if "GLOBAL" not in new_shocks:
                        new_shocks["GLOBAL"] = {}
                    current_val = new_shocks["GLOBAL"].get("inflation_increase", 0.0)
                    new_shocks["GLOBAL"]["inflation_increase"] = current_val + (shock_magnitude * 0.5)
                    
        return new_shocks
