"""
Unit tests for deterministic graph shock propagation, loop protection, and strategy resilience in Phase 9.
"""
import pytest
from app.engine.strategies.trade import TradeStrategy
from app.engine.strategies.manager import StrategyManager

@pytest.fixture
def trade_strategy():
    return TradeStrategy()

@pytest.fixture
def strategy_manager(trade_strategy):
    manager = StrategyManager()
    manager.register_strategy(trade_strategy)
    return manager

def test_fallback_global_propagation(trade_strategy):
    """Verify fallback to 'GLOBAL' when dependency_map is null or empty."""
    initial_shocks = {"USA": {"export_reduction": -0.20}}
    config = {"decay_factor": 0.1, "propagation_threshold": 0.01}
    
    shocks = trade_strategy.apply_shock({}, initial_shocks, config, dependency_map=None)
    assert "GLOBAL" in shocks
    # Magnitude check: abs(-0.20) * 0.9 * 0.5 = 0.09
    assert abs(shocks["GLOBAL"]["inflation_increase"] - 0.09) < 1e-6

def test_deterministic_order_and_magnitude_routing(trade_strategy):
    """Verify shocks route precisely to target graph edges in sorted deterministic order."""
    initial_shocks = {"USA": {"export_reduction": -0.30}}
    config = {"decay_factor": 0.2, "propagation_threshold": 0.01}
    
    dependency_map = {
        "USA": {
            "edges": [
                {"source": "USA", "target": "JPN", "properties": {"weight": 0.4}},
                {"source": "USA", "target": "CAN", "properties": {"weight": 0.8}},
                {"source": "USA", "target": "DEU", "properties": {"weight": 0.3}},
            ]
        }
    }
    
    shocks_run1 = trade_strategy.apply_shock({}, initial_shocks, config, dependency_map=dependency_map)
    shocks_run2 = trade_strategy.apply_shock({}, initial_shocks, config, dependency_map=dependency_map)
    
    # 1. Determinism verification
    assert list(shocks_run1.keys()) == list(shocks_run2.keys())
    assert shocks_run1 == shocks_run2
    assert "GLOBAL" not in shocks_run1
    
    # 2. Magnitude verification (primary * weight * (1 - decay))
    # For CAN: abs(-0.30) * 0.8 * 0.8 = 0.192
    assert abs(shocks_run1["CAN"]["inflation_increase"] - 0.192) < 1e-6
    # For JPN: abs(-0.30) * 0.4 * 0.8 = 0.096
    assert abs(shocks_run1["JPN"]["inflation_increase"] - 0.096) < 1e-6
    # For DEU: abs(-0.30) * 0.3 * 0.8 = 0.072
    assert abs(shocks_run1["DEU"]["inflation_increase"] - 0.072) < 1e-6

def test_cycle_and_loop_protection(trade_strategy):
    """Verify visited sets protect against infinite loops or self-referential graph cycles."""
    initial_shocks = {"CHN": {"export_reduction": -0.50}}
    config = {"decay_factor": 0.1, "propagation_threshold": 0.01}
    
    # Cyclic loop: CHN -> AUS -> CHN or self-loop CHN -> CHN
    dependency_map = {
        "CHN": {
            "edges": [
                {"source": "CHN", "target": "CHN", "properties": {"weight": 0.9}}, # Self loop should be ignored
                {"source": "CHN", "target": "AUS", "properties": {"weight": 0.5}},
            ]
        }
    }
    
    shocks = trade_strategy.apply_shock({}, initial_shocks, config, dependency_map=dependency_map)
    
    # Self loop must NOT trigger inflation back on CHN during 1st pass
    assert "CHN" not in shocks
    assert "AUS" in shocks

def test_multiple_simultaneous_shocks_aggregation(strategy_manager):
    """Verify simultaneous export shocks from multiple nodes cleanly aggregate on mutual trade targets."""
    initial_shocks = {
        "USA": {"export_reduction": -0.20},
        "CHN": {"export_reduction": -0.20}
    }
    config = {"decay_factor": 0.0, "propagation_threshold": 0.01}
    
    # Both USA and CHN trade with MEX
    dependency_map = {
        "USA": {"edges": [{"source": "USA", "target": "MEX", "properties": {"weight": 0.5}}]},
        "CHN": {"edges": [{"source": "CHN", "target": "MEX", "properties": {"weight": 0.5}}]}
    }
    
    cumulative = strategy_manager.apply_all({}, initial_shocks, config, dependency_map=dependency_map)
    
    assert "MEX" in cumulative
    # From USA: 0.20 * 0.5 * 1.0 = 0.10
    # From CHN: 0.20 * 0.5 * 1.0 = 0.10
    # Aggregate target total: 0.20
    assert abs(cumulative["MEX"]["inflation_increase"] - 0.20) < 1e-6
