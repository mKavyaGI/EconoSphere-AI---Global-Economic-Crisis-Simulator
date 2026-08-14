from typing import List, Dict, Any
from app.engine.interfaces import IndicatorCalculator, GlobalState, ShockVector


class CalculatorManager:
    """Orchestrates independent indicator calculators."""

    def __init__(self):
        self.calculators: List[IndicatorCalculator] = []

    def register_calculator(self, calculator: IndicatorCalculator):
        self.calculators.append(calculator)

    def calculate_all(self, state: GlobalState, cumulative_shocks: ShockVector) -> GlobalState:
        """
        Sequentially mutates the state through all registered calculators.
        """
        # We work on a copy of the state to avoid mutating the original baseline
        import copy
        new_state = copy.deepcopy(state)

        for calc in self.calculators:
            new_state = calc.calculate(new_state, cumulative_shocks)
            
        return new_state
