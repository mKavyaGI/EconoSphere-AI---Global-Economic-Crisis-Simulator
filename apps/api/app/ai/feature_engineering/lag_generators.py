"""
Lag variable generators, rolling moving average calculators, and differential growth rate estimators
for non-stationary macroeconomic observation series.
"""
from typing import List, Dict, Optional
import math

class LagGenerator:
    """
    Mathematical lag and time-series feature transformer.
    Generates temporal memory indicators without data leakage across horizons.
    """
    @staticmethod
    def generate_lags(series: List[float], lags: List[int] = [1, 3, 6, 12]) -> Dict[str, float]:
        """Returns lag dictionary for the most recent observation in the historical series."""
        n = len(series)
        result = {}
        if n == 0:
            for lag in lags:
                result[f"lag_{lag}"] = 0.0
            return result
            
        current_val = series[-1]
        for lag in lags:
            if n > lag:
                result[f"lag_{lag}"] = series[-1 - lag]
            else:
                # If history is shorter than lag, fallback to earliest observation
                result[f"lag_{lag}"] = series[0]
        return result

    @staticmethod
    def calculate_growth_rates(series: List[float]) -> Dict[str, float]:
        """Computes MoM (or immediate step) and YoY (12 step or earliest) percentage growth rates."""
        n = len(series)
        if n < 2:
            return {"mom_growth": 0.0, "yoy_growth": 0.0}
            
        curr = series[-1]
        prev_1 = series[-2]
        mom = ((curr - prev_1) / abs(prev_1)) * 100.0 if prev_1 != 0 else 0.0
        
        yoy_idx = max(0, n - 12)
        prev_yoy = series[yoy_idx]
        yoy = ((curr - prev_yoy) / abs(prev_yoy)) * 100.0 if prev_yoy != 0 else mom
        
        return {
            "mom_growth": round(mom, 4),
            "yoy_growth": round(yoy, 4)
        }

    @staticmethod
    def calculate_rolling_statistics(series: List[float], window: int = 6) -> Dict[str, float]:
        """Computes rolling simple moving average (SMA) and standard deviation volatility envelope."""
        n = len(series)
        if n == 0:
            return {"rolling_sma": 0.0, "rolling_std": 0.0}
            
        sub_series = series[-window:] if n >= window else series
        k = len(sub_series)
        sma = sum(sub_series) / k
        
        variance = sum((x - sma) ** 2 for x in sub_series) / k
        std = math.sqrt(variance)
        
        return {
            "rolling_sma": round(sma, 4),
            "rolling_std": round(std, 4)
        }
