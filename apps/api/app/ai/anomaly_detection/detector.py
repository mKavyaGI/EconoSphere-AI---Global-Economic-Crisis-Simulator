"""
Unsupervised Anomaly Detection Engine implementing AbstractAnomalyDetector.
Scans global macroeconomic observations for severe deviations and assigns root-cause attributions.
"""
import math
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from app.ai.interfaces import AbstractAnomalyDetector
from app.ai.schemas import AnomalyAlert, AnomalySeverity
from app.core.config import settings

class AnomalyDetectorEngine(AbstractAnomalyDetector):
    """
    Real-time economic disruption and anomaly scanning engine.
    """
    def __init__(self):
        self.z_threshold = settings.ai_anomaly_zscore_threshold

    async def scan_for_anomalies(
        self,
        global_indicators: Dict[str, Dict[str, float]],
        recent_shocks: Optional[List[Dict[str, Any]]] = None
    ) -> List[AnomalyAlert]:
        alerts: List[AnomalyAlert] = []
        if not global_indicators:
            return alerts

        metrics_pool: Dict[str, List[float]] = {}
        for iso3, ind_map in global_indicators.items():
            for k, val in ind_map.items():
                if k not in metrics_pool:
                    metrics_pool[k] = []
                metrics_pool[k].append(float(val))

        stats_map: Dict[str, Dict[str, float]] = {}
        for k, values in metrics_pool.items():
            n = len(values)
            if n < 2:
                stats_map[k] = {"mean": sum(values)/max(1, n), "std": 1.0}
            else:
                mean = sum(values)/n
                variance = sum((x - mean)**2 for x in values)/(n - 1)
                stats_map[k] = {"mean": mean, "std": max(0.001, math.sqrt(variance))}

        alert_counter = 101
        for iso3, ind_map in global_indicators.items():
            for k, val in ind_map.items():
                st = stats_map.get(k, {"mean": val, "std": 1.0})
                z = abs((val - st["mean"]) / st["std"])
                
                is_severe_inflation = (k == "Inflation Rate" and val > 12.0)
                is_gdp_collapse = (k == "GDP Growth" and val < -4.0)
                
                if z >= self.z_threshold or is_severe_inflation or is_gdp_collapse:
                    severity = AnomalySeverity.CRITICAL if (z > 3.0 or is_gdp_collapse or val > 50.0) else AnomalySeverity.SEVERE
                    
                    anomaly_type = f"UNEXPECTED_{k.upper().replace(' ', '_')}_DEVIATION"
                    if is_severe_inflation:
                        anomaly_type = "RAPID_INFLATIONARY_SPIKE"
                    elif is_gdp_collapse:
                        anomaly_type = "SUDDEN_GDP_CONTRACTION"
                        
                    causes = [
                        f"Statistical Z-score deviation of {round(z, 2)}σ from global baseline ({round(st['mean'], 2)})",
                        "Neo4j upstream trade corridor supply elasticity breakdown",
                        "Aggressive customs tariff or monetary shock contagion"
                    ]
                    if recent_shocks:
                        causes.append("Correlated with recently applied simulation scenario events")

                    alerts.append(AnomalyAlert(
                        anomaly_id=f"ANOM-{iso3}-{alert_counter}",
                        detected_anomaly=anomaly_type,
                        severity=severity,
                        timestamp=datetime.now(timezone.utc),
                        affected_countries=[iso3],
                        possible_causes=causes,
                        anomaly_score_z=round(z, 2)
                    ))
                    alert_counter += 1

        return alerts
