from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from app.models.simulation import SimulationStatus

class SimulationConfig(BaseModel):
    time_horizons: List[int] = Field(default=[1, 7, 30, 180, 365, 1095], description="Days to simulate")
    max_depth: int = Field(default=3, description="Maximum degrees of separation for shock propagation")
    propagation_threshold: float = Field(default=0.01, description="Minimum shock intensity to propagate")
    decay_factor: float = Field(default=0.1, description="Decay factor per node hop")
    random_seed: Optional[int] = None
    confidence_level: float = Field(default=0.95, description="Confidence interval for forecasting")
    max_iterations: int = Field(default=10, description="Max iterations to resolve loops")


class SimulationStartRequest(BaseModel):
    scenario_id: int
    config: Optional[SimulationConfig] = None


class SimulationRunResponse(BaseModel):
    id: int
    scenario_id: int
    status: SimulationStatus
    config: dict
    started_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SimulationSnapshotResponse(BaseModel):
    id: int
    run_id: int
    horizon_days: int
    timestamp: datetime
    global_state: dict
    active_shocks: dict

    class Config:
        from_attributes = True
