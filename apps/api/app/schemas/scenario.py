from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.models.scenario import EventCategory, EventStatus, ScenarioStatus

# Event Schemas
class ScenarioEventBase(BaseModel):
    parent_event_id: Optional[int] = None
    title: str
    category: EventCategory
    status: EventStatus
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Placeholders
    shockIntensity: Optional[float] = None
    propagationDelay: Optional[int] = None
    recoveryRate: Optional[float] = None
    affectedNodes: Optional[List[str]] = None
    confidence: Optional[float] = None
    simulationWeight: Optional[float] = None

class ScenarioEventCreate(ScenarioEventBase):
    pass

class ScenarioEventUpdate(BaseModel):
    """All fields optional — allows partial updates (only send what changed)."""
    parent_event_id: Optional[int] = None
    title: Optional[str] = None
    category: Optional[EventCategory] = None
    status: Optional[EventStatus] = None
    parameters: Optional[Dict[str, Any]] = None
    shockIntensity: Optional[float] = None
    propagationDelay: Optional[int] = None
    recoveryRate: Optional[float] = None
    affectedNodes: Optional[List[str]] = None
    confidence: Optional[float] = None
    simulationWeight: Optional[float] = None

class ScenarioEvent(ScenarioEventBase):
    id: int
    version_id: int

    class Config:
        from_attributes = True

# Version Schemas
class ScenarioVersionBase(BaseModel):
    version_number: int

class ScenarioVersionCreate(ScenarioVersionBase):
    pass

class ScenarioVersion(ScenarioVersionBase):
    id: int
    scenario_id: int
    created_at: datetime
    events: List[ScenarioEvent] = []

    class Config:
        from_attributes = True

# Scenario Schemas
class ScenarioBase(BaseModel):
    title: str
    description: Optional[str] = None
    author: Optional[str] = None
    is_template: bool = False
    tags: Optional[List[str]] = []
    status: ScenarioStatus = ScenarioStatus.DRAFT

class ScenarioCreate(ScenarioBase):
    pass

class ScenarioUpdate(ScenarioBase):
    title: Optional[str] = None

class Scenario(ScenarioBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    versions: List[ScenarioVersion] = []

    class Config:
        from_attributes = True
