from sqlalchemy import Column, String, Integer, Float, ForeignKey, DateTime, Boolean, Enum, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.database.base import Base

class ScenarioStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    PUBLISHED = "PUBLISHED"
    ARCHIVED = "ARCHIVED"

class EventCategory(str, enum.Enum):
    ECONOMIC = "ECONOMIC"
    POLITICAL = "POLITICAL"
    ENVIRONMENTAL = "ENVIRONMENTAL"
    FINANCIAL = "FINANCIAL"
    HEALTH = "HEALTH"
    ENERGY = "ENERGY"
    TECHNOLOGY = "TECHNOLOGY"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"

class EventStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    ARCHIVED = "ARCHIVED"

class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False, index=True)
    description = Column(String(1024), nullable=True)
    author = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_template = Column(Boolean, default=False, index=True)
    tags = Column(JSON, nullable=True)  # List of strings
    status = Column(Enum(ScenarioStatus), nullable=False, default=ScenarioStatus.DRAFT, index=True)

    versions = relationship("ScenarioVersion", back_populates="scenario", cascade="all, delete-orphan", lazy="selectin")


class ScenarioVersion(Base):
    __tablename__ = "scenario_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    scenario = relationship("Scenario", back_populates="versions")
    events = relationship("ScenarioEvent", back_populates="version", cascade="all, delete-orphan", lazy="selectin")


class ScenarioEvent(Base):
    __tablename__ = "scenario_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_id = Column(Integer, ForeignKey("scenario_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    parent_event_id = Column(Integer, ForeignKey("scenario_events.id", ondelete="SET NULL"), nullable=True, index=True)
    
    title = Column(String(255), nullable=False)
    category = Column(Enum(EventCategory), nullable=False, default=EventCategory.ECONOMIC, index=True)
    status = Column(Enum(EventStatus), nullable=False, default=EventStatus.DRAFT)
    
    # Store event-specific settings here
    parameters = Column(JSON, nullable=False, default={})
    
    # Phase 9 Placeholders
    shockIntensity = Column(Float, nullable=True)
    propagationDelay = Column(Integer, nullable=True) # in days or months
    recoveryRate = Column(Float, nullable=True)
    affectedNodes = Column(JSON, nullable=True) # List of node IDs
    confidence = Column(Float, nullable=True)
    simulationWeight = Column(Float, nullable=True)

    version = relationship("ScenarioVersion", back_populates="events")
    parent_event = relationship("ScenarioEvent", remote_side=[id], back_populates="sub_events")
    sub_events = relationship("ScenarioEvent", back_populates="parent_event", lazy="selectin")

