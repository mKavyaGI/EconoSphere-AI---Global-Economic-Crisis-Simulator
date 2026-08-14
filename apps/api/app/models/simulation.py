from datetime import datetime
from typing import Optional, Any
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Enum, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
import enum

from app.database.base import Base


class SimulationStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


class LogSeverity(str, enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[SimulationStatus] = mapped_column(Enum(SimulationStatus), default=SimulationStatus.PENDING, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    scenario = relationship("Scenario", backref="simulation_runs", lazy="selectin")
    snapshots = relationship("SimulationSnapshot", back_populates="run", cascade="all, delete-orphan", lazy="selectin")
    logs = relationship("SimulationLog", back_populates="run", cascade="all, delete-orphan", lazy="selectin")


class SimulationSnapshot(Base):
    __tablename__ = "simulation_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("simulation_runs.id", ondelete="CASCADE"), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    global_state: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    active_shocks: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    run = relationship("SimulationRun", back_populates="snapshots")


class SimulationLog(Base):
    __tablename__ = "simulation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("simulation_runs.id", ondelete="CASCADE"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    component: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(String, nullable=False)
    severity: Mapped[LogSeverity] = mapped_column(Enum(LogSeverity), default=LogSeverity.INFO, nullable=False)
    affected_country: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    affected_indicator: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    run = relationship("SimulationRun", back_populates="logs")
