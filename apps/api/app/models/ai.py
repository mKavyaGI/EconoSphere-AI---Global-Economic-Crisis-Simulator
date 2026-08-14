"""
SQLAlchemy ORM models for AI Forecasting & Decision Intelligence persistence.

Stores forecast results, risk scorecards, and inference audit logs in PostgreSQL
so that users can review historical AI predictions beyond the Redis 1-hour TTL.
"""
from datetime import datetime, timezone
from typing import Any
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, JSON, Text, Index, ForeignKey
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.database.base import Base


class AIForecast(Base):
    """
    Persists a complete CountryForecastResponse snapshot to PostgreSQL.
    Each row represents one forecast run for a given country + model combination.
    """
    __tablename__ = "ai_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    iso3 = Column(String(3), nullable=False, index=True)
    model_type = Column(String(64), nullable=False)
    model_version = Column(String(64), nullable=False)
    overall_economic_trajectory = Column(String(64), nullable=False)
    # Full CountryForecastResponse.predictions list stored as JSONB
    predictions = Column(JSONB, nullable=False, default=list)
    generated_at = Column(DateTime(timezone=True), nullable=False,
                          default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_ai_forecasts_iso3_model", "iso3", "model_type"),
        Index("ix_ai_forecasts_generated_at", "generated_at"),
    )


class AIRiskScorecard(Base):
    """
    Persists a CountryRiskScorecard to PostgreSQL.
    Each row is one risk evaluation for a country with 7-dimension breakdown.
    """
    __tablename__ = "ai_risk_scorecards"

    id = Column(Integer, primary_key=True, index=True)
    iso3 = Column(String(3), nullable=False, index=True)
    overall_risk_score = Column(Float, nullable=False)
    overall_risk_tier = Column(String(16), nullable=False)
    # Full dimensions dict stored as JSONB: {dim_name: RiskDimensionScore}
    dimensions = Column(JSONB, nullable=False, default=dict)
    evaluated_at = Column(DateTime(timezone=True), nullable=False,
                          default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_ai_risk_scorecards_iso3", "iso3"),
        Index("ix_ai_risk_scorecards_evaluated_at", "evaluated_at"),
    )


class AIInferenceLog(Base):
    """
    Immutable append-only audit log for every AI inference call.
    Provides full traceability: which model made which prediction at what time.
    """
    __tablename__ = "ai_inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    iso3 = Column(String(3), nullable=False, index=True)
    indicator = Column(String(128), nullable=False)
    model_id = Column(String(128), nullable=False, index=True)
    point_prediction = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    inferred_at = Column(DateTime(timezone=True), nullable=False,
                         default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_ai_inference_logs_iso3_model", "iso3", "model_id"),
        Index("ix_ai_inference_logs_inferred_at", "inferred_at"),
    )


class AIModelRegistry(Base):
    """
    Persistent model registry table — stores ModelMetadata in PostgreSQL
    so model lineage survives application restarts and Redis flushes.
    """
    __tablename__ = "ai_model_registry"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(128), nullable=False, unique=True, index=True)
    model_type = Column(String(64), nullable=False)
    version = Column(String(64), nullable=False)
    deployment_status = Column(String(32), nullable=False, default="STAGING")
    evaluation_metrics = Column(JSONB, nullable=False, default=dict)
    hyperparameters = Column(JSONB, nullable=False, default=dict)
    feature_list = Column(JSONB, nullable=False, default=list)
    trained_at = Column(DateTime(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc))
    registered_at = Column(DateTime(timezone=True), nullable=False,
                           default=lambda: datetime.now(timezone.utc))
