"""Add AI persistence tables

Revision ID: a1b2c3d4e5f6
Revises: 89c84394de95
Create Date: 2026-07-27 22:30:00.000000

Creates four tables for AI Forecasting & Decision Intelligence (Phase 10):
- ai_forecasts: Full forecast snapshots (predictions, trajectory)
- ai_risk_scorecards: 7-dimension sovereign risk evaluations
- ai_inference_logs: Immutable audit log of every model inference call
- ai_model_registry: Persistent model metadata and lineage
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '89c84394de95'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create AI persistence tables."""

    # ai_forecasts — stores full CountryForecastResponse per run
    op.create_table(
        'ai_forecasts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('iso3', sa.String(length=3), nullable=False),
        sa.Column('model_type', sa.String(length=64), nullable=False),
        sa.Column('model_version', sa.String(length=64), nullable=False),
        sa.Column('overall_economic_trajectory', sa.String(length=64), nullable=False),
        sa.Column('predictions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('generated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ai_forecasts_id'), 'ai_forecasts', ['id'], unique=False)
    op.create_index('ix_ai_forecasts_iso3_model', 'ai_forecasts', ['iso3', 'model_type'], unique=False)
    op.create_index('ix_ai_forecasts_generated_at', 'ai_forecasts', ['generated_at'], unique=False)

    # ai_risk_scorecards — stores CountryRiskScorecard per evaluation
    op.create_table(
        'ai_risk_scorecards',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('iso3', sa.String(length=3), nullable=False),
        sa.Column('overall_risk_score', sa.Float(), nullable=False),
        sa.Column('overall_risk_tier', sa.String(length=16), nullable=False),
        sa.Column('dimensions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('evaluated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ai_risk_scorecards_id'), 'ai_risk_scorecards', ['id'], unique=False)
    op.create_index('ix_ai_risk_scorecards_iso3', 'ai_risk_scorecards', ['iso3'], unique=False)
    op.create_index('ix_ai_risk_scorecards_evaluated_at', 'ai_risk_scorecards', ['evaluated_at'], unique=False)

    # ai_inference_logs — immutable append-only audit trail
    op.create_table(
        'ai_inference_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('iso3', sa.String(length=3), nullable=False),
        sa.Column('indicator', sa.String(length=128), nullable=False),
        sa.Column('model_id', sa.String(length=128), nullable=False),
        sa.Column('point_prediction', sa.Float(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('inferred_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ai_inference_logs_id'), 'ai_inference_logs', ['id'], unique=False)
    op.create_index('ix_ai_inference_logs_iso3_model', 'ai_inference_logs', ['iso3', 'model_id'], unique=False)
    op.create_index('ix_ai_inference_logs_inferred_at', 'ai_inference_logs', ['inferred_at'], unique=False)

    # ai_model_registry — persistent model metadata and lineage
    op.create_table(
        'ai_model_registry',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.String(length=128), nullable=False),
        sa.Column('model_type', sa.String(length=64), nullable=False),
        sa.Column('version', sa.String(length=64), nullable=False),
        sa.Column('deployment_status', sa.String(length=32), nullable=False),
        sa.Column('evaluation_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('hyperparameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('feature_list', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('trained_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('registered_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_id', name='uq_ai_model_registry_model_id')
    )
    op.create_index(op.f('ix_ai_model_registry_id'), 'ai_model_registry', ['id'], unique=False)
    op.create_index(op.f('ix_ai_model_registry_model_id'), 'ai_model_registry', ['model_id'], unique=True)


def downgrade() -> None:
    """Drop AI persistence tables."""
    op.drop_index(op.f('ix_ai_model_registry_model_id'), table_name='ai_model_registry')
    op.drop_index(op.f('ix_ai_model_registry_id'), table_name='ai_model_registry')
    op.drop_table('ai_model_registry')

    op.drop_index('ix_ai_inference_logs_inferred_at', table_name='ai_inference_logs')
    op.drop_index('ix_ai_inference_logs_iso3_model', table_name='ai_inference_logs')
    op.drop_index(op.f('ix_ai_inference_logs_id'), table_name='ai_inference_logs')
    op.drop_table('ai_inference_logs')

    op.drop_index('ix_ai_risk_scorecards_evaluated_at', table_name='ai_risk_scorecards')
    op.drop_index('ix_ai_risk_scorecards_iso3', table_name='ai_risk_scorecards')
    op.drop_index(op.f('ix_ai_risk_scorecards_id'), table_name='ai_risk_scorecards')
    op.drop_table('ai_risk_scorecards')

    op.drop_index('ix_ai_forecasts_generated_at', table_name='ai_forecasts')
    op.drop_index('ix_ai_forecasts_iso3_model', table_name='ai_forecasts')
    op.drop_index(op.f('ix_ai_forecasts_id'), table_name='ai_forecasts')
    op.drop_table('ai_forecasts')
