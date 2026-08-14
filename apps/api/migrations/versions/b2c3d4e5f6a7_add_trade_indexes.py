"""Add explicit indexes for scenario and trade network query access patterns

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-28 10:00:00.000000

Creates high-performance B-tree indexes for Phase 7 (Global Trade Intelligence)
and Phase 8 (Scenario Management) query access patterns:
- scenarios (status, is_template, title)
- scenario_versions (scenario_id)
- scenario_events (version_id, parent_event_id, category)
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Scenario filtering and title lookup indexes
    op.create_index('ix_scenarios_status', 'scenarios', ['status'], unique=False)
    op.create_index('ix_scenarios_is_template', 'scenarios', ['is_template'], unique=False)
    op.create_index('ix_scenarios_title', 'scenarios', ['title'], unique=False)

    # Hierarchical relational indexes for fast scenario graph loading
    op.create_index('ix_scenario_versions_scenario_id', 'scenario_versions', ['scenario_id'], unique=False)
    op.create_index('ix_scenario_events_version_id', 'scenario_events', ['version_id'], unique=False)
    op.create_index('ix_scenario_events_parent_event_id', 'scenario_events', ['parent_event_id'], unique=False)
    op.create_index('ix_scenario_events_category', 'scenario_events', ['category'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_scenario_events_category', table_name='scenario_events')
    op.drop_index('ix_scenario_events_parent_event_id', table_name='scenario_events')
    op.drop_index('ix_scenario_events_version_id', table_name='scenario_events')
    op.drop_index('ix_scenario_versions_scenario_id', table_name='scenario_versions')
    op.drop_index('ix_scenarios_title', table_name='scenarios')
    op.drop_index('ix_scenarios_is_template', table_name='scenarios')
    op.drop_index('ix_scenarios_status', table_name='scenarios')
