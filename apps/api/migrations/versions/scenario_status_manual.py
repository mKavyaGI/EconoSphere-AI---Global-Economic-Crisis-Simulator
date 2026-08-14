"""manual status update

Revision ID: scenario_status_manual
Revises: e1161522f790
Create Date: 2026-07-23 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'scenario_status_manual'
down_revision: Union[str, Sequence[str], None] = 'e1161522f790'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass

def downgrade() -> None:
    pass
