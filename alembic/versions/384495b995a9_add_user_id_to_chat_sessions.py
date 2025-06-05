"""add_user_id_to_chat_sessions

Revision ID: 384495b995a9
Revises: f8e6fc69710d
Create Date: 2025-06-05 17:18:34.034349

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '384495b995a9'
down_revision: Union[str, None] = 'f8e6fc69710d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
