"""add_user_id_to_chat_sessions

Revision ID: 418796245a49
Revises: 384495b995a9
Create Date: 2025-06-05 17:18:40.810854

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '418796245a49'
down_revision: Union[str, None] = '384495b995a9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('chat_sessions', sa.Column('user_id', sa.UUID(as_uuid=True), nullable=False))
    op.create_index(op.f('ix_chat_sessions_user_id'), 'chat_sessions', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_chat_sessions_user_id'), table_name='chat_sessions')
    op.drop_column('chat_sessions', 'user_id')
