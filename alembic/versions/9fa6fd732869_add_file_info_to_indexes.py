"""add_file_info_to_indexes

Revision ID: 9fa6fd732869
Revises: 9f304976c801
Create Date: 2025-06-09 15:40:53.276860

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9fa6fd732869'
down_revision: Union[str, None] = '9f304976c801'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Add file information fields to indexes table
    op.add_column('indexes', sa.Column('file_md5', sa.String(32), nullable=True))
    op.add_column('indexes', sa.Column('file_path', sa.Text(), nullable=True))
    op.add_column('indexes', sa.Column('file_name', sa.String(255), nullable=True))
    op.add_column('indexes', sa.Column('file_size', sa.BigInteger(), nullable=True))
    op.add_column('indexes', sa.Column('file_extension', sa.String(50), nullable=True))
    op.add_column('indexes', sa.Column('mime_type', sa.String(255), nullable=True))
    op.add_column('indexes', sa.Column('document_count', sa.Integer(), nullable=True))
    op.add_column('indexes', sa.Column('node_count', sa.Integer(), nullable=True))
    op.add_column('indexes', sa.Column('vector_dimension', sa.Integer(), nullable=True))
    op.add_column('indexes', sa.Column('processing_config', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # Create index on file_md5 for fast lookup
    op.execute('CREATE INDEX IF NOT EXISTS ix_indexes_file_md5 ON indexes (file_md5)')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Remove file information fields from indexes table
    op.drop_index('ix_indexes_file_md5', table_name='indexes')
    op.drop_column('indexes', 'processing_config')
    op.drop_column('indexes', 'vector_dimension')
    op.drop_column('indexes', 'node_count')
    op.drop_column('indexes', 'document_count')
    op.drop_column('indexes', 'mime_type')
    op.drop_column('indexes', 'file_extension')
    op.drop_column('indexes', 'file_size')
    op.drop_column('indexes', 'file_name')
    op.drop_column('indexes', 'file_path')
    op.drop_column('indexes', 'file_md5')
    # ### end Alembic commands ###
