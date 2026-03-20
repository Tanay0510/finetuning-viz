"""Add zoo fields to recipes

Revision ID: add_zoo_fields
Revises: add_google_auth
Create Date: 2026-03-19 17:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_zoo_fields'
down_revision = 'add_google_auth'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('recipes', sa.Column('description', sa.Text(), nullable=True))
    op.add_column('recipes', sa.Column('is_public', sa.Boolean(), nullable=True))
    # Set existing recipes to private
    op.execute("UPDATE recipes SET is_public = FALSE WHERE is_public IS NULL")
    op.alter_column('recipes', 'is_public', nullable=False)


def downgrade():
    op.drop_column('recipes', 'is_public')
    op.drop_column('recipes', 'description')
