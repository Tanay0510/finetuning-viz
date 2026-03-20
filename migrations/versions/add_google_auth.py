"""Add google auth fields

Revision ID: add_google_auth
Revises: add_recipes_table
Create Date: 2026-03-19 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_google_auth'
down_revision = 'add_recipes_table'
branch_labels = None
depends_on = None


def upgrade():
    # Add google_id and make password nullable
    op.add_column('users', sa.Column('google_id', sa.String(length=100), nullable=True))
    op.create_unique_constraint(None, 'users', ['google_id'])
    op.alter_column('users', 'password',
               existing_type=sa.VARCHAR(length=200),
               nullable=True)


def downgrade():
    op.alter_column('users', 'password',
               existing_type=sa.VARCHAR(length=200),
               nullable=False)
    op.drop_constraint(None, 'users', type_='unique')
    op.drop_column('users', 'google_id')
