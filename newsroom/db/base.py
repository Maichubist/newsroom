"""Engine, session factory, and schema bootstrap.

Per the owner's decision we build the schema from a clean slate with
``create_all`` instead of Alembic migrations (a deviation from architecture
§11/§14, reversible later when the schema starts to evolve across stages).
``create_all`` is idempotent (``checkfirst=True``), which gives us the
"re-run does not break" property the готовності criteria ask for.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# text-embedding-3-small dimensionality. The model NAME is stored per embedding
# row (item_embeddings.model), so the model can change; the column width is fixed
# at table-creation time, so changing it later means a real schema change.
EMBEDDING_DIM = 1536


class Base(DeclarativeBase):
    pass


def make_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    url = url or os.environ["NEWSROOM_DATABASE_URL"]
    return create_engine(url, echo=echo, future=True, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


# Additive columns introduced after a table already exists in the field. create_all
# only creates missing *tables*, never adds a column to an existing one, so these
# non-destructive ALTERs bring an older DB up to date. Each is idempotent
# (ADD COLUMN IF NOT EXISTS); keep them append-only (architecture: only additive
# migrations, nothing dropped or renamed).
_ADDITIVE_COLUMNS = (
    "ALTER TABLE events ADD COLUMN IF NOT EXISTS significance double precision",
    "ALTER TABLE events ADD COLUMN IF NOT EXISTS curated varchar(16)",
)


def init_db(engine: Engine) -> None:
    """Create the pgvector extension and all tables, then apply additive column
    migrations. Safe to call repeatedly."""
    # Import registers every model on Base.metadata before create_all.
    from newsroom import models  # noqa: F401

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for stmt in _ADDITIVE_COLUMNS:
            conn.execute(text(stmt))
