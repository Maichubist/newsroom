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
    "ALTER TABLE events ADD COLUMN IF NOT EXISTS duplicate_of bigint",
    # classification cache — the LLM classifier runs once per event, not per item
    "ALTER TABLE events ADD COLUMN IF NOT EXISTS side varchar(8)",
    "ALTER TABLE events ADD COLUMN IF NOT EXISTS is_first_source boolean",
    "ALTER TABLE events ADD COLUMN IF NOT EXISTS is_rumor boolean",
    "ALTER TABLE events ADD COLUMN IF NOT EXISTS classifier_model varchar(64)",
    "ALTER TABLE events ADD COLUMN IF NOT EXISTS keywords jsonb",
    # learned taxonomy pyramid (charter v0.3 §3.1)
    "ALTER TABLE events ADD COLUMN IF NOT EXISTS topic_path jsonb",
    "ALTER TABLE events ADD COLUMN IF NOT EXISTS topic_leaf_id bigint",
    # taxonomy node engagement heat (Phase 2)
    "ALTER TABLE taxonomy_nodes ADD COLUMN IF NOT EXISTS heat double precision DEFAULT 0.0",
    "ALTER TABLE taxonomy_nodes ADD COLUMN IF NOT EXISTS heat_events integer DEFAULT 0",
    "ALTER TABLE taxonomy_nodes ADD COLUMN IF NOT EXISTS heat_at timestamptz",
    # local media file deleted after publication (phash reuse-archive stays in DB)
    "ALTER TABLE media_assets ADD COLUMN IF NOT EXISTS purged_at timestamptz",
    # Telegram message id for re-fetching url-less media via Telethon
    "ALTER TABLE media_assets ADD COLUMN IF NOT EXISTS source_ref text",
    # decisions is the hottest write path (every stage + observe-mode dedup logs here).
    # These indexes keep the frequent lookups (by entity, by stage/time) and retention
    # pruning fast as the table grows. Additive; safe to re-run.
    "CREATE INDEX IF NOT EXISTS idx_decisions_entity ON decisions (entity_type, entity_id, stage)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_stage_created ON decisions (stage, created_at)",
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
