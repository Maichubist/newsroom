from __future__ import annotations

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from newsroom.db.base import EMBEDDING_DIM, init_db
from newsroom.models import Item, ItemEmbedding, Source

pytestmark = pytest.mark.pg


EXPECTED_TABLES = {
    "sources", "items", "item_versions", "media_assets", "item_embeddings",
    "entities", "item_entities", "events", "event_items", "stories",
    "story_versions", "claims", "claim_evidence", "publications",
    "publication_metrics", "channel_metrics", "reputation_events",
    "decisions", "system_state",
}


def test_all_tables_created(pg_engine):
    tables = set(inspect(pg_engine).get_table_names())
    assert EXPECTED_TABLES.issubset(tables)


def test_init_db_is_idempotent(pg_engine):
    # Second run must not raise (clean-slate create_all with checkfirst).
    init_db(pg_engine)
    init_db(pg_engine)


def test_insert_source_and_item_roundtrip(pg_engine):
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://ex/rss", name="Ex",
                     origin="ua", tier="media")
        s.add(src)
        s.flush()
        item = Item(source_id=src.id, external_id="ext-1", title="T",
                    content_hash="a" * 64, status="new")
        s.add(item)
        s.commit()

        got = s.execute(
            select(Item).where(Item.source_id == src.id, Item.external_id == "ext-1")
        ).scalar_one()
        assert got.title == "T"
        assert got.fetched_at is not None  # server_default fired


def test_unique_source_external_id(pg_engine):
    from sqlalchemy.exc import IntegrityError

    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://dup/rss", name="Dup",
                     origin="ua", tier="media")
        s.add(src)
        s.flush()
        s.add(Item(source_id=src.id, external_id="same", content_hash="b" * 64))
        s.add(Item(source_id=src.id, external_id="same", content_hash="c" * 64))
        with pytest.raises(IntegrityError):
            s.commit()


def test_pgvector_roundtrip_and_distance(pg_engine):
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://vec/rss", name="Vec",
                     origin="ua", tier="media")
        s.add(src)
        s.flush()
        item = Item(source_id=src.id, external_id="v-1", content_hash="d" * 64)
        s.add(item)
        s.flush()
        vec = [0.0] * EMBEDDING_DIM
        vec[0] = 1.0
        s.add(ItemEmbedding(item_id=item.id, model="text-embedding-3-small", vector=vec))
        s.commit()

        # cosine distance to itself is ~0 — proves the vector extension works.
        dist = s.execute(
            text("SELECT vector <=> :q FROM item_embeddings WHERE item_id = :iid"),
            {"q": str(vec), "iid": item.id},
        ).scalar_one()
        assert dist == pytest.approx(0.0, abs=1e-6)
