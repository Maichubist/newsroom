from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsroom.collectors.base import RawItem, RawMedia
from newsroom.collectors.ingest import (
    mark_source_failure,
    mark_source_success,
    upsert_raw_item,
)
from newsroom.config.sources import SourceConfig
from newsroom.models import Item, ItemVersion, MediaAsset, Source
from newsroom.sources.registry import health_report, sync_sources

pytestmark = pytest.mark.pg


def _make_source(session: Session, handle: str) -> int:
    src = Source(kind="rss", handle_or_url=handle, name="S", origin="ua", tier="media")
    session.add(src)
    session.flush()
    return src.id


def test_upsert_is_idempotent(pg_engine):
    with Session(pg_engine) as s:
        sid = _make_source(s, "https://idem/rss")
        raw = RawItem(source_id=sid, external_id="e1", title="T", text="body")
        _, created1 = upsert_raw_item(s, raw)
        _, created2 = upsert_raw_item(s, raw)   # same content -> no dup, no version
        s.commit()

        assert created1 is True and created2 is False
        n = s.scalar(select(func.count()).select_from(Item).where(Item.source_id == sid, Item.external_id == "e1"))
        assert n == 1
        versions = s.scalars(select(ItemVersion)).all()
        assert versions == []


def test_edit_creates_version_and_marks_edited(pg_engine):
    with Session(pg_engine) as s:
        sid = _make_source(s, "https://edit/rss")
        upsert_raw_item(s, RawItem(source_id=sid, external_id="e1", title="T", text="old body"))
        s.commit()

        item, created = upsert_raw_item(s, RawItem(source_id=sid, external_id="e1", title="T", text="new body"))
        s.commit()

        assert created is False
        assert item.text == "new body"
        assert item.edited_at is not None
        versions = s.scalars(select(ItemVersion).where(ItemVersion.item_id == item.id)).all()
        assert len(versions) == 1 and versions[0].text == "old body"


def test_media_persisted_on_create(pg_engine):
    with Session(pg_engine) as s:
        sid = _make_source(s, "https://media/rss")
        raw = RawItem(source_id=sid, external_id="m1", title="T", text="b",
                      media=[RawMedia(kind="image", url="https://x/y.jpg", size_bytes=99)])
        item, _ = upsert_raw_item(s, raw)
        s.commit()
        media = s.scalars(select(MediaAsset).where(MediaAsset.item_id == item.id)).all()
        assert len(media) == 1 and media[0].kind == "image" and media[0].storage_key is None


def test_source_health_transitions(pg_engine):
    with Session(pg_engine) as s:
        sid = _make_source(s, "https://health/rss")
        src = s.get(Source, sid)
        mark_source_failure(src, "boom")
        mark_source_failure(src, "boom again")
        s.commit()
        assert s.get(Source, sid).consecutive_failures == 2

        mark_source_success(s.get(Source, sid))
        s.commit()
        src = s.get(Source, sid)
        assert src.consecutive_failures == 0 and src.last_error is None and src.last_success_at is not None


def test_sync_sources_upsert_and_health_preserved(pg_engine):
    cfg = SourceConfig(kind="rss", handle_or_url="https://sync/rss", name="Old",
                       origin="ua", tier="media")
    with Session(pg_engine) as s:
        r1 = sync_sources(s, [cfg])
        s.commit()
        sid = s.scalar(select(Source.id).where(Source.handle_or_url == "https://sync/rss"))
        mark_source_failure(s.get(Source, sid), "err")
        s.commit()

    # re-sync with a renamed source: descriptive field updates, health untouched
    cfg2 = SourceConfig(kind="rss", handle_or_url="https://sync/rss", name="New name",
                        origin="ua", tier="official", is_official=True)
    with Session(pg_engine) as s:
        r2 = sync_sources(s, [cfg2])
        s.commit()
        src = s.get(Source, sid)
        assert r1 == {"created": 1, "updated": 0}
        assert r2 == {"created": 0, "updated": 1}
        assert src.name == "New name" and src.tier == "official" and src.is_official is True
        assert src.consecutive_failures == 1  # health survived the config sync


def test_health_report_shape(pg_engine):
    with Session(pg_engine) as s:
        _make_source(s, "https://report/rss")
        s.commit()
        report = health_report(s)
        assert isinstance(report, list) and report
        row = report[0]
        assert {"name", "healthy", "consecutive_failures", "last_success_at"} <= set(row)
