from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.collectors.ingest import upsert_raw_item
from newsroom.collectors.telegram import (
    backfill_cursor,
    mark_item_deleted,
    message_to_raw_item,
)
from newsroom.models import Item, ItemVersion, Source

pytestmark = pytest.mark.pg
UTC = dt.timezone.utc


def _tg_source(session: Session, handle: str) -> int:
    src = Source(kind="telegram", handle_or_url=handle, name="Chan", origin="ua", tier="media")
    session.add(src)
    session.flush()
    return src.id


def _msg(**kw):
    return SimpleNamespace(**kw)


def test_persist_edit_and_delete(pg_engine):
    with Session(pg_engine) as s:
        sid = _tg_source(s, "@editchan")
        raw = message_to_raw_item(_msg(id=100, message="original", date=dt.datetime(2026, 9, 10, 9, 0, tzinfo=UTC)), source_id=sid)
        item, created = upsert_raw_item(s, raw)
        s.commit()
        assert created is True

        # edit: same message id, new text -> version + edited_at (§12)
        edited = message_to_raw_item(_msg(id=100, message="edited text", date=dt.datetime(2026, 9, 10, 9, 5, tzinfo=UTC)), source_id=sid)
        item2, created2 = upsert_raw_item(s, edited)
        s.commit()
        assert created2 is False and item2.text == "edited text" and item2.edited_at is not None
        versions = s.scalars(select(ItemVersion).where(ItemVersion.item_id == item2.id)).all()
        assert len(versions) == 1 and versions[0].text == "original"

        # delete -> deleted_at
        assert mark_item_deleted(s, sid, "100") is True
        s.commit()
        assert s.get(Item, item2.id).deleted_at is not None
        # unknown id is a no-op
        assert mark_item_deleted(s, sid, "999") is False


def test_backfill_cursor_is_max_message_id(pg_engine):
    with Session(pg_engine) as s:
        sid = _tg_source(s, "@backfill")
        assert backfill_cursor(s, sid) == 0            # never collected
        for mid in (5, 10, 11):
            upsert_raw_item(s, message_to_raw_item(_msg(id=mid, message=f"m{mid}"), source_id=sid))
        s.commit()
        assert backfill_cursor(s, sid) == 11           # resume above the newest stored id


def test_album_persists_as_one_item(pg_engine):
    from newsroom.collectors.telegram import merge_album
    from newsroom.models import MediaAsset

    with Session(pg_engine) as s:
        sid = _tg_source(s, "@album")
        gid = 42
        members = [
            message_to_raw_item(_msg(id=200, grouped_id=gid, photo=object()), source_id=sid),
            message_to_raw_item(_msg(id=201, grouped_id=gid, message="cap", photo=object()), source_id=sid),
        ]
        item, created = upsert_raw_item(s, merge_album(members))
        s.commit()
        assert created is True and item.external_id == "200" and item.text == "cap"
        media = s.scalars(select(MediaAsset).where(MediaAsset.item_id == item.id)).all()
        assert len(media) == 2 and all(m.storage_key is None for m in media)  # not downloaded in 1a
