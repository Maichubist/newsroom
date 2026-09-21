from __future__ import annotations

import asyncio
import datetime as dt
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.collectors.ingest import upsert_raw_item
from newsroom.collectors.telegram import TelegramCollector, message_to_raw_item
from newsroom.db import make_session_factory
from newsroom.models import Item, ItemVersion, MediaAsset, Source

pytestmark = pytest.mark.pg
UTC = dt.timezone.utc


def _msg(**kw):
    return SimpleNamespace(**kw)


def _tg_source(session: Session, handle: str) -> int:
    src = Source(kind="telegram", handle_or_url=handle, name="C", origin="ua", tier="media")
    session.add(src)
    session.flush()
    return src.id


class FakeClient:
    """Minimal stand-in for a Telethon client's iter_messages (backfill)."""

    def __init__(self, messages, flood_once: BaseException | None = None):
        self._messages = messages
        self._flood_once = flood_once
        self.calls = 0

    def iter_messages(self, peer, min_id=0, reverse=True, limit=None):
        self.calls += 1
        flood = self._flood_once if (self.calls == 1 and self._flood_once) else None
        msgs = [m for m in sorted(self._messages, key=lambda x: x.id) if m.id > min_id]
        if not reverse:            # Telethon default: newest-first
            msgs = list(reversed(msgs))
        if limit is not None:
            msgs = msgs[:limit]

        async def gen():
            if flood is not None:
                raise flood
            for m in msgs:
                yield m

        return gen()


def test_backfill_ingests_only_above_cursor(pg_engine):
    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        sid = _tg_source(s, "@bf")
        upsert_raw_item(s, message_to_raw_item(_msg(id=10, message="old"), sid))  # cursor -> 10
        s.commit()

    client = FakeClient([_msg(id=9, message="below"), _msg(id=11, message="a"), _msg(id=12, message="b")])
    collector = TelegramCollector(sf)
    n = asyncio.run(collector.backfill_source(client, sid, peer="x", channel_username="bf"))

    assert n == 2  # only 11 and 12 (id > 10)
    with Session(pg_engine) as s:
        ids = set(s.scalars(select(Item.external_id).where(Item.source_id == sid)).all())
        assert ids == {"10", "11", "12"}


def test_fresh_channel_seeds_only_recent_not_whole_history(pg_engine):
    # a never-seen channel (cursor 0) must ingest only the most recent N posts,
    # not its entire history — otherwise a busy aggregator floods the pipeline
    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        sid = _tg_source(s, "@huge")
        s.commit()

    history = [_msg(id=i, message=f"m{i}") for i in range(1, 101)]   # 100 posts in history
    client = FakeClient(history)
    collector = TelegramCollector(sf)
    n = asyncio.run(collector.backfill_source(client, sid, peer="x", first_seed_limit=5))

    assert n == 5
    with Session(pg_engine) as s:
        ids = sorted(int(x) for x in s.scalars(select(Item.external_id).where(Item.source_id == sid)).all())
        assert ids == [96, 97, 98, 99, 100]     # only the 5 most recent


def test_backfill_drops_messages_older_than_window(pg_engine):
    # only posts within the 24h window are ingested, whatever the source
    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        sid = _tg_source(s, "@aged")
        s.commit()

    now = dt.datetime.now(UTC)
    msgs = [
        _msg(id=1, message="old", date=now - dt.timedelta(hours=48)),
        _msg(id=2, message="fresh", date=now - dt.timedelta(hours=2)),
        _msg(id=3, message="edge-old", date=now - dt.timedelta(hours=30)),
    ]
    collector = TelegramCollector(sf)
    n = asyncio.run(collector.backfill_source(FakeClient(msgs), sid, peer="x"))

    assert n == 1
    with Session(pg_engine) as s:
        ids = set(s.scalars(select(Item.external_id).where(Item.source_id == sid)).all())
        assert ids == {"2"}     # only the fresh post


def test_backfill_waits_out_floodwait(pg_engine):
    class FloodWaitError(Exception):
        def __init__(self, seconds):
            self.seconds = seconds

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        sid = _tg_source(s, "@flood")
        s.commit()

    slept: list[float] = []

    async def sleeper(sec):
        slept.append(sec)

    client = FakeClient([_msg(id=1, message="x")], flood_once=FloodWaitError(3))
    collector = TelegramCollector(sf, floodwait_sleeper=sleeper)
    n = asyncio.run(collector.backfill_source(client, sid, peer="x"))

    assert n == 1 and slept == [3] and client.calls == 2  # retried after waiting


def test_realtime_new_message_and_album(pg_engine):
    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        sid = _tg_source(s, "@rt")
        s.commit()

    collector = TelegramCollector(sf)
    collector.on_new_message(sid, _msg(id=50, message="solo"))            # persisted at once
    collector.on_new_message(sid, _msg(id=60, grouped_id=9, photo=object()))
    collector.on_new_message(sid, _msg(id=61, grouped_id=9, message="cap", photo=object()))

    # album still buffering: only the solo post is in the DB so far
    with Session(pg_engine) as s:
        assert set(s.scalars(select(Item.external_id).where(Item.source_id == sid)).all()) == {"50"}

    flushed = collector.flush_albums(now=dt.datetime.now(UTC) + dt.timedelta(seconds=5))
    assert len(flushed) == 1

    with Session(pg_engine) as s:
        ids = set(s.scalars(select(Item.external_id).where(Item.source_id == sid)).all())
        assert ids == {"50", "60"}                                        # album id = earliest member
        album = s.execute(select(Item).where(Item.source_id == sid, Item.external_id == "60")).scalar_one()
        assert album.text == "cap"
        media = s.scalars(select(MediaAsset).where(MediaAsset.item_id == album.id)).all()
        assert len(media) == 2


class FakeLiveClient(FakeClient):
    """FakeClient + get_entity, for the live re-subscribe/backfill path."""

    def __init__(self, messages, *, resolvable=None):
        super().__init__(messages)
        self._resolvable = resolvable          # set of handles that resolve; None = all
        self.entity_calls: list[str] = []

    async def get_entity(self, handle):
        self.entity_calls.append(handle)
        if self._resolvable is not None and handle not in self._resolvable:
            raise RuntimeError(f"cannot resolve {handle}")
        return handle


def test_sync_and_backfill_new_picks_up_added_channels_without_restart(pg_engine):
    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        _tg_source(s, "@a")
        _tg_source(s, "@b")
        s.commit()

    collector = TelegramCollector(sf)
    client = FakeLiveClient([_msg(id=1, message="x")])
    n = asyncio.run(collector.sync_and_backfill_new(client, first_seed_limit=5))
    assert n == 2                                          # both initial channels backfilled
    assert set(collector._by_username) == {"a", "b"}       # live map seeded
    assert len(collector._backfilled) == 2

    with Session(pg_engine) as s:                          # a channel added while "running"
        _tg_source(s, "@c")
        s.commit()

    n2 = asyncio.run(collector.sync_and_backfill_new(client, first_seed_limit=5))
    assert n2 == 1                                         # only the new one is backfilled
    assert set(collector._by_username) == {"a", "b", "c"}  # and it now streams (in the map)


def test_sync_and_backfill_new_retries_unresolvable_channel(pg_engine):
    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        _tg_source(s, "@ok")
        _tg_source(s, "@bad")
        s.commit()

    collector = TelegramCollector(sf)
    n = asyncio.run(collector.sync_and_backfill_new(
        FakeLiveClient([_msg(id=1, message="x")], resolvable={"@ok"}), first_seed_limit=5))
    assert n == 1                                          # @bad could not resolve -> skipped
    assert set(collector._by_username) == {"ok", "bad"}    # but still streamable via the map

    # a later sync retries @bad (not permanently given up) — now it resolves
    n2 = asyncio.run(collector.sync_and_backfill_new(
        FakeLiveClient([_msg(id=2, message="y")], resolvable={"@ok", "@bad"}), first_seed_limit=5))
    assert n2 == 1                                         # @bad backfilled on retry; @ok skipped (done)


def test_realtime_edit_and_delete(pg_engine):
    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        sid = _tg_source(s, "@ed")
        s.commit()

    collector = TelegramCollector(sf)
    collector.on_new_message(sid, _msg(id=70, message="v1"))
    collector.on_edit(sid, _msg(id=70, message="v2"))

    with Session(pg_engine) as s:
        item = s.execute(select(Item).where(Item.source_id == sid, Item.external_id == "70")).scalar_one()
        assert item.text == "v2" and item.edited_at is not None
        assert len(s.scalars(select(ItemVersion).where(ItemVersion.item_id == item.id)).all()) == 1

    assert collector.on_delete(sid, [70, 999]) == 1  # 70 exists, 999 does not
    with Session(pg_engine) as s:
        item = s.execute(select(Item).where(Item.source_id == sid, Item.external_id == "70")).scalar_one()
        assert item.deleted_at is not None
