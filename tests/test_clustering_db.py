from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from newsroom.analyze.clustering import EventClusterer
from newsroom.db import make_session_factory
from newsroom.db.base import EMBEDDING_DIM
from newsroom.models import Event, EventItem, Item, Source

pytestmark = pytest.mark.pg


def _vec(*idx: int) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    for i in idx:
        v[i] = 1.0
    return v


def _make_items(pg_engine, handle: str, n: int) -> list[int]:
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url=handle, name="S", origin="ua", tier="media")
        s.add(src)
        s.flush()
        ids = []
        for i in range(n):
            it = Item(source_id=src.id, external_id=f"{handle}-{i}", content_hash=f"{handle}{i}".ljust(64, "0"))
            s.add(it)
            s.flush()
            ids.append(it.id)
        s.commit()
        return ids


def test_first_item_opens_event(pg_engine):
    (i0,) = _make_items(pg_engine, "clu-open", 1)
    clu = EventClusterer(make_session_factory(pg_engine))
    r = clu.assign(i0, _vec(0, 1, 2))
    assert r.created_new is True
    with Session(pg_engine) as s:
        ei = s.execute(select(EventItem).where(EventItem.item_id == i0)).scalar_one()
        assert ei.event_id == r.event_id and ei.role == "origin"


def test_similar_item_joins_and_updates_centroid(pg_engine):
    i0, i1 = _make_items(pg_engine, "clu-join", 2)
    clu = EventClusterer(make_session_factory(pg_engine), threshold=0.83)
    r0 = clu.assign(i0, _vec(3, 4, 5))
    r1 = clu.assign(i1, _vec(3, 4, 5))       # identical -> cosine 1.0 -> joins
    assert r1.created_new is False and r1.event_id == r0.event_id and r1.similarity == pytest.approx(1.0)
    with Session(pg_engine) as s:
        n = s.scalar(select(func.count()).select_from(EventItem).where(EventItem.event_id == r0.event_id))
        assert n == 2


def test_dissimilar_item_opens_new_event(pg_engine):
    i0, i1 = _make_items(pg_engine, "clu-diff", 2)
    clu = EventClusterer(make_session_factory(pg_engine), threshold=0.83)
    r0 = clu.assign(i0, _vec(6, 7))
    r1 = clu.assign(i1, _vec(100, 101))      # orthogonal -> new event
    assert r1.created_new is True and r1.event_id != r0.event_id


def test_event_outside_window_is_not_joined(pg_engine):
    i0, i1 = _make_items(pg_engine, "clu-window", 2)
    sf = make_session_factory(pg_engine)
    clu = EventClusterer(sf, threshold=0.83, window_hours=48)
    r0 = clu.assign(i0, _vec(8, 9))
    # age the event past the window
    with Session(pg_engine) as s:
        old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=100)
        s.execute(text("UPDATE events SET updated_at = :ts WHERE id = :id"), {"ts": old, "id": r0.event_id})
        s.commit()
    r1 = clu.assign(i1, _vec(8, 9))          # identical vector, but old event is out of window
    assert r1.created_new is True and r1.event_id != r0.event_id
