from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.orm import Session

from newsroom.editorial.dedup import dedup_pending, parse_groups


# --- parse_groups (offline) ---------------------------------------------------

def test_parse_groups_keeps_valid_pairs():
    raw = '{"groups": [[1, 2], [3]]}'
    assert parse_groups(raw, {1, 2, 3}) == [[1, 2]]      # singleton dropped


def test_parse_groups_drops_unknown_ids_and_bad_json():
    assert parse_groups('{"groups": [[1, 99]]}', {1, 2}) == []   # 99 unknown -> group of 1 -> dropped
    assert parse_groups("garbage", {1, 2}) == []
    assert parse_groups(None, {1}) == []


# --- dedup_pending (pg) -------------------------------------------------------

class FakeGrouper:
    model = "fake-dedup"

    def __init__(self, groups):
        self._groups = groups
        self.seen: list[int] = []

    def group(self, events):
        self.seen = [eid for eid, _ in events]
        return self._groups


def _event(pg_engine, title):
    from newsroom.models import Event

    with Session(pg_engine) as s:
        ev = Event(status="confirmed", rubric="politics", title=title,
                   first_seen_at=dt.datetime.now(dt.timezone.utc))
        s.add(ev)
        s.flush()
        eid = ev.id
        s.commit()
        return eid


@pytest.mark.pg
def test_dedup_marks_non_canonical_duplicates(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event

    sf = make_session_factory(pg_engine)
    a = _event(pg_engine, "Переговори — джерело A")
    b = _event(pg_engine, "Переговори — джерело B")
    c = _event(pg_engine, "Зовсім інша подія")

    grouper = FakeGrouper([[a, b]])          # a and b are the same story
    stats = dedup_pending(sf, grouper, window_hours=48)
    assert stats["duplicates"] == 1 and stats["groups"] == 1

    with Session(pg_engine) as s:
        assert s.get(Event, a).duplicate_of is None          # earliest = canonical
        assert s.get(Event, b).duplicate_of == a             # marked duplicate of a
        assert s.get(Event, c).duplicate_of is None          # untouched

    # idempotent: re-running doesn't double-mark
    assert dedup_pending(sf, grouper, window_hours=48)["duplicates"] == 0


@pytest.mark.pg
def test_dedup_no_call_when_fewer_than_two(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    _event(pg_engine, "Одна подія")
    grouper = FakeGrouper([[1, 2]])
    stats = dedup_pending(sf, grouper, window_hours=48)
    assert stats["duplicates"] == 0 and grouper.seen == []    # grouper not invoked


@pytest.mark.pg
def test_duplicate_event_is_not_drafted(pg_engine):
    from pathlib import Path

    from newsroom.analyze.ai_accent import load_ai_accent
    from newsroom.analyze.stoplist import load_stoplist
    from newsroom.db import make_session_factory
    from newsroom.editorial import DraftContent, EditorialPipeline, produce_drafts
    from newsroom.models import Event, Publication

    CONFIG = Path(__file__).resolve().parents[1] / "config"
    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        canon = Event(status="confirmed", rubric="politics", title="Канонічна", first_seen_at=now)
        s.add(canon)
        s.flush()
        dup = Event(status="confirmed", rubric="politics", title="Дубль",
                    duplicate_of=canon.id, first_seen_at=now)
        s.add(dup)
        s.flush()
        canon_id, dup_id = canon.id, dup.id
        s.commit()

    class _Gen:
        model = "g"

        def generate(self, ctx, *, feedback=None):
            return DraftContent(headline=ctx.title or "x", body="Конкретний текст події.")

    pipe = EditorialPipeline(sf, generator=_Gen(),
                             stoplist_rules=load_stoplist(CONFIG / "stoplist.yaml"),
                             ai_accent_patterns=load_ai_accent(CONFIG / "ai_accent.yaml"))
    produce_drafts(sf, pipe, limit=50)

    from sqlalchemy import select
    with Session(pg_engine) as s:
        drafted = set(s.execute(select(Publication.event_id)).scalars().all())
        assert canon_id in drafted and dup_id not in drafted   # duplicate skipped
