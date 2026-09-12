from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from newsroom.analyze.stories import StoryLinker, link_pending, mark_dormant, slugify
from newsroom.db import make_session_factory
from newsroom.db.base import EMBEDDING_DIM
from newsroom.models import Event, Story, StoryVersion

pytestmark = pytest.mark.pg


def _vec(*idx):
    v = [0.0] * EMBEDDING_DIM
    for i in idx:
        v[i] = 1.0
    return v


def _event(pg_engine, vec, *, title="Подія", rubric="politics") -> int:
    with Session(pg_engine) as s:
        ev = Event(centroid=vec, status="confirmed", title=title, rubric=rubric,
                   first_seen_at=dt.datetime.now(dt.timezone.utc))
        s.add(ev)
        s.flush()
        eid = ev.id
        s.commit()
        return eid


# --- slugify (offline-ish, pure) ----------------------------------------------

def test_slugify_basic():
    assert slugify("Іран та США: переговори!") == "іран-та-сша-переговори"
    assert slugify("") == "story"


# --- linking / lifecycle ------------------------------------------------------

def test_first_event_opens_story(pg_engine):
    sf = make_session_factory(pg_engine)
    eid = _event(pg_engine, _vec(1, 2, 3), title="Іран")
    r = StoryLinker(sf).assign(eid)
    assert r.created_new is True and r.version == 1
    with Session(pg_engine) as s:
        ev = s.get(Event, eid)
        story = s.get(Story, r.story_id)
        assert ev.story_id == r.story_id
        assert story.state == "new" and story.slug.endswith(f"-{story.id}")
        assert story.hashtag is None                     # < 3 events


def test_similar_event_joins_and_advances_state(pg_engine):
    sf = make_session_factory(pg_engine)
    linker = StoryLinker(sf, threshold=0.80)
    e1 = _event(pg_engine, _vec(4, 5, 6))
    e2 = _event(pg_engine, _vec(4, 5, 6))            # identical -> joins
    r1 = linker.assign(e1)
    r2 = linker.assign(e2)
    assert r2.created_new is False and r2.story_id == r1.story_id and r2.version == 2
    with Session(pg_engine) as s:
        assert s.get(Story, r1.story_id).state == "developing"
        n = s.scalar(select(func.count()).select_from(Event).where(Event.story_id == r1.story_id))
        assert n == 2


def test_dissimilar_event_opens_new_story(pg_engine):
    sf = make_session_factory(pg_engine)
    linker = StoryLinker(sf, threshold=0.80)
    r1 = linker.assign(_event(pg_engine, _vec(7, 8)))
    r2 = linker.assign(_event(pg_engine, _vec(200, 201)))
    assert r2.story_id != r1.story_id and r2.created_new is True


def test_hashtag_appears_after_three_events(pg_engine):
    sf = make_session_factory(pg_engine)
    linker = StoryLinker(sf, threshold=0.80)
    story_id = None
    for _ in range(3):
        r = linker.assign(_event(pg_engine, _vec(9, 10, 11), rubric="politics"))
        story_id = r.story_id
    with Session(pg_engine) as s:
        assert s.get(Story, story_id).hashtag == "#politics"


def test_link_pending_links_unlinked_and_merges_similar(pg_engine):
    sf = make_session_factory(pg_engine)
    # two near-identical events (should share one story) + one distinct event
    _event(pg_engine, _vec(30, 31, 32), title="Матч А")
    _event(pg_engine, _vec(30, 31, 32), title="Матч А (передрук)")
    _event(pg_engine, _vec(150, 151), title="Інша тема")

    stats = link_pending(sf, StoryLinker(sf, threshold=0.80), limit=50)
    assert stats["linked"] == 3 and stats["new_stories"] == 2   # two similar merged into one story

    with Session(pg_engine) as s:
        assert s.scalar(select(func.count()).select_from(Event).where(Event.story_id.is_(None))) == 0
        assert s.scalar(select(func.count()).select_from(Story)) == 2

    # idempotent: nothing left to link on a second tick
    assert link_pending(sf, StoryLinker(sf), limit=50)["linked"] == 0


def test_mark_dormant(pg_engine):
    sf = make_session_factory(pg_engine)
    r = StoryLinker(sf).assign(_event(pg_engine, _vec(12, 13)))
    with Session(pg_engine) as s:
        old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)
        s.execute(text("UPDATE stories SET last_event_at = :ts WHERE id = :id"), {"ts": old, "id": r.story_id})
        s.commit()
    assert mark_dormant(sf, dormant_days=5) >= 1
    with Session(pg_engine) as s:
        assert s.get(Story, r.story_id).state == "dormant"
