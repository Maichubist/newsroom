from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.analyze.ai_accent import load_ai_accent
from newsroom.analyze.stoplist import load_stoplist
from newsroom.db import make_session_factory
from newsroom.editorial import DraftContent, EditorialPipeline
from newsroom.models import Decision, Event, EventItem, Item, Publication, Source, Story

pytestmark = pytest.mark.pg

CONFIG = Path(__file__).resolve().parents[1] / "config"
STOP = load_stoplist(CONFIG / "stoplist.yaml")
ACCENT = load_ai_accent(CONFIG / "ai_accent.yaml")


class FakeGenerator:
    model = "fake-gen"

    def __init__(self, drafts):
        self._drafts = list(drafts)
        self.feedback_calls = []

    def generate(self, ctx, *, feedback=None):
        self.feedback_calls.append(feedback)
        return self._drafts.pop(0) if self._drafts else self._drafts_last

    _drafts_last = DraftContent(headline="Fallback", body="Fallback body text.")


def _event(pg_engine, *, rubric="politics", status="confirmed", title="Подія", hashtag=None) -> int:
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        story = Story(slug=f"s-{title}", title=title, state="developing",
                      rubric=rubric, hashtag=hashtag, last_event_at=now)
        s.add(story)
        s.flush()
        ev = Event(status=status, rubric=rubric, title=title, story_id=story.id, first_seen_at=now)
        s.add(ev)
        s.flush()
        src = Source(kind="rss", handle_or_url=f"{title}-src", name="Джерело", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id=f"{title}-1", content_hash=title.ljust(64, "0"))
        s.add(it)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
        s.commit()
        return ev.id


def _pub(pg_engine, event_id):
    with Session(pg_engine) as s:
        return s.execute(select(Publication).where(Publication.event_id == event_id)).scalar_one()


def test_clean_draft_is_stored_and_passes(pg_engine):
    eid = _event(pg_engine, hashtag="#політика")
    gen = FakeGenerator([DraftContent(headline="НБУ знизив ставку",
                                      body="Ставку знижено до 13% — дешевші кредити.")])
    pipe = EditorialPipeline(make_session_factory(pg_engine), generator=gen,
                             stoplist_rules=STOP, ai_accent_patterns=ACCENT)
    r = pipe.produce(eid)

    assert r.critic_ok is True and r.regenerated is False and len(gen.feedback_calls) == 1
    pub = _pub(pg_engine, eid)
    assert pub.status == "draft" and pub.headline == "НБУ знизив ставку"
    assert "Ставку знижено до 13% — дешевші кредити." in pub.body
    assert pub.features["critic_ok"] is True and pub.model == "fake-gen"


def test_ai_accent_triggers_one_regeneration(pg_engine):
    eid = _event(pg_engine)
    gen = FakeGenerator([
        DraftContent(headline="Ставка", body="Таким чином, ставку знижено."),   # soft ai_accent
        DraftContent(headline="Ставка", body="Ставку знижено до 13% сьогодні."),  # clean
    ])
    pipe = EditorialPipeline(make_session_factory(pg_engine), generator=gen,
                             stoplist_rules=STOP, ai_accent_patterns=ACCENT)
    r = pipe.produce(eid)

    assert r.regenerated is True and r.critic_ok is True
    assert len(gen.feedback_calls) == 2 and gen.feedback_calls[1] is not None  # feedback passed on retry


def test_stoplist_block_stored_but_flagged(pg_engine):
    eid = _event(pg_engine, rubric="war", status="confirmed", title="Атака")
    block = DraftContent(headline="Атака", body="Шахеди курсом на Київ увечері.")
    gen = FakeGenerator([block, block])  # still blocked after regenerate
    pipe = EditorialPipeline(make_session_factory(pg_engine), generator=gen,
                             stoplist_rules=STOP, ai_accent_patterns=ACCENT)
    r = pipe.produce(eid)

    assert r.critic_ok is False and any(h.startswith("stoplist_block") for h in r.hard)
    pub = _pub(pg_engine, eid)
    assert pub.status == "draft" and pub.features["critic_ok"] is False  # not publishable


def test_fallback_generation_stores_no_draft_and_holds(pg_engine):
    # when the generator only produces a fallback (e.g. the LLM 429'd), no post is
    # stored — the event stays undrafted for the next tick, and the miss is journaled
    eid = _event(pg_engine, title="Подія без тексту")
    fb = DraftContent(headline="Новина", body="Новина", fallback=True)
    gen = FakeGenerator([fb, fb, fb])
    pipe = EditorialPipeline(make_session_factory(pg_engine), generator=gen,
                             stoplist_rules=STOP, ai_accent_patterns=ACCENT)
    r = pipe.produce(eid)

    assert r.publication_id is None and r.critic_ok is False and "generation_failed" in r.hard
    with Session(pg_engine) as s:
        assert s.execute(select(Publication).where(Publication.event_id == eid)).first() is None
        dec = s.execute(
            select(Decision).where(Decision.entity_id == str(eid), Decision.decision == "generation_failed")
        ).scalar_one()
        assert dec.stage == "edit"


def test_transient_first_failure_recovers_on_retry(pg_engine):
    # first attempt fell back, second attempt is real content -> a draft is stored
    eid = _event(pg_engine, title="Подія що відновилась")
    gen = FakeGenerator([
        DraftContent(headline="X", body="Y", fallback=True),
        DraftContent(headline="Справжній заголовок", body="Справжній конкретний текст події."),
    ])
    pipe = EditorialPipeline(make_session_factory(pg_engine), generator=gen,
                             stoplist_rules=STOP, ai_accent_patterns=ACCENT)
    r = pipe.produce(eid)

    assert r.publication_id is not None
    pub = _pub(pg_engine, eid)
    assert pub.headline == "Справжній заголовок"
