from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsroom.analyze.ai_accent import load_ai_accent
from newsroom.analyze.stoplist import load_stoplist
from newsroom.db import make_session_factory
from newsroom.editorial import DraftContent, EditorialPipeline, produce_drafts
from newsroom.models import Event, Publication

pytestmark = pytest.mark.pg

CONFIG = Path(__file__).resolve().parents[1] / "config"
STOP = load_stoplist(CONFIG / "stoplist.yaml")
ACCENT = load_ai_accent(CONFIG / "ai_accent.yaml")


class FakeGenerator:
    model = "fake-gen"

    def generate(self, ctx, *, feedback=None):
        return DraftContent(headline=ctx.title or "Новина", lead="Суть події тут.",
                            what_it_means="Наслідок.")


def _event(pg_engine, *, status: str, title: str, update_type: str | None = None) -> int:
    with Session(pg_engine) as s:
        ev = Event(status=status, rubric="politics", title=title, update_type=update_type,
                   first_seen_at=dt.datetime.now(dt.timezone.utc))
        s.add(ev)
        s.flush()
        eid = ev.id
        s.commit()
        return eid


def test_produce_drafts_only_publishable_and_idempotent(pg_engine):
    sf = make_session_factory(pg_engine)
    e_conf = _event(pg_engine, status="confirmed", title="Підтверджена подія")
    e_rep = _event(pg_engine, status="reported", title="Повідомляють подія")
    e_sig = _event(pg_engine, status="signal", title="Сигнал")     # not publishable

    pipe = EditorialPipeline(sf, generator=FakeGenerator(), stoplist_rules=STOP, ai_accent_patterns=ACCENT)

    stats = produce_drafts(sf, pipe, limit=50)
    assert stats["produced"] == 2 and stats["ok"] == 2

    with Session(pg_engine) as s:
        pub_events = set(s.execute(select(Publication.event_id)).scalars().all())
        assert e_conf in pub_events and e_rep in pub_events and e_sig not in pub_events
        assert s.scalar(select(func.count()).select_from(Publication)) == 2

    # second tick: events already have drafts -> nothing produced
    assert produce_drafts(sf, pipe, limit=50)["produced"] == 0


def test_produce_drafts_skips_summary_only_update_types(pg_engine):
    sf = make_session_factory(pg_engine)
    e_new = _event(pg_engine, status="confirmed", title="Новий факт", update_type="new_fact")
    e_ref = _event(pg_engine, status="confirmed", title="Спростування", update_type="refutation")
    e_cons = _event(pg_engine, status="confirmed", title="Наслідок", update_type="consequence")
    e_conf = _event(pg_engine, status="confirmed", title="Підтвердження", update_type="confirmation")
    e_react = _event(pg_engine, status="confirmed", title="Реакція", update_type="reaction")
    e_minor = _event(pg_engine, status="confirmed", title="Дрібниця", update_type="minor")
    e_unclassified = _event(pg_engine, status="confirmed", title="Без класифікації")

    pipe = EditorialPipeline(sf, generator=FakeGenerator(), stoplist_rules=STOP, ai_accent_patterns=ACCENT)
    produce_drafts(sf, pipe, limit=50)

    with Session(pg_engine) as s:
        drafted = set(s.execute(select(Publication.event_id)).scalars().all())
    # postworthy + unclassified are drafted; summary-only classifications are skipped
    assert {e_new, e_ref, e_cons, e_unclassified} <= drafted
    assert not ({e_conf, e_react, e_minor} & drafted)
