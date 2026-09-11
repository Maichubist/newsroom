from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsroom.reputation import (
    REP_CONFIRMED,
    REP_COPY,
    REP_FIRST,
    REP_LEAK_REFUTED,
    REP_REFUTED,
    record_event_origins,
    record_event_outcome,
    record_pending,
    record_reputation,
    reputation_score,
    score_from_counts,
)

UTC = dt.timezone.utc


# --- score_from_counts (offline) ----------------------------------------------

def test_score_rewards_first_and_confirmed_penalises_refuted():
    assert score_from_counts({REP_FIRST: 2, REP_CONFIRMED: 1}) == pytest.approx(2 * 2.0 + 3.0)
    assert score_from_counts({REP_REFUTED: 1}) == pytest.approx(-4.0)
    assert score_from_counts({REP_COPY: 4}) == pytest.approx(-2.0)
    assert score_from_counts({}) == 0.0
    assert score_from_counts({"unknown_kind": 5}) == 0.0   # unknown kinds ignored


# --- record_reputation dedup + origins/outcome/score (pg) ---------------------

def _source(s, handle, *, tier="media"):
    from newsroom.models import Source

    src = Source(kind="rss", handle_or_url=handle, name=handle, origin="ua", tier=tier)
    s.add(src)
    s.flush()
    return src.id


def _event(s, *, status, first_source_id=None):
    from newsroom.models import Event

    ev = Event(status=status, title="e", first_source_id=first_source_id,
               first_seen_at=dt.datetime.now(UTC))
    s.add(ev)
    s.flush()
    return ev.id


@pytest.mark.pg
def test_record_reputation_dedups(pg_engine):
    from newsroom.models import ReputationEvent

    with Session(pg_engine) as s:
        src = _source(s, "https://r/1")
        ev = _event(s, status="confirmed")
        first = record_reputation(s, src, REP_FIRST, event_id=ev)
        dup = record_reputation(s, src, REP_FIRST, event_id=ev)   # same fact -> None
        s.commit()
        assert first is not None and dup is None
        assert s.scalar(select(func.count()).select_from(ReputationEvent)
                        .where(ReputationEvent.source_id == src)) == 1


@pytest.mark.pg
def test_record_origins_credits_first_and_copy(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import EventItem, Item, ReputationEvent

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        origin_src = _source(s, "https://r/origin")
        copy_src = _source(s, "https://r/copy")
        ev = _event(s, status="reported")
        oi = Item(source_id=origin_src, external_id="o", content_hash="o", title="o")
        ci = Item(source_id=copy_src, external_id="c", content_hash="c", title="c")
        s.add_all([oi, ci])
        s.flush()
        s.add(EventItem(event_id=ev, item_id=oi.id, role="origin"))
        s.add(EventItem(event_id=ev, item_id=ci.id, role="copy"))
        s.commit()
        event_id, origin_id, copy_id = ev, origin_src, copy_src

    counts = record_event_origins(sf, event_id)
    assert counts == {"first": 1, "copy": 1}
    with Session(pg_engine) as s:
        kinds = {(r.source_id, r.kind) for r in
                 s.execute(select(ReputationEvent).where(ReputationEvent.event_id == event_id)).scalars()}
        assert (origin_id, REP_FIRST) in kinds and (copy_id, REP_COPY) in kinds

    # idempotent
    assert record_event_origins(sf, event_id) == {"first": 0, "copy": 0}


@pytest.mark.pg
def test_record_outcome_and_score(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = _source(s, "https://r/outcome")
        ev = _event(s, status="confirmed", first_source_id=src)
        s.commit()
        event_id, src_id = ev, src

    assert record_event_outcome(sf, event_id) == REP_CONFIRMED
    assert record_event_outcome(sf, event_id) is None   # already recorded

    with Session(pg_engine) as s:
        result = reputation_score(s, src_id)
        assert result["counts"].get(REP_CONFIRMED) == 1
        assert result["score"] == pytest.approx(3.0)


@pytest.mark.pg
def test_leak_source_gets_leak_refuted(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = _source(s, "https://r/leak", tier="leak")
        ev = _event(s, status="refuted", first_source_id=src)
        s.commit()
        event_id = ev

    assert record_event_outcome(sf, event_id) == REP_LEAK_REFUTED


@pytest.mark.pg
def test_record_pending_processes_origins_and_outcomes(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import EventItem, Item, ReputationEvent

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = _source(s, "https://r/pending")
        confirmed = _event(s, status="confirmed", first_source_id=src)
        signal = _event(s, status="signal", first_source_id=src)  # not newsworthy -> ignored
        # give the confirmed event an origin item too (belt and suspenders)
        it = Item(source_id=src, external_id="p", content_hash="p", title="p")
        s.add(it)
        s.flush()
        s.add(EventItem(event_id=confirmed, item_id=it.id, role="origin"))
        s.commit()
        confirmed_id, signal_id = confirmed, signal

    stats = record_pending(sf, limit=50)
    assert stats["origins"] >= 1 and stats["outcomes"] == 1

    with Session(pg_engine) as s:
        # the signal event was never credited
        assert s.scalar(select(func.count()).select_from(ReputationEvent)
                        .where(ReputationEvent.event_id == signal_id)) == 0
        # the confirmed event has both a first and a confirmed
        kinds = {r.kind for r in s.execute(
            select(ReputationEvent).where(ReputationEvent.event_id == confirmed_id)).scalars()}
        assert REP_FIRST in kinds and REP_CONFIRMED in kinds
