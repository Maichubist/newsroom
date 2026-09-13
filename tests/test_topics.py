from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.orm import Session

from newsroom.analyze.topics import (
    compute_hot_topics,
    load_hot_topics,
    normalize_keyword,
    refresh_hot_topics,
    topic_heat,
)

UTC = dt.timezone.utc


# --- pure (offline) -----------------------------------------------------------

def test_normalize_keyword():
    assert normalize_keyword("  Дрон  ") == "дрон"
    assert normalize_keyword("Сили   Оборони") == "сили оборони"


def test_topic_heat_picks_hottest_keyword():
    heat = {"дрон": 0.9, "погода": 0.1}
    assert topic_heat(["Дрон", "щось"], heat) == 0.9      # normalized match, max
    assert topic_heat(["невідоме"], heat) == 0.0
    assert topic_heat([], heat) == 0.0
    assert topic_heat(["дрон"], {}) == 0.0


# --- compute_hot_topics (pg) --------------------------------------------------

_SEQ = [0]


def _event(s, keywords, *, n_items=1, source_id=None):
    from newsroom.models import Event, EventItem, Item, Source

    if source_id is None:
        _SEQ[0] += 1
        src = Source(kind="telegram", handle_or_url=f"@c{_SEQ[0]}", name="c",
                     origin="ua", tier="media")
        s.add(src)
        s.flush()
        source_id = src.id
    ev = Event(status="confirmed", rubric="war", title="t", keywords=keywords,
               first_seen_at=dt.datetime.now(UTC))
    s.add(ev)
    s.flush()
    for i in range(n_items):
        it = Item(source_id=source_id, external_id=f"{ev.id}-{i}",
                  content_hash=f"{ev.id}-{i}".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id))
    return ev.id


@pytest.mark.pg
def test_compute_hot_topics_ranks_by_volume(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        # "дрон" appears in 3 events (one carried by 3 channels) -> hottest
        _event(s, ["дрон", "покровськ"], n_items=3)
        _event(s, ["дрон"], n_items=1)
        _event(s, ["дрон", "суми"], n_items=1)
        _event(s, ["вибори"], n_items=1)          # only 1 event -> below min_events
        s.commit()
        result = compute_hot_topics(s, min_events=2)

    topics = {t["topic"]: t for t in result["topics"]}
    assert "дрон" in topics and topics["дрон"]["events"] == 3
    assert topics["дрон"]["posts"] == 5           # 3 + 1 + 1 items
    assert "вибори" not in topics                 # min_events filter
    assert result["heat"]["дрон"] == max(result["heat"].values())


@pytest.mark.pg
def test_refresh_and_load_hot_topics(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        _event(s, ["мобілізація", "тцк"], n_items=2)
        _event(s, ["мобілізація"], n_items=2)
        s.commit()
    result = refresh_hot_topics(sf)
    assert result.get("topics")
    with Session(pg_engine) as s:
        heat = load_hot_topics(s)
    assert "мобілізація" in heat and 0.0 <= heat["мобілізація"] <= 1.0


@pytest.mark.pg
def test_compute_hot_topics_empty_without_keywords(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        s.add(Event(status="confirmed", title="no kw", first_seen_at=dt.datetime.now(UTC)))
        s.commit()
        assert compute_hot_topics(s) == {}
