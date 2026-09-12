from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.orm import Session

from newsroom.analyze.demand import (
    DemandCollector,
    engagement_rate,
    record_item_metric,
    record_source_metric,
    select_items_to_measure,
)
from newsroom.publishers.metrics import MessageStats


# --- engagement_rate (offline) ------------------------------------------------

def test_engagement_rate_normalises_by_subscribers():
    a = engagement_rate(MessageStats(views=100, reactions={"👍": 10}, forwards=5), 1000)
    b = engagement_rate(MessageStats(views=100, reactions={"👍": 10}, forwards=5), 100000)
    assert a > b        # same raw counts, smaller channel -> higher rate


def test_engagement_rate_weights_forwards_above_reactions():
    reacts_only = engagement_rate(MessageStats(reactions={"👍": 10}), 1000)
    fwd_only = engagement_rate(MessageStats(forwards=10), 1000)
    assert fwd_only > reacts_only        # forwards weighted higher (default 2x)


def test_engagement_rate_none_without_subscribers():
    assert engagement_rate(MessageStats(forwards=5), None) is None
    assert engagement_rate(MessageStats(forwards=5), 0) is None


# --- recording + selection (pg) -----------------------------------------------

def _tg_source_with_item(pg_engine, *, handle, external_id, published_at):
    from newsroom.models import Item, Source

    with Session(pg_engine) as s:
        src = Source(kind="telegram", handle_or_url=handle, name=handle, origin="ua", tier="aggregator")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id=external_id,
                  content_hash=(handle + external_id).ljust(64, "0")[:64],
                  title="t", published_at=published_at)
        s.add(it)
        s.flush()
        s.commit()
        return src.id, it.id


@pytest.mark.pg
def test_select_items_to_measure_only_recent_telegram(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Item, Source

    now = dt.datetime.now(dt.timezone.utc)
    sid, recent_id = _tg_source_with_item(pg_engine, handle="@a", external_id="10", published_at=now)
    # an old telegram post -> excluded
    with Session(pg_engine) as s:
        old_src = s.get(Source, sid)
        old = Item(source_id=old_src.id, external_id="9", content_hash="old".ljust(64, "0"),
                   title="old", published_at=now - dt.timedelta(hours=72))
        s.add(old)
        # an RSS item -> excluded (not telegram)
        rss = Source(kind="rss", handle_or_url="https://x/feed", name="R", origin="ua", tier="media")
        s.add(rss)
        s.flush()
        s.add(Item(source_id=rss.id, external_id="r1", content_hash="r1".ljust(64, "0"),
                   title="r", published_at=now))
        s.commit()

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        picked = select_items_to_measure(s, max_age_hours=48, limit=50)
    ids = {i for i, _, _ in picked}
    assert recent_id in ids and len(ids) == 1        # only the recent telegram item


@pytest.mark.pg
def test_collector_records_item_and_source_metrics(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import ItemMetric, SourceMetric

    now = dt.datetime.now(dt.timezone.utc)
    sid, item_id = _tg_source_with_item(pg_engine, handle="@b", external_id="42", published_at=now)
    sf = make_session_factory(pg_engine)

    class FakeSource:
        def message_stats(self, handle, message_id):
            assert handle == "@b" and message_id == "42"
            return MessageStats(views=500, reactions={"🔥": 20}, forwards=8)

        def subscribers(self, handle):
            return 12345

    collector = DemandCollector(sf, FakeSource())
    assert collector.collect_items()["recorded"] == 1
    assert collector.collect_sources()["recorded"] == 1

    with Session(pg_engine) as s:
        im = s.query(ItemMetric).one()
        assert im.item_id == item_id and im.views == 500 and im.forwards == 8
        sm = s.query(SourceMetric).one()
        assert sm.source_id == sid and sm.subscribers == 12345


@pytest.mark.pg
def test_compute_rubric_demand_learns_and_discounts_aggregators(pg_engine):
    from newsroom.analyze.demand import compute_rubric_demand, load_demand, store_demand
    from newsroom.db import make_session_factory
    from newsroom.models import (Event, EventItem, Item, ItemMetric, Source,
                                  SourceMetric)

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)

    def _post(handle, tier, rubric, forwards, subs):
        with Session(pg_engine) as s:
            src = Source(kind="telegram", handle_or_url=handle, name=handle, origin="ua", tier=tier)
            s.add(src)
            s.flush()
            s.add(SourceMetric(source_id=src.id, subscribers=subs))
            it = Item(source_id=src.id, external_id=handle + "1",
                      content_hash=(handle).ljust(64, "0")[:64], title="t", published_at=now)
            s.add(it)
            s.flush()
            ev = Event(status="confirmed", rubric=rubric, title="t", first_seen_at=now)
            s.add(ev)
            s.flush()
            s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
            s.add(ItemMetric(item_id=it.id, views=1000, forwards=forwards))
            s.commit()

    _post("@pol", "media", "politics", forwards=100, subs=1000)   # high engagement
    _post("@spo", "media", "sport", forwards=2, subs=1000)        # low engagement

    with Session(pg_engine) as s:
        demand = compute_rubric_demand(s, window_days=7)
    assert demand["politics"] > demand["sport"]
    assert demand["politics"] == pytest.approx(1.0) and demand["sport"] == pytest.approx(0.0)

    # store/load roundtrip
    with Session(pg_engine) as s:
        store_demand(s, demand)
        s.commit()
    with Session(pg_engine) as s:
        assert load_demand(s)["politics"] == pytest.approx(1.0)


@pytest.mark.pg
def test_compute_rubric_demand_empty_without_subscribers(pg_engine):
    # no source_metrics -> engagement can't be normalised -> no demand learned
    from newsroom.analyze.demand import compute_rubric_demand
    from newsroom.models import Event, EventItem, Item, ItemMetric, Source

    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        src = Source(kind="telegram", handle_or_url="@ns", name="ns", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id="n1", content_hash="n1".ljust(64, "0"),
                  title="t", published_at=now)
        s.add(it)
        s.flush()
        ev = Event(status="confirmed", rubric="politics", title="t", first_seen_at=now)
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
        s.add(ItemMetric(item_id=it.id, forwards=50))
        s.commit()
        assert compute_rubric_demand(s, window_days=7) == {}


@pytest.mark.pg
def test_record_helpers_append_snapshots(pg_engine):
    from newsroom.models import ItemMetric

    now = dt.datetime.now(dt.timezone.utc)
    _sid, item_id = _tg_source_with_item(pg_engine, handle="@c", external_id="1", published_at=now)
    with Session(pg_engine) as s:
        record_item_metric(s, item_id, MessageStats(views=1))
        record_item_metric(s, item_id, MessageStats(views=2))   # time series: two snapshots
        s.commit()
        assert s.query(ItemMetric).filter_by(item_id=item_id).count() == 2
