from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsroom.publishers.metrics import MessageStats, MetricsCollector

UTC = dt.timezone.utc


# --- MessageStats (offline) ---------------------------------------------------

def test_message_stats_is_empty():
    assert MessageStats().is_empty() is True
    assert MessageStats(views=0).is_empty() is False        # a zero view count is data
    assert MessageStats(reactions={"👍": 3}).is_empty() is False


class FakeSource:
    """Serves canned stats keyed by channel_ref; counts subscriber reads."""

    def __init__(self, by_ref, subscribers=1234):
        self._by_ref = by_ref
        self._subscribers = subscribers
        self.sub_reads = 0

    def message_stats(self, channel_ref):
        return self._by_ref.get(channel_ref)

    def channel_subscribers(self, channel):
        self.sub_reads += 1
        return self._subscribers


def _published(pg_engine, *, channel_ref, when=None, channel="telegram", status="published"):
    from newsroom.models import Publication

    with Session(pg_engine) as s:
        pub = Publication(channel=channel, kind="post", status=status, headline="h", body="b",
                          channel_ref=channel_ref,
                          published_at=(when or dt.datetime.now(UTC)))
        s.add(pub)
        s.flush()
        pid = pub.id
        s.commit()
        return pid


# --- MetricsCollector (pg) ----------------------------------------------------

@pytest.mark.pg
def test_collect_publications_appends_snapshots(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import PublicationMetric

    sf = make_session_factory(pg_engine)
    pid = _published(pg_engine, channel_ref="42")
    source = FakeSource({"42": MessageStats(views=100, reactions={"👍": 5}, forwards=2)})
    collector = MetricsCollector(sf, source)

    stats = collector.collect_publications(limit=50)
    assert stats["polled"] == 1 and stats["recorded"] == 1
    # a second tick appends another snapshot (time series, no dedup)
    collector.collect_publications(limit=50)

    with Session(pg_engine) as s:
        rows = s.execute(select(PublicationMetric).where(PublicationMetric.publication_id == pid)
                         .order_by(PublicationMetric.id)).scalars().all()
        assert len(rows) == 2
        assert rows[0].views == 100 and rows[0].forwards == 2 and rows[0].reactions == {"👍": 5}


@pytest.mark.pg
def test_collect_skips_empty_and_old(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import PublicationMetric

    sf = make_session_factory(pg_engine)
    recent_empty = _published(pg_engine, channel_ref="1")          # source returns empty
    old = _published(pg_engine, channel_ref="2",
                     when=dt.datetime.now(UTC) - dt.timedelta(hours=100))  # outside window
    source = FakeSource({"1": MessageStats(), "2": MessageStats(views=9)})
    collector = MetricsCollector(sf, source, max_age_hours=48)

    stats = collector.collect_publications(limit=50)
    assert stats["polled"] == 1              # only the recent one is polled; old is filtered out
    assert stats["recorded"] == 0            # its stats were empty -> nothing recorded
    with Session(pg_engine) as s:
        assert s.scalar(select(func.count()).select_from(PublicationMetric)) == 0
        _ = recent_empty, old


@pytest.mark.pg
def test_collect_channel_records_subscribers(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import ChannelMetric

    sf = make_session_factory(pg_engine)
    source = FakeSource({}, subscribers=5000)
    collector = MetricsCollector(sf, source, channel="telegram")

    assert collector.collect_channel() is True
    with Session(pg_engine) as s:
        row = s.execute(select(ChannelMetric).where(ChannelMetric.channel == "telegram")).scalars().one()
        assert row.subscribers == 5000
