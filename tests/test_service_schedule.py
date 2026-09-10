from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.orm import Session

from newsroom.db import make_session_factory
from newsroom.models import Source
from newsroom.service import collect_due_rss, due_source_ids

pytestmark = pytest.mark.pg
UTC = dt.timezone.utc

_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><item><title>t</title><link>https://x/1</link><guid>g1</guid></item></channel></rss>"""


def _src(session, handle, **kw):
    src = Source(kind="rss", handle_or_url=handle, name=handle, origin="ua", tier="media", **kw)
    session.add(src)
    session.flush()
    return src.id


def test_due_source_ids_respects_poll_interval(pg_engine):
    now = dt.datetime.now(UTC)
    with Session(pg_engine) as s:
        never = _src(s, "https://due/never", poll_interval=300)
        recent = _src(s, "https://due/recent", poll_interval=300, last_success_at=now - dt.timedelta(seconds=60))
        stale = _src(s, "https://due/stale", poll_interval=300, last_success_at=now - dt.timedelta(seconds=600))
        inactive = _src(s, "https://due/inactive", poll_interval=300, active=False)
        s.commit()

        due = set(due_source_ids(s, now))
        assert never in due and stale in due          # never-collected + stale are due
        assert recent not in due                      # collected 60s ago, interval 300s
        assert inactive not in due                     # inactive is never polled


def test_collect_due_rss_collects_only_due(pg_engine):
    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(UTC)
    with Session(pg_engine) as s:
        due = _src(s, "https://cdue/due", poll_interval=300)  # never collected -> due
        fresh = _src(s, "https://cdue/fresh", poll_interval=300, last_success_at=now)
        s.commit()

    results = collect_due_rss(sf, now=now, fetch=lambda url: _FEED)
    collected = {r.source_id for r in results}
    assert due in collected and fresh not in collected

    # after collecting, the due source is no longer due on the next tick
    with Session(pg_engine) as s:
        assert due not in set(due_source_ids(s, dt.datetime.now(UTC)))
