from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsroom.collectors.rss import RssCollector
from newsroom.db import make_session_factory
from newsroom.models import Item, Source

pytestmark = pytest.mark.pg

_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>t</title>
<item><title>Zelensky meets EU leaders in Kyiv</title><link>https://ex/ua</link><guid>ua1</guid>
  <description>Talks about the war in Ukraine.</description></item>
<item><title>US election latest: Ohio results</title><link>https://ex/us</link><guid>us1</guid>
  <description>Domestic politics, nothing about the war.</description></item>
</channel></rss>"""


def _source(pg_engine, *, origin: str) -> int:
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url=f"https://f/{origin}", name="F",
                     origin=origin, lang="en", tier="media")
        s.add(src)
        s.flush()
        sid = src.id
        s.commit()
        return sid


def _titles(pg_engine, source_id: int) -> set[str]:
    with Session(pg_engine) as s:
        return set(s.execute(select(Item.title).where(Item.source_id == source_id)).scalars().all())


def test_world_feed_keeps_only_ukraine_items(pg_engine):
    sf = make_session_factory(pg_engine)
    sid = _source(pg_engine, origin="world")
    collector = RssCollector(sf, fetch=lambda url: _FEED)

    result = collector.collect_source(sid)
    assert result.ok and result.created == 1              # only the Ukraine item
    assert _titles(pg_engine, sid) == {"Zelensky meets EU leaders in Kyiv"}


def test_ua_feed_keeps_all_items(pg_engine):
    sf = make_session_factory(pg_engine)
    sid = _source(pg_engine, origin="ua")                 # not filtered
    collector = RssCollector(sf, fetch=lambda url: _FEED)

    result = collector.collect_source(sid)
    assert result.ok and result.created == 2              # both kept
