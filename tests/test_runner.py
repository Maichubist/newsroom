from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsroom.config.sources import SourceConfig
from newsroom.models import Item, Source
from newsroom.runner import collect_all_rss
from newsroom.sources.registry import sync_sources

pytestmark = pytest.mark.pg

_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><language>uk</language>
<item><title>One</title><link>https://f/1</link><guid>g1</guid><description>body</description></item>
<item><title>Two</title><link>https://f/2</link><guid>g2</guid><description>body</description></item>
</channel></rss>"""


def _sf(pg_engine):
    from newsroom.db import make_session_factory
    return make_session_factory(pg_engine)


def test_collect_all_rss_collects_every_active_source(pg_engine):
    good = "https://runner-good/rss"
    bad = "https://runner-bad/rss"
    with Session(pg_engine) as s:
        sync_sources(s, [
            SourceConfig(kind="rss", handle_or_url=good, name="Good", origin="ua", tier="media"),
            SourceConfig(kind="rss", handle_or_url=bad, name="Bad", origin="world", tier="media"),
        ])
        s.commit()

    def fetch(url: str) -> bytes:
        if url == bad:
            raise RuntimeError("dead feed")
        return _FEED

    results = collect_all_rss(_sf(pg_engine), fetch=fetch)
    by_ok = {r.ok for r in results}
    assert True in by_ok and False in by_ok           # good succeeded, bad failed
    assert any(r.created == 2 for r in results)        # good ingested both items

    with Session(pg_engine) as s:
        good_id = s.scalar(select(Source.id).where(Source.handle_or_url == good))
        bad_id = s.scalar(select(Source.id).where(Source.handle_or_url == bad))
        assert s.scalar(select(func.count()).select_from(Item).where(Item.source_id == good_id)) == 2
        assert s.get(Source, good_id).consecutive_failures == 0
        assert s.get(Source, good_id).last_success_at is not None
        # a dead source degrades quietly and is visible via health fields
        assert s.get(Source, bad_id).consecutive_failures == 1
        assert "dead feed" in (s.get(Source, bad_id).last_error or "")


def test_second_pass_is_idempotent(pg_engine):
    handle = "https://runner-idem/rss"
    with Session(pg_engine) as s:
        sync_sources(s, [SourceConfig(kind="rss", handle_or_url=handle, name="I", origin="ua", tier="media")])
        s.commit()

    collect_all_rss(_sf(pg_engine), fetch=lambda url: _FEED)
    collect_all_rss(_sf(pg_engine), fetch=lambda url: _FEED)   # restart -> no duplicates

    with Session(pg_engine) as s:
        sid = s.scalar(select(Source.id).where(Source.handle_or_url == handle))
        assert s.scalar(select(func.count()).select_from(Item).where(Item.source_id == sid)) == 2
