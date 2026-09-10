"""Collect-only runner (stage 1a). Syncs sources from config and does one RSS
collect pass over all active feeds. Nothing is published. Telegram collection
(M6) and the publisher (M7) are wired separately.

    python -m newsroom.runner
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from sqlalchemy import select

from newsroom.collectors.rss import CollectResult, RssCollector
from newsroom.logsetup import bind
from newsroom.models import Source

log = logging.getLogger("newsroom.runner")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"


def collect_all_rss(session_factory, fetch: Callable[[str], bytes] | None = None) -> list[CollectResult]:
    """One collect pass over every active RSS source. Idempotent by design."""
    with session_factory() as s:
        source_ids = list(s.scalars(
            select(Source.id).where(Source.kind == "rss", Source.active.is_(True))
        ).all())
    collector = RssCollector(session_factory, fetch=fetch)
    return [collector.collect_source(sid) for sid in source_ids]


def main() -> None:  # pragma: no cover — thin wiring, exercised via collect_all_rss tests
    from dotenv import load_dotenv

    from newsroom.config.sources import load_sources
    from newsroom.db import init_db, make_engine, make_session_factory
    from newsroom.logsetup import setup_logging
    from newsroom.sources.registry import health_report, sync_sources

    load_dotenv()
    setup_logging()
    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    configs = load_sources(CONFIG_PATH)
    with session_factory() as s:
        synced = sync_sources(s, configs)
        s.commit()
    log.info("sources synced", extra=bind(**synced))

    results = collect_all_rss(session_factory)
    log.info("collect pass done", extra=bind(
        sources=len(results),
        created=sum(r.created for r in results),
        updated=sum(r.updated for r in results),
        failed=sum(not r.ok for r in results),
    ))

    with session_factory() as s:
        for row in health_report(s):
            if not row["healthy"]:
                log.warning("source unhealthy", extra=bind(**row))


if __name__ == "__main__":  # pragma: no cover
    main()
