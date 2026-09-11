"""Long-running collect-only service (stage 1a).

Two concurrent duties, nothing published:
  * RSS: poll each source on its own `poll_interval` (due-based scheduler).
  * Telegram: realtime + backfill via TelegramCollector (only if the flag is on).

The scheduling *decisions* (`due_source_ids`, `collect_due_rss`) are sync and
pg-tested; the async loops are thin wiring (`# pragma: no cover`).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from pathlib import Path
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.collectors.rss import CollectResult, RssCollector
from newsroom.logsetup import bind
from newsroom.models import Source

log = logging.getLogger("newsroom.service")

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def verify_enabled() -> bool:
    return os.getenv("VERIFY_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def factbase_enabled() -> bool:
    return os.getenv("FACTBASE_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def factcheck_enabled() -> bool:
    return os.getenv("FACTCHECK_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def editorial_enabled() -> bool:
    return os.getenv("EDITORIAL_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def due_source_ids(session: Session, now: dt.datetime | None = None) -> list[int]:
    """Active RSS sources whose poll_interval has elapsed since the last success
    (never-collected and previously-failing sources are always due)."""
    now = now or _utc_now()
    sources = session.scalars(
        select(Source).where(Source.kind == "rss", Source.active.is_(True))
    ).all()
    due: list[int] = []
    for s in sources:
        if s.last_success_at is None:
            due.append(s.id)
            continue
        if (now - s.last_success_at).total_seconds() >= (s.poll_interval or 300):
            due.append(s.id)
    return due


def collect_due_rss(session_factory, now: dt.datetime | None = None,
                    fetch: Callable[[str], bytes] | None = None) -> list[CollectResult]:
    """One scheduler tick: collect only the RSS sources that are due right now."""
    with session_factory() as s:
        ids = due_source_ids(s, now)
    collector = RssCollector(session_factory, fetch=fetch)
    return [collector.collect_source(sid) for sid in ids]


async def poll_rss_forever(session_factory, *, tick_seconds: float = 30.0,
                           stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Run the RSS scheduler until `stop` is set. Blocking collection runs in a
    worker thread so the event loop stays free for the Telegram realtime stream."""
    while not (stop and stop.is_set()):
        try:
            results = await asyncio.to_thread(collect_due_rss, session_factory)
            if results:
                log.info("rss tick", extra=bind(collected=len(results),
                                                created=sum(r.created for r in results)))
        except Exception:
            log.exception("rss tick failed")
        await asyncio.sleep(tick_seconds)


def build_verifier_from_env(session_factory):  # pragma: no cover — needs OpenAI
    """Assemble a Verifier from env + config (LLM classifier + OpenAI embeddings)."""
    from newsroom.analyze.classifier import LLMClassifier
    from newsroom.analyze.embeddings import OpenAIEmbedder
    from newsroom.analyze.risk import load_risk_matrix
    from newsroom.analyze.signal import load_filters
    from newsroom.analyze.stoplist import load_stoplist
    from newsroom.analyze.verify import Verifier

    return Verifier(
        session_factory,
        classifier=LLMClassifier(),
        embedder=OpenAIEmbedder(),
        risk_matrix=load_risk_matrix(CONFIG_DIR / "risk.yaml"),
        filters=load_filters(CONFIG_DIR / "filters.yaml"),
        stoplist_rules=load_stoplist(CONFIG_DIR / "stoplist.yaml"),
    )


async def verify_forever(session_factory, verifier, *, tick_seconds: float = 20.0,
                         stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Run the verification pass over `new` items until stopped. Nothing published."""
    from newsroom.analyze.verify import verify_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(verify_pending, session_factory, verifier)
            if stats.get("processed"):
                log.info("verify tick", extra=bind(**stats))
        except Exception:
            log.exception("verify tick failed")
        await asyncio.sleep(tick_seconds)


def build_factbase_builder_from_env(session_factory):  # pragma: no cover — needs OpenAI
    """Assemble the fact-base builder (LLM fact extractor + OpenAI embeddings)."""
    from newsroom.analyze.embeddings import OpenAIEmbedder
    from newsroom.factbase import FactBaseBuilder, LLMFactExtractor

    return FactBaseBuilder(session_factory, extractor=LLMFactExtractor(), embedder=OpenAIEmbedder())


async def factbase_forever(session_factory, builder, *, tick_seconds: float = 30.0,
                           stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Build the shared fact base for publishable events until stopped. Runs
    before fact-check and editorial so both work from one viewpoint. Nothing published."""
    from newsroom.factbase import build_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(build_pending, session_factory, builder)
            if stats.get("events"):
                log.info("factbase tick", extra=bind(**stats))
        except Exception:
            log.exception("factbase tick failed")
        await asyncio.sleep(tick_seconds)


def build_factchecker_from_env(session_factory):  # pragma: no cover — needs OpenAI
    """Assemble the fact-checker: LLM claim extractor + composite evidence
    (own corpus + official registry) + LLM verdict. Refutation-DB clients are
    off until configured, so they are not wired in here yet."""
    from newsroom.analyze.embeddings import OpenAIEmbedder
    from newsroom.factcheck import (
        CompositeEvidenceSearcher,
        CorpusEvidenceSearcher,
        FactChecker,
        LLMClaimExtractor,
        LLMVerdictJudge,
        OfficialRegistrySearcher,
    )

    embedder = OpenAIEmbedder()
    searcher = CompositeEvidenceSearcher([
        CorpusEvidenceSearcher(session_factory, embedder),
        OfficialRegistrySearcher(session_factory, embedder),
    ])
    return FactChecker(
        session_factory,
        extractor=LLMClaimExtractor(),
        searcher=searcher,
        judge=LLMVerdictJudge(),
    )


async def factcheck_forever(session_factory, checker, *, tick_seconds: float = 30.0,
                            stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Fact-check publishable events (claims -> evidence -> verdict) until stopped.
    Runs before editorial so drafts can build on verified claims. Nothing published."""
    from newsroom.factcheck import check_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(check_pending, session_factory, checker)
            if stats.get("events"):
                log.info("factcheck tick", extra=bind(**stats))
        except Exception:
            log.exception("factcheck tick failed")
        await asyncio.sleep(tick_seconds)


def build_editorial_pipeline_from_env(session_factory):  # pragma: no cover — needs OpenAI
    """Assemble the editorial pipeline (LLM generator + charter configs)."""
    from newsroom.analyze.ai_accent import load_ai_accent
    from newsroom.analyze.stoplist import load_stoplist
    from newsroom.editorial import EditorialPipeline, LLMGenerator

    return EditorialPipeline(
        session_factory,
        generator=LLMGenerator(),
        stoplist_rules=load_stoplist(CONFIG_DIR / "stoplist.yaml"),
        ai_accent_patterns=load_ai_accent(CONFIG_DIR / "ai_accent.yaml"),
    )


async def editorial_forever(session_factory, pipeline, *, tick_seconds: float = 30.0,
                            stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Draft posts for publishable events until stopped. Nothing is published."""
    from newsroom.editorial import produce_drafts

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(produce_drafts, session_factory, pipeline)
            if stats.get("produced"):
                log.info("editorial tick", extra=bind(**stats))
        except Exception:
            log.exception("editorial tick failed")
        await asyncio.sleep(tick_seconds)


async def run_service() -> None:  # pragma: no cover — process entrypoint
    """`python -m newsroom.service` — the collect-only daemon."""
    from dotenv import load_dotenv

    from newsroom.collectors.telegram import TelegramCollector, telegram_enabled
    from newsroom.config.sources import load_sources
    from newsroom.db import init_db, make_engine, make_session_factory
    from newsroom.logsetup import setup_logging
    from newsroom.runner import CONFIG_PATH
    from newsroom.sources.registry import sync_sources

    load_dotenv()
    setup_logging()
    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as s:
        sync_sources(s, load_sources(CONFIG_PATH))
        s.commit()

    tasks = [asyncio.create_task(poll_rss_forever(session_factory))]
    if telegram_enabled():
        tasks.append(asyncio.create_task(TelegramCollector(session_factory).start()))
        log.info("telegram collection enabled")
    else:
        log.info("telegram collection disabled (COLLECTOR_TELEGRAM_ENABLED off)")

    if verify_enabled():
        verifier = build_verifier_from_env(session_factory)
        tasks.append(asyncio.create_task(verify_forever(session_factory, verifier)))
        log.info("verification enabled")
    else:
        log.info("verification disabled (VERIFY_ENABLED off)")

    if factbase_enabled():
        builder = build_factbase_builder_from_env(session_factory)
        tasks.append(asyncio.create_task(factbase_forever(session_factory, builder)))
        log.info("fact base enabled")
    else:
        log.info("fact base disabled (FACTBASE_ENABLED off)")

    if factcheck_enabled():
        checker = build_factchecker_from_env(session_factory)
        tasks.append(asyncio.create_task(factcheck_forever(session_factory, checker)))
        log.info("fact-checking enabled")
    else:
        log.info("fact-checking disabled (FACTCHECK_ENABLED off)")

    if editorial_enabled():
        pipeline = build_editorial_pipeline_from_env(session_factory)
        tasks.append(asyncio.create_task(editorial_forever(session_factory, pipeline)))
        log.info("editorial drafting enabled")
    else:
        log.info("editorial drafting disabled (EDITORIAL_ENABLED off)")

    await asyncio.gather(*tasks)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_service())
