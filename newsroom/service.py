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


def factcheck_high_only() -> bool:
    return os.getenv("FACTCHECK_HIGH_ONLY", "false").strip().lower() in {"1", "true", "yes"}


def stories_enabled() -> bool:
    return os.getenv("STORIES_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def story_updates_enabled() -> bool:
    return os.getenv("STORY_UPDATES_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def significance_enabled() -> bool:
    return os.getenv("SIGNIFICANCE_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def curation_enabled() -> bool:
    return os.getenv("CURATION_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def editorial_enabled() -> bool:
    return os.getenv("EDITORIAL_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def media_ogimage_enabled() -> bool:
    return os.getenv("MEDIA_OGIMAGE_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def media_download_enabled() -> bool:
    return os.getenv("MEDIA_DOWNLOAD_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def media_check_enabled() -> bool:
    return os.getenv("MEDIA_CHECK_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def media_moderation_enabled() -> bool:
    return os.getenv("MEDIA_MODERATION_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def reputation_enabled() -> bool:
    return os.getenv("REPUTATION_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def monitoring_enabled() -> bool:
    return os.getenv("MONITORING_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def demand_metrics_enabled() -> bool:
    return os.getenv("DEMAND_METRICS_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def metrics_enabled() -> bool:
    return os.getenv("METRICS_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


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
                           significance_threshold: float | None = None,
                           stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Build the shared fact base for publishable events until stopped. Runs
    before fact-check and editorial so both work from one viewpoint. Skips
    low-significance events when a threshold is set. Nothing published."""
    from newsroom.factbase import build_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(build_pending, session_factory, builder,
                                            significance_threshold=significance_threshold)
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
                            significance_threshold: float | None = None,
                            risk_levels: tuple[str, ...] | None = None,
                            stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Fact-check publishable events (claims -> evidence -> verdict) until stopped.
    Runs before editorial so drafts can build on verified claims. Skips
    low-significance events when a threshold is set (biggest token saving); with
    `risk_levels` set, only checks those risk levels (FACTCHECK_HIGH_ONLY).
    Nothing published."""
    from newsroom.factcheck import check_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(check_pending, session_factory, checker,
                                            significance_threshold=significance_threshold,
                                            risk_levels=risk_levels)
            if stats.get("events"):
                log.info("factcheck tick", extra=bind(**stats))
        except Exception:
            log.exception("factcheck tick failed")
        await asyncio.sleep(tick_seconds)


def build_story_linker_from_env(session_factory):
    """Assemble the story linker (§7). Pure vector similarity — no LLM."""
    from newsroom.analyze.stories import StoryLinker

    return StoryLinker(session_factory)


async def stories_forever(session_factory, linker, *, tick_seconds: float = 30.0,
                          stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Link each clustered event to a story (§7) and age idle stories to dormant.
    Runs before story-updates and editorial, so near-identical events share a
    story (repeats become summary-only, not duplicate posts) and story posts can
    reply-chain. Nothing is published."""
    from newsroom.analyze.stories import link_pending, mark_dormant

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(link_pending, session_factory, linker)
            dormant = await asyncio.to_thread(mark_dormant, session_factory)
            if stats.get("linked") or dormant:
                log.info("story-link tick", extra=bind(dormant=dormant, **stats))
        except Exception:
            log.exception("story-link tick failed")
        await asyncio.sleep(tick_seconds)


def build_story_updater_from_env(session_factory):  # pragma: no cover — needs OpenAI
    """Assemble the story-update classifier (LLM)."""
    from newsroom.editorial import LLMUpdateClassifier, StoryUpdater

    return StoryUpdater(session_factory, classifier=LLMUpdateClassifier())


async def story_updates_forever(session_factory, updater, *, tick_seconds: float = 30.0,
                                significance_threshold: float | None = None,
                                stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Classify how each new event moves its story (update_type) and update the
    running summary, before editorial drafts posts. Skips low-significance events
    when a threshold is set. Nothing is published."""
    from newsroom.editorial import classify_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(classify_pending, session_factory, updater,
                                            significance_threshold=significance_threshold)
            if stats.get("classified"):
                log.info("story-update tick", extra=bind(**stats))
        except Exception:
            log.exception("story-update tick failed")
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


async def significance_forever(session_factory, config, *, tick_seconds: float = 30.0,
                               stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Score each postable event's significance (T1 gate) before editorial, so
    niche/minor news is skipped. Deterministic — no LLM. Nothing is published."""
    from newsroom.analyze.significance import score_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(score_pending, session_factory, config)
            if stats.get("scored"):
                log.info("significance tick", extra=bind(**stats))
        except Exception:
            log.exception("significance tick failed")
        await asyncio.sleep(tick_seconds)


def build_editorial_ranker_from_env():  # pragma: no cover — needs OpenAI
    """Assemble the LLM editorial ranker (comparative curation)."""
    from newsroom.editorial import LLMEditorialRanker

    return LLMEditorialRanker()


async def curation_forever(session_factory, ranker, *, significance_threshold: float | None = None,
                           window_hours: int = 6, tick_seconds: float = 120.0,
                           stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Mark recent significant events publish/hold (must-publish deterministically,
    the rest by comparative LLM ranking), before editorial. Only publish-marked
    events are drafted — the count follows the news, not a fixed rate."""
    from newsroom.editorial import curate_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(
                curate_pending, session_factory, ranker,
                significance_threshold=significance_threshold, window_hours=window_hours)
            if stats.get("curated"):
                log.info("curation tick", extra=bind(**stats))
        except Exception:
            log.exception("curation tick failed")
        await asyncio.sleep(tick_seconds)


async def editorial_forever(session_factory, pipeline, *, tick_seconds: float = 30.0,
                            significance_threshold: float | None = None,
                            require_curation: bool = False,
                            stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Draft posts for publishable events until stopped. When a significance
    threshold is given, events scored below it are skipped (T1 gate); with
    require_curation, only events the curation marked publish are drafted. Nothing is
    published."""
    from newsroom.editorial import produce_drafts

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(
                produce_drafts, session_factory, pipeline,
                significance_threshold=significance_threshold,
                require_curation=require_curation)
            if stats.get("produced"):
                log.info("editorial tick", extra=bind(**stats))
        except Exception:
            log.exception("editorial tick failed")
        await asyncio.sleep(tick_seconds)


def build_publisher_from_env(session_factory):
    """Assemble the Publisher: Telegram adapter (PUBLISH_ENABLED + credentials) +
    stop-list + limits. No OpenAI — publishing is DB + Telegram only."""
    from newsroom.analyze.stoplist import load_stoplist
    from newsroom.publishers import Publisher, Supervisor, TelegramPublisher, load_limits

    telegram = TelegramPublisher.from_env()
    return Publisher(
        session_factory,
        telegram=telegram,
        stoplist_rules=load_stoplist(CONFIG_DIR / "stoplist.yaml"),
        limits=load_limits(CONFIG_DIR / "limits.yaml"),
        supervisor=Supervisor.from_env(telegram),
    )


async def publish_forever(session_factory, publisher, *, tick_seconds: float = 20.0,
                          stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Publish critic-passed drafts that clear the gate. The Publisher no-ops when
    the master switch is off; each draft still passes the stop button, stop-list,
    limits and surge check before anything is sent."""
    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(publisher.publish_pending)
            if stats.get("published") or stats.get("blocked"):
                log.info("publish tick", extra=bind(**stats))
        except Exception:
            log.exception("publish tick failed")
        await asyncio.sleep(tick_seconds)


def build_media_downloader_from_env(session_factory):  # pragma: no cover — Pillow/network
    """Assemble the media downloader: local store + Pillow decoder + HTTP fetch."""
    from newsroom.media import LocalMediaStore, MediaDownloader, PillowDecoder

    store_dir = os.getenv("MEDIA_STORE_DIR", "./media")
    return MediaDownloader(session_factory, store=LocalMediaStore(store_dir), decoder=PillowDecoder())


def build_ogimage_resolver_from_env(session_factory):
    """Assemble the og:image resolver (HTTP page fetch). No LLM."""
    from newsroom.media import OgImageResolver

    return OgImageResolver(session_factory)


async def ogimage_forever(session_factory, resolver, *, tick_seconds: float = 45.0,
                          stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Resolve og:image for filter-passed items whose feed carried no media, so the
    downloader then has something to fetch. Runs before media download."""
    from newsroom.media import resolve_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(resolve_pending, session_factory, resolver)
            if stats.get("found"):
                log.info("og-image tick", extra=bind(**stats))
        except Exception:
            log.exception("og-image tick failed")
        await asyncio.sleep(tick_seconds)


async def media_download_forever(session_factory, downloader, *, tick_seconds: float = 45.0,
                                 stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Download media for filter-passed items and compute pHashes, feeding the
    reuse check. Nothing is published."""
    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(downloader.download_pending)
            if stats.get("stored"):
                log.info("media-download tick", extra=bind(**stats))
        except Exception:
            log.exception("media-download tick failed")
        await asyncio.sleep(tick_seconds)


async def media_check_forever(session_factory, *, tick_seconds: float = 60.0,
                              stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Check publishable events' media for recycled images (pHash). DB-only, no
    LLM; dormant until media is downloaded. Nothing is published."""
    from newsroom.factcheck import MediaChecker, check_media_pending

    checker = MediaChecker(session_factory)
    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(check_media_pending, session_factory, checker)
            if stats.get("events"):
                log.info("media-check tick", extra=bind(**stats))
        except Exception:
            log.exception("media-check tick failed")
        await asyncio.sleep(tick_seconds)


def build_image_moderator_from_env():  # pragma: no cover — OpenAI vision
    from newsroom.media import OpenAIImageModerator

    return OpenAIImageModerator()


async def media_moderation_forever(session_factory, moderator, *, tick_seconds: float = 60.0,
                                   stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Moderate downloaded images (the media stop-list) so verified media can
    attach to posts. Nothing is published."""
    from newsroom.media import moderate_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(moderate_pending, session_factory, moderator)
            if stats.get("events"):
                log.info("media-moderation tick", extra=bind(**stats))
        except Exception:
            log.exception("media-moderation tick failed")
        await asyncio.sleep(tick_seconds)


async def reputation_forever(session_factory, *, tick_seconds: float = 60.0,
                             stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Credit source reputation (first / copy / confirmed / refuted) from settled
    events. DB-only, no LLM."""
    from newsroom.reputation import record_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(record_pending, session_factory)
            if stats.get("origins") or stats.get("outcomes"):
                log.info("reputation tick", extra=bind(**stats))
        except Exception:
            log.exception("reputation tick failed")
        await asyncio.sleep(tick_seconds)


async def monitoring_forever(session_factory, *, tick_seconds: float = 120.0,
                             stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Draft corrections when a source retracts an item behind a published post
    (architecture §9.7). DB-only, no LLM. Corrections stay drafts."""
    from newsroom.monitoring import monitor_publications

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(monitor_publications, session_factory)
            if stats.get("corrections"):
                log.info("monitoring tick", extra=bind(**stats))
        except Exception:
            log.exception("monitoring tick failed")
        await asyncio.sleep(tick_seconds)


async def demand_forever(session_factory, collector, *, tick_seconds: float = 1800.0,
                         sources_every: int = 4, stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Snapshot engagement of monitored source posts + channel sizes via the shared
    Telethon session, so demand data accumulates. Slow cadence (engagement changes
    slowly); subscriber counts even slower. Reads only — nothing is published."""
    from newsroom.analyze.demand import DemandCollector, TelethonDemandSource

    loop = asyncio.get_running_loop()
    client = await collector.wait_client()
    demand = DemandCollector(session_factory, TelethonDemandSource(client, loop))
    ticks = 0
    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(demand.collect_items)
            if ticks % sources_every == 0:
                await asyncio.to_thread(demand.collect_sources)
            if stats.get("recorded"):
                log.info("demand tick", extra=bind(**stats))
        except Exception:
            log.exception("demand tick failed")
        ticks += 1
        await asyncio.sleep(tick_seconds)


async def metrics_forever(session_factory, collector, channel_chat_id, *,
                          tick_seconds: float = 600.0, channel_every: int = 6,
                          stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Snapshot publication and channel metrics via the collector's shared Telethon
    session (architecture §5.3, §11). Waits for the collector to connect, resolves
    the publish channel once, then polls on a slow cadence. The reading account
    must be a member of the publish channel to see message stats.
    """
    from newsroom.publishers import MetricsCollector, TelethonMetricsSource

    loop = asyncio.get_running_loop()
    client = await collector.wait_client()
    try:
        entity = await client.get_entity(channel_chat_id)
    except Exception as exc:  # noqa: BLE001
        # Expected until the reading account is a member of the publish channel:
        # Telethon can only resolve a channel it has seen. Metrics stay off.
        log.warning("metrics off: reading account cannot resolve the publish channel "
                    "(join it with that account first)", extra=bind(error=str(exc)))
        return

    source = TelethonMetricsSource(client, entity, loop)
    mc = MetricsCollector(session_factory, source, channel="telegram")
    ticks = 0
    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(mc.collect_publications)
            if ticks % channel_every == 0:
                await asyncio.to_thread(mc.collect_channel)
            if stats.get("recorded"):
                log.info("metrics tick", extra=bind(**stats))
        except Exception:
            log.exception("metrics tick failed")
        ticks += 1
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

    # Significance threshold is loaded once and shared: the scoring loop uses the
    # full config, while factbase / factcheck / story-updates / editorial take just
    # the threshold to skip low-significance events (no tokens on unpostable news).
    significance_threshold = None
    sig_config = None
    if significance_enabled():
        from newsroom.analyze.significance import load_significance_config

        sig_config = load_significance_config(CONFIG_DIR / "significance.yaml")
        significance_threshold = sig_config.threshold

    tasks = [asyncio.create_task(poll_rss_forever(session_factory))]
    if sig_config is not None:
        tasks.append(asyncio.create_task(significance_forever(session_factory, sig_config)))
        log.info("significance gate enabled", extra=bind(threshold=significance_threshold))
    else:
        log.info("significance gate disabled (SIGNIFICANCE_ENABLED off)")
    telegram_collector = None
    if telegram_enabled():
        telegram_collector = TelegramCollector(session_factory)
        tasks.append(asyncio.create_task(telegram_collector.start()))
        log.info("telegram collection enabled")
    else:
        log.info("telegram collection disabled (COLLECTOR_TELEGRAM_ENABLED off)")

    if verify_enabled():
        verifier = build_verifier_from_env(session_factory)
        tasks.append(asyncio.create_task(verify_forever(session_factory, verifier)))
        log.info("verification enabled")
    else:
        log.info("verification disabled (VERIFY_ENABLED off)")

    if media_ogimage_enabled():
        resolver = build_ogimage_resolver_from_env(session_factory)
        tasks.append(asyncio.create_task(ogimage_forever(session_factory, resolver)))
        log.info("og:image resolution enabled")
    else:
        log.info("og:image resolution disabled (MEDIA_OGIMAGE_ENABLED off)")

    if media_download_enabled():
        downloader = build_media_downloader_from_env(session_factory)
        tasks.append(asyncio.create_task(media_download_forever(session_factory, downloader)))
        log.info("media download enabled")
    else:
        log.info("media download disabled (MEDIA_DOWNLOAD_ENABLED off)")

    if factbase_enabled():
        builder = build_factbase_builder_from_env(session_factory)
        tasks.append(asyncio.create_task(factbase_forever(
            session_factory, builder, significance_threshold=significance_threshold)))
        log.info("fact base enabled")
    else:
        log.info("fact base disabled (FACTBASE_ENABLED off)")

    if factcheck_enabled():
        checker = build_factchecker_from_env(session_factory)
        fc_risk = ("high", "critical") if factcheck_high_only() else None
        tasks.append(asyncio.create_task(factcheck_forever(
            session_factory, checker, significance_threshold=significance_threshold,
            risk_levels=fc_risk)))
        log.info("fact-checking enabled", extra=bind(high_only=bool(fc_risk)))
    else:
        log.info("fact-checking disabled (FACTCHECK_ENABLED off)")

    if media_check_enabled():
        tasks.append(asyncio.create_task(media_check_forever(session_factory)))
        log.info("media check enabled")
    else:
        log.info("media check disabled (MEDIA_CHECK_ENABLED off)")

    if media_moderation_enabled():
        moderator = build_image_moderator_from_env()
        tasks.append(asyncio.create_task(media_moderation_forever(session_factory, moderator)))
        log.info("media moderation enabled")
    else:
        log.info("media moderation disabled (MEDIA_MODERATION_ENABLED off)")

    if stories_enabled():
        linker = build_story_linker_from_env(session_factory)
        tasks.append(asyncio.create_task(stories_forever(session_factory, linker)))
        log.info("story linking enabled")
    else:
        log.info("story linking disabled (STORIES_ENABLED off)")

    if story_updates_enabled():
        updater = build_story_updater_from_env(session_factory)
        tasks.append(asyncio.create_task(story_updates_forever(
            session_factory, updater, significance_threshold=significance_threshold)))
        log.info("story updates enabled")
    else:
        log.info("story updates disabled (STORY_UPDATES_ENABLED off)")

    if curation_enabled():
        ranker = build_editorial_ranker_from_env()
        tasks.append(asyncio.create_task(curation_forever(
            session_factory, ranker, significance_threshold=significance_threshold)))
        log.info("editorial curation enabled")
    else:
        log.info("editorial curation disabled (CURATION_ENABLED off)")

    if editorial_enabled():
        pipeline = build_editorial_pipeline_from_env(session_factory)
        tasks.append(asyncio.create_task(editorial_forever(
            session_factory, pipeline, significance_threshold=significance_threshold,
            require_curation=curation_enabled())))
        log.info("editorial drafting enabled", extra=bind(require_curation=curation_enabled()))
    else:
        log.info("editorial drafting disabled (EDITORIAL_ENABLED off)")

    publisher = build_publisher_from_env(session_factory)
    if publisher.telegram.is_enabled():
        tasks.append(asyncio.create_task(publish_forever(session_factory, publisher)))
        log.info("publishing enabled")
        if os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip():
            from newsroom.publishers import SupervisionBot

            bot = SupervisionBot(session_factory, publisher.telegram)
            tasks.append(asyncio.create_task(bot.poll_forever()))
            log.info("supervision bot enabled")
    else:
        log.info("publishing disabled (PUBLISH_ENABLED off or no credentials)")

    if reputation_enabled():
        tasks.append(asyncio.create_task(reputation_forever(session_factory)))
        log.info("reputation enabled")
    else:
        log.info("reputation disabled (REPUTATION_ENABLED off)")

    if monitoring_enabled():
        tasks.append(asyncio.create_task(monitoring_forever(session_factory)))
        log.info("post-publication monitoring enabled")
    else:
        log.info("post-publication monitoring disabled (MONITORING_ENABLED off)")

    if metrics_enabled():
        channel_chat_id = publisher.telegram.active_chat_id
        if telegram_collector is not None and channel_chat_id is not None:
            tasks.append(asyncio.create_task(
                metrics_forever(session_factory, telegram_collector, channel_chat_id)))
            log.info("metrics enabled")
        else:
            log.info("metrics disabled (needs telegram collection + a publish channel)")
    else:
        log.info("metrics disabled (METRICS_ENABLED off)")

    if demand_metrics_enabled():
        if telegram_collector is not None:
            tasks.append(asyncio.create_task(demand_forever(session_factory, telegram_collector)))
            log.info("demand metrics enabled")
        else:
            log.info("demand metrics disabled (needs telegram collection)")
    else:
        log.info("demand metrics disabled (DEMAND_METRICS_ENABLED off)")

    await asyncio.gather(*tasks)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_service())
