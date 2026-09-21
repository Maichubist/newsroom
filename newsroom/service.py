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


def dedup_enabled() -> bool:
    return os.getenv("DEDUP_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def curation_enabled() -> bool:
    return os.getenv("CURATION_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def digest_enabled() -> bool:
    return os.getenv("DIGEST_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


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


def telegram_autosync_subs_enabled() -> bool:
    return os.getenv("TELEGRAM_AUTOSYNC_SUBS", "false").strip().lower() in {"1", "true", "yes"}


def topics_enabled() -> bool:
    return os.getenv("TOPICS_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def taxonomy_merge_enabled() -> bool:
    return os.getenv("TAXONOMY_MERGE_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def metrics_enabled() -> bool:
    return os.getenv("METRICS_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def prepublish_dedup_enabled() -> bool:
    return os.getenv("PREPUBLISH_DEDUP_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def prepublish_dedup_enforce() -> bool:
    return os.getenv("PREPUBLISH_DEDUP_ENFORCE", "false").strip().lower() in {"1", "true", "yes"}


def prepublish_enrich_enabled() -> bool:
    return os.getenv("PREPUBLISH_ENRICH_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def ingest_dedup_enabled() -> bool:
    return os.getenv("INGEST_DEDUP_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def ingest_dedup_enforce() -> bool:
    return os.getenv("INGEST_DEDUP_ENFORCE", "false").strip().lower() in {"1", "true", "yes"}


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
    from newsroom.analyze.spine import load_spine
    from newsroom.analyze.stoplist import load_stoplist
    from newsroom.analyze.verify import Verifier

    return Verifier(
        session_factory,
        classifier=LLMClassifier(),
        embedder=OpenAIEmbedder(),
        risk_matrix=load_risk_matrix(CONFIG_DIR / "risk.yaml"),
        filters=load_filters(CONFIG_DIR / "filters.yaml"),
        stoplist_rules=load_stoplist(CONFIG_DIR / "stoplist.yaml"),
        spine=load_spine(CONFIG_DIR / "taxonomy_spine.yaml"),
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
                           require_dedup_settled: bool = False,
                           stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Build the shared fact base for publishable events until stopped. Runs before
    fact-check and editorial so both work from one viewpoint. When require_dedup_settled,
    waits for the ingest-dedup verdict so a duplicate is merged first. Nothing published."""
    from newsroom.factbase import build_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(build_pending, session_factory, builder,
                                            require_dedup_settled=require_dedup_settled)
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
                            risk_levels: tuple[str, ...] | None = None,
                            require_dedup_settled: bool = False,
                            stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Fact-check publishable events (claims -> evidence -> verdict) until stopped. Runs
    before editorial so drafts can build on verified claims. With `risk_levels` set, only
    checks those risk levels (FACTCHECK_HIGH_ONLY). When require_dedup_settled, waits for
    the ingest-dedup verdict first. Nothing published."""
    from newsroom.factcheck import check_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(check_pending, session_factory, checker,
                                            risk_levels=risk_levels,
                                            require_dedup_settled=require_dedup_settled)
            if stats.get("events"):
                log.info("factcheck tick", extra=bind(**stats))
        except Exception:
            log.exception("factcheck tick failed")
        await asyncio.sleep(tick_seconds)


def build_story_linker_from_env(session_factory):
    """Assemble the story linker (§7). Pure vector similarity — no LLM. Threshold
    tunable via STORY_THRESHOLD (calibrated default separates cross-source duplicates
    from distinct same-theme events)."""
    from newsroom.analyze.stories import DEFAULT_STORY_THRESHOLD, StoryLinker

    try:
        threshold = float(os.getenv("STORY_THRESHOLD", str(DEFAULT_STORY_THRESHOLD)))
    except (TypeError, ValueError):
        threshold = DEFAULT_STORY_THRESHOLD
    return StoryLinker(session_factory, threshold=threshold)


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
                                require_dedup_settled: bool = False,
                                stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Classify how each new event moves its story (update_type) and update the running
    summary, before editorial drafts posts. When require_dedup_settled, waits for the
    ingest-dedup verdict first. Nothing is published."""
    from newsroom.editorial import classify_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(classify_pending, session_factory, updater,
                                            require_dedup_settled=require_dedup_settled)
            if stats.get("classified"):
                log.info("story-update tick", extra=bind(**stats))
        except Exception:
            log.exception("story-update tick failed")
        await asyncio.sleep(tick_seconds)


def build_editorial_pipeline_from_env(session_factory):  # pragma: no cover — needs OpenAI
    """Assemble the editorial pipeline (LLM generator + charter configs)."""
    from newsroom.analyze.ai_accent import load_ai_accent
    from newsroom.analyze.spine import load_spine
    from newsroom.analyze.stoplist import load_stoplist
    from newsroom.editorial import EditorialPipeline, LLMGenerator

    return EditorialPipeline(
        session_factory,
        generator=LLMGenerator(),
        stoplist_rules=load_stoplist(CONFIG_DIR / "stoplist.yaml"),
        ai_accent_patterns=load_ai_accent(CONFIG_DIR / "ai_accent.yaml"),
        spine=load_spine(CONFIG_DIR / "taxonomy_spine.yaml"),
    )


def build_dedup_grouper_from_env():  # pragma: no cover — needs OpenAI
    """Assemble the LLM batch dedup grouper."""
    from newsroom.editorial import LLMDedupGrouper

    return LLMDedupGrouper()


async def dedup_forever(session_factory, grouper, *, tick_seconds: float = 900.0,
                        stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Group same-story duplicates in the recent window and mark non-canonicals
    (events.duplicate_of), so only one of a duplicate set is posted. Runs before
    curation/editorial. Slow cadence (one batched call). Nothing is published."""
    from newsroom.editorial import dedup_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(dedup_pending, session_factory, grouper)
            if stats.get("duplicates"):
                log.info("dedup tick", extra=bind(**stats))
        except Exception:
            log.exception("dedup tick failed")
        await asyncio.sleep(tick_seconds)


def build_ingest_dedup_from_env(session_factory):  # pragma: no cover — needs OpenAI
    """Assemble the ingest-time event dedup (Phase B): the pairwise LLM arbiter (reuses the
    publish-time judge) + merge config. The judge is only used for the grey zone."""
    from newsroom.analyze.ingest_dedup import IngestDedup, load_merge_config
    from newsroom.publishers.predup import LLMTwinJudge

    return IngestDedup(session_factory, judge=LLMTwinJudge(),
                       config=load_merge_config(CONFIG_DIR / "dedup.yaml"))


async def ingest_dedup_forever(session_factory, dedup, *, enforce: bool = False,
                               tick_seconds: float = 60.0,
                               stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Merge duplicate events EARLY (Phase B) — right after clustering, before the fact
    base / fact-check / editorial spend on them. Observe mode only logs the verdict.
    Nothing is published."""
    from newsroom.analyze.ingest_dedup import dedup_new_events

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(dedup_new_events, session_factory, dedup, enforce=enforce)
            if stats.get("merged") or stats.get("checked"):
                log.info("ingest dedup tick", extra=bind(**stats))
        except Exception:
            log.exception("ingest dedup tick failed")
        await asyncio.sleep(tick_seconds)


def build_editorial_ranker_from_env():  # pragma: no cover — needs OpenAI
    """Assemble the LLM editorial ranker (comparative curation)."""
    from newsroom.editorial import LLMEditorialRanker

    return LLMEditorialRanker()


async def curation_forever(session_factory, ranker, *, grouper=None, digest_config=None,
                           window_hours: int = 6, tick_seconds: float = 120.0,
                           require_dedup_settled: bool = False, dedup_every: int = 15,
                           stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Mark recent events publish/hold (must-publish deterministically,
    the rest by comparative LLM ranking), before editorial. Only publish-marked
    events are drafted — the count follows the news, not a fixed rate.

    Per tick: reserve attacks for the digest (cheap, DB-only), then curate. The old LLM
    BATCH dedup runs only every `dedup_every` ticks (≈30 min at a 120s tick), not every
    tick: publish-time + ingest dedup (Phase A/B) now catch cross-source twins
    incrementally, so re-sending ~80 titles to the LLM every 2 min was mostly wasted work.
    It stays as slow background insurance for anything they miss. Reserve still runs every
    tick so an attack is caught before curation can must-publish it."""
    from newsroom.editorial import curate_pending, dedup_pending, reserve_digests

    ticks = 0
    while not (stop and stop.is_set()):
        try:
            if digest_config is not None:
                rstats = await asyncio.to_thread(reserve_digests, session_factory, digest_config)
                if rstats.get("reserved"):
                    log.info("digest reserve tick", extra=bind(**rstats))
            if grouper is not None and ticks % max(1, dedup_every) == 0:
                dstats = await asyncio.to_thread(dedup_pending, session_factory, grouper)
                if dstats.get("duplicates"):
                    log.info("dedup tick", extra=bind(**dstats))
            stats = await asyncio.to_thread(
                curate_pending, session_factory, ranker, window_hours=window_hours,
                require_dedup_settled=require_dedup_settled)
            if stats.get("curated"):
                log.info("curation tick", extra=bind(**stats))
        except Exception:
            log.exception("curation tick failed")
        ticks += 1
        await asyncio.sleep(tick_seconds)


async def digest_forever(session_factory, publisher, *, tick_seconds: float = 300.0,
                         stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Reserve attack events and, at each window's publish time (Kyiv), post one
    combined 'Обстріли за ніч/день' digest instead of many individual posts."""
    from newsroom.editorial import load_digest_config, publish_due_digests, reserve_digests

    config = load_digest_config(CONFIG_DIR / "digest.yaml")
    while not (stop and stop.is_set()):
        try:
            await asyncio.to_thread(reserve_digests, session_factory, config)
            stats = await asyncio.to_thread(publish_due_digests, session_factory, config, publisher)
            if stats.get("digests"):
                log.info("digest tick", extra=bind(**stats))
        except Exception:
            log.exception("digest tick failed")
        await asyncio.sleep(tick_seconds)


async def editorial_forever(session_factory, pipeline, *, tick_seconds: float = 30.0,
                            require_curation: bool = False,
                            stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Draft posts for publishable events until stopped. With require_curation, only
    events the curation marked publish are drafted. Nothing is published."""
    from newsroom.editorial import produce_drafts

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(
                produce_drafts, session_factory, pipeline,
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

    from newsroom.media import LocalMediaStore

    telegram = TelegramPublisher.from_env()
    # The store lets the publisher upload Telegram-origin media (no public URL). Purge
    # additionally deletes the local files right after a post goes out (they are never
    # read again). On by default; MEDIA_PURGE_AFTER_PUBLISH=false keeps them.
    media_store = LocalMediaStore(os.getenv("MEDIA_STORE_DIR", "./media"))
    purge = os.getenv("MEDIA_PURGE_AFTER_PUBLISH", "true").strip().lower() in {"1", "true", "yes"}

    # Publish-time dedup (Phase A): the last-second twin check. Off by default. In observe
    # mode (PREPUBLISH_DEDUP_ENFORCE off) it only LOGS its verdict; enforce acts on it.
    predup = None
    if prepublish_dedup_enabled():
        from newsroom.publishers import LLMTwinJudge, PrepublishDedup, load_predup_config

        predup = PrepublishDedup(
            session_factory,
            judge=LLMTwinJudge(),
            config=load_predup_config(CONFIG_DIR / "dedup.yaml"),
        )
        log.info("publish-time dedup enabled", extra=bind(enforce=prepublish_dedup_enforce()))
    else:
        log.info("publish-time dedup disabled (PREPUBLISH_DEDUP_ENABLED off)")

    # Taxonomy spine (charter v0.3): rubric -> oversight, so the supervisor is notified for
    # the confirmed overseen set (war/defense/security/mobilization/politics/geopolitics/
    # corruption), not just risk_level==critical. Corruption especially — a speculative
    # info-attack vector — must reach a human before it stands.
    from newsroom.analyze.spine import load_spine

    spine = load_spine(CONFIG_DIR / "taxonomy_spine.yaml")

    # When vision moderation is off, don't require a vision verdict to attach media —
    # otherwise no media could ever attach. The reuse (pHash) check still gates.
    return Publisher(
        session_factory,
        telegram=telegram,
        stoplist_rules=load_stoplist(CONFIG_DIR / "stoplist.yaml"),
        limits=load_limits(CONFIG_DIR / "limits.yaml"),
        supervisor=Supervisor.from_env(telegram),
        media_store=media_store,
        purge_media_after_publish=purge,
        require_vision=media_moderation_enabled(),
        predup=predup,
        predup_enforce=prepublish_dedup_enforce(),
        spine=spine,
        enrich_on_duplicate=prepublish_enrich_enabled(),
    )


async def publish_forever(session_factory, publisher, *, tick_seconds: float = 15.0,
                          stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Publish critic-passed drafts that clear the gate, ONE at a time by default so
    posts come out spaced (most-significant first), not as a burst. Volume is decided
    upstream by curation, not here — PUBLISH_BATCH only controls how many go per tick.
    The Publisher no-ops when the master switch is off; each draft still passes the
    stop button, stop-list, limits and surge check before anything is sent."""
    try:
        batch = max(1, int(os.getenv("PUBLISH_BATCH", "1")))
    except (TypeError, ValueError):
        batch = 1
    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(publisher.publish_pending, limit=batch)
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


async def media_purge_forever(session_factory, *, tick_seconds: float = 3600.0,
                              older_than_hours: int = 48,
                              stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Periodically delete local media files that will never be (re)published (stored, old,
    not tied to an in-flight post). The post-publish purge only cleans what actually
    publishes; most collected media belongs to events that never post, so without this the
    store grows unbounded. pHash stays in the DB, so dropping files is safe."""
    from newsroom.media import LocalMediaStore, purge_stale_media

    store = LocalMediaStore(os.getenv("MEDIA_STORE_DIR", "./media"))
    while not (stop and stop.is_set()):
        try:
            removed = await asyncio.to_thread(purge_stale_media, session_factory, store,
                                              older_than_hours=older_than_hours)
            if removed:
                log.info("stale media purge tick", extra=bind(removed=removed))
        except Exception:
            log.exception("stale media purge tick failed")
        await asyncio.sleep(tick_seconds)


def build_tg_media_downloader_from_env(session_factory):  # pragma: no cover — Pillow
    """Assemble the Telegram media downloader (local store + Pillow decoder). Fetch is
    via the collector's live Telethon client, passed in at loop time."""
    from newsroom.media import LocalMediaStore, PillowDecoder, TelegramMediaDownloader

    store_dir = os.getenv("MEDIA_STORE_DIR", "./media")
    # 50MB so Telegram videos (uploaded multipart, Bot API limit 50MB) download too, not just
    # the 25MB default — TG video was mostly skipped at download before.
    return TelegramMediaDownloader(session_factory, store=LocalMediaStore(store_dir),
                                   decoder=PillowDecoder(), max_bytes=50 * 1024 * 1024)


async def tg_media_download_forever(session_factory, collector, downloader, *, tick_seconds: float = 45.0,
                                    stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Download url-less Telegram media for filter-passed items via the collector's
    shared Telethon session, then store + pHash it like HTTP media. Runs on the main
    loop (Telethon calls stay on their client's loop). Nothing is published."""
    client = await collector.wait_client()
    while not (stop and stop.is_set()):
        try:
            stats = await downloader.download_pending(client)
            if stats.get("stored"):
                log.info("tg-media tick", extra=bind(**stats))
        except Exception:
            log.exception("tg-media tick failed")
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


async def media_moderation_forever(session_factory, moderator, *, store=None, tick_seconds: float = 60.0,
                                   stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Moderate downloaded images (the media stop-list) so verified media can
    attach to posts. Telegram images have no URL, so a store is passed to moderate
    them from the stored bytes. Nothing is published."""
    from newsroom.media import moderate_pending

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(moderate_pending, session_factory, moderator, store=store)
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


async def subscriptions_sync_forever(session_factory, collector, *, tick_seconds: float = 900.0,
                                     stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Auto-register the reading account's channel subscriptions as telegram sources via the
    shared Telethon session, then LIVE-subscribe them: refresh the collector's source map and
    backfill the new channels, so a newly-added subscription starts collecting WITHOUT a
    restart (the realtime handlers listen to all chats and filter by the live map). Slow cadence."""
    from newsroom.sources.discovery import list_subscribed_channels, sync_subscriptions

    client = await collector.wait_client()
    while not (stop and stop.is_set()):
        try:
            channels = await list_subscribed_channels(client)
            stats = await asyncio.to_thread(sync_subscriptions, session_factory, channels)
            # pick up newly-added (or manually-added) sources live: refresh the map + backfill
            backfilled = await collector.sync_and_backfill_new(client)
            if stats.get("added") or backfilled:
                log.info("subscriptions synced", extra=bind(backfilled=backfilled, **stats))
        except Exception:
            log.exception("subscription sync failed")
        await asyncio.sleep(tick_seconds)


async def demand_forever(session_factory, collector, *, tick_seconds: float = 1800.0,
                         sources_every: int = 4, analytics_every: int = 2,
                         stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Snapshot engagement of monitored source posts + channel sizes via the shared
    Telethon session, then recompute the learned per-rubric demand index (DB-only) so
    curation can read it. Slow cadence. Reads only — nothing is published."""
    from newsroom.analyze.demand import DemandCollector, TelethonDemandSource, refresh_demand

    loop = asyncio.get_running_loop()
    client = await collector.wait_client()
    demand = DemandCollector(session_factory, TelethonDemandSource(client, loop))
    ticks = 0
    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(demand.collect_items)
            if ticks % sources_every == 0:
                await asyncio.to_thread(demand.collect_sources)
            if ticks % analytics_every == 0:
                by_rubric = await asyncio.to_thread(refresh_demand, session_factory)
                if by_rubric:
                    log.info("demand index", extra=bind(rubrics=len(by_rubric)))
            if stats.get("recorded"):
                log.info("demand tick", extra=bind(**stats))
        except Exception:
            log.exception("demand tick failed")
        ticks += 1
        await asyncio.sleep(tick_seconds)


async def topics_forever(session_factory, *, tick_seconds: float = 1800.0,
                         stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Recompute the taxonomy-pyramid engagement heat from competitor engagement, so
    curation and the admin console read what topics are hot now (charter v0.3 §3.2). DB-only,
    no LLM (topic paths are extracted upstream in verify). Nothing is published."""
    from newsroom.analyze.taxonomy import refresh_taxonomy_heat

    while not (stop and stop.is_set()):
        try:
            heat = await asyncio.to_thread(refresh_taxonomy_heat, session_factory)
            if heat.get("nodes"):
                log.info("taxonomy heat", extra=bind(nodes=heat["nodes"]))
        except Exception:
            log.exception("taxonomy heat tick failed")
        await asyncio.sleep(tick_seconds)


def build_synonym_grouper_from_env():  # pragma: no cover — needs OpenAI
    """Assemble the LLM synonym grouper for taxonomy-node merging."""
    from newsroom.analyze.taxonomy import LLMSynonymGrouper

    return LLMSynonymGrouper()


async def taxonomy_merge_forever(session_factory, grouper, *, tick_seconds: float = 3600.0,
                                 stop: asyncio.Event | None = None) -> None:  # pragma: no cover
    """Fold synonym sibling nodes together so the learned tree does not fragment into
    near-duplicate topics ("удар бпла" / "атака дронів"). LLM-driven, so a slow cadence.
    Nothing is published."""
    from newsroom.analyze.taxonomy import merge_synonym_nodes

    while not (stop and stop.is_set()):
        try:
            stats = await asyncio.to_thread(merge_synonym_nodes, session_factory, grouper)
            if stats.get("merged"):
                log.info("taxonomy merge", extra=bind(**stats))
        except Exception:
            log.exception("taxonomy merge tick failed")
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

    # record the cost of every LLM call (completions + embeddings) to the llm_calls table
    from newsroom.llm_recorder import make_db_recorder
    from newsroom.llmutil import set_usage_recorder

    set_usage_recorder(make_db_recorder(session_factory))

    with session_factory() as s:
        sync_sources(s, load_sources(CONFIG_PATH))
        s.commit()

    tasks = [asyncio.create_task(poll_rss_forever(session_factory))]
    telegram_collector = None
    if telegram_enabled():
        telegram_collector = TelegramCollector(session_factory)
        tasks.append(asyncio.create_task(telegram_collector.start()))
        log.info("telegram collection enabled")
        if telegram_autosync_subs_enabled():
            tasks.append(asyncio.create_task(
                subscriptions_sync_forever(session_factory, telegram_collector)))
            log.info("telegram subscription auto-sync enabled")
        else:
            log.info("telegram subscription auto-sync disabled (TELEGRAM_AUTOSYNC_SUBS off)")
    else:
        log.info("telegram collection disabled (COLLECTOR_TELEGRAM_ENABLED off)")

    if verify_enabled():
        verifier = build_verifier_from_env(session_factory)
        # recover items claimed (`processing`) but left unfinished by a previous crash
        try:
            from newsroom.analyze.verify import reset_stuck_processing

            reset = reset_stuck_processing(session_factory)
            if reset:
                log.warning("startup: reset stuck verify items", extra=bind(count=reset))
        except Exception:
            log.exception("verify processing reset failed")
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
        # Telegram media has no URL — fetch it via the collector's Telethon session.
        if telegram_collector is not None:
            tg_downloader = build_tg_media_downloader_from_env(session_factory)
            tasks.append(asyncio.create_task(
                tg_media_download_forever(session_factory, telegram_collector, tg_downloader)))
            log.info("telegram media download enabled")
        # Sweep stale local media (most collected media never publishes, so post-publish
        # purge alone lets the store grow forever). Gated by the same purge flag.
        purge = os.getenv("MEDIA_PURGE_AFTER_PUBLISH", "true").strip().lower() in {"1", "true", "yes"}
        if purge:
            tasks.append(asyncio.create_task(media_purge_forever(session_factory)))
            log.info("stale media purge enabled")
    else:
        log.info("media download disabled (MEDIA_DOWNLOAD_ENABLED off)")

    # In ENFORCE mode the LLM stages wait for the ingest-dedup verdict, so a duplicate is
    # merged before they spend tokens (Phase B savings). In observe mode there are no merges,
    # so no gating — the pipeline is untouched during calibration.
    dedup_gate = ingest_dedup_enforce()

    if factbase_enabled():
        builder = build_factbase_builder_from_env(session_factory)
        tasks.append(asyncio.create_task(factbase_forever(
            session_factory, builder, require_dedup_settled=dedup_gate)))
        log.info("fact base enabled")
    else:
        log.info("fact base disabled (FACTBASE_ENABLED off)")

    if factcheck_enabled():
        checker = build_factchecker_from_env(session_factory)
        fc_risk = ("high", "critical") if factcheck_high_only() else None
        tasks.append(asyncio.create_task(factcheck_forever(
            session_factory, checker,
            risk_levels=fc_risk, require_dedup_settled=dedup_gate)))
        log.info("fact-checking enabled", extra=bind(high_only=bool(fc_risk)))
    else:
        log.info("fact-checking disabled (FACTCHECK_ENABLED off)")

    if media_check_enabled():
        tasks.append(asyncio.create_task(media_check_forever(session_factory)))
        log.info("media check enabled")
    else:
        log.info("media check disabled (MEDIA_CHECK_ENABLED off)")

    if media_moderation_enabled():
        from newsroom.media import LocalMediaStore

        moderator = build_image_moderator_from_env()
        mod_store = LocalMediaStore(os.getenv("MEDIA_STORE_DIR", "./media"))
        tasks.append(asyncio.create_task(
            media_moderation_forever(session_factory, moderator, store=mod_store)))
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
            session_factory, updater, require_dedup_settled=dedup_gate)))
        log.info("story updates enabled")
    else:
        log.info("story updates disabled (STORY_UPDATES_ENABLED off)")

    # Ingest-time event merge (Phase B): fold duplicate events into an earlier twin right
    # after clustering, before the fact base / fact-check / editorial spend on them.
    if ingest_dedup_enabled():
        ingest_dedup = build_ingest_dedup_from_env(session_factory)
        tasks.append(asyncio.create_task(
            ingest_dedup_forever(session_factory, ingest_dedup, enforce=ingest_dedup_enforce())))
        log.info("ingest dedup enabled", extra=bind(enforce=ingest_dedup_enforce()))
    else:
        log.info("ingest dedup disabled (INGEST_DEDUP_ENABLED off)")

    # Dedup runs together with curation (dedup first, same tick) so a duplicate is
    # marked before curation can pass it on to drafting. Only when curation is off
    # does dedup need its own loop.
    grouper = build_dedup_grouper_from_env() if dedup_enabled() else None
    if dedup_enabled():
        log.info("LLM batch dedup enabled", extra=bind(with_curation=curation_enabled()))
    else:
        log.info("LLM batch dedup disabled (DEDUP_ENABLED off)")

    # Reserve attacks for the digest inside the curation tick (before must-publish can
    # grab them individually), so shelling/drone events consolidate into one digest.
    curation_digest_config = None
    if curation_enabled() and digest_enabled():
        from newsroom.editorial import load_digest_config

        curation_digest_config = load_digest_config(CONFIG_DIR / "digest.yaml")

    if curation_enabled():
        ranker = build_editorial_ranker_from_env()
        tasks.append(asyncio.create_task(curation_forever(
            session_factory, ranker, grouper=grouper, digest_config=curation_digest_config,
            require_dedup_settled=dedup_gate)))
        log.info("editorial curation enabled", extra=bind(digest_reserve=curation_digest_config is not None))
    else:
        if grouper is not None:            # curation off: dedup still needs a loop
            tasks.append(asyncio.create_task(dedup_forever(session_factory, grouper)))
        log.info("editorial curation disabled (CURATION_ENABLED off)")

    if editorial_enabled():
        pipeline = build_editorial_pipeline_from_env(session_factory)
        tasks.append(asyncio.create_task(editorial_forever(
            session_factory, pipeline,
            require_curation=curation_enabled())))
        log.info("editorial drafting enabled", extra=bind(require_curation=curation_enabled()))
    else:
        log.info("editorial drafting disabled (EDITORIAL_ENABLED off)")

    publisher = build_publisher_from_env(session_factory)
    if publisher.telegram.is_enabled():
        # Recover posts stuck mid-send from a previous crash BEFORE the publish loop starts,
        # so an ambiguous delivery is flagged for a human instead of silently re-sent.
        try:
            stuck = await asyncio.to_thread(publisher.reconcile_pending_deliveries)
            if stuck:
                log.warning("startup: reconciled stuck deliveries", extra=bind(count=stuck))
        except Exception:
            log.exception("delivery reconciliation failed")
        tasks.append(asyncio.create_task(publish_forever(session_factory, publisher)))
        log.info("publishing enabled")
        admin_raw = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()
        if admin_raw:
            from newsroom.publishers import AdminConsole, SupervisionBot

            try:
                admin_id = int(admin_raw)
            except (TypeError, ValueError):
                admin_id = None
            admin_user_raw = os.getenv("TELEGRAM_ADMIN_USER_ID", "").strip()
            try:
                admin_user_id = int(admin_user_raw) if admin_user_raw else (
                    admin_id if admin_id is not None and admin_id > 0 else None
                )
            except (TypeError, ValueError):
                admin_user_id = None
            bot = SupervisionBot(session_factory, publisher.telegram, admin_chat_id=admin_id,
                                 admin_user_id=admin_user_id,
                                 console=AdminConsole(session_factory, spine=publisher.spine))
            tasks.append(asyncio.create_task(bot.poll_forever()))
            log.info("supervision bot + admin console enabled")
        if digest_enabled():
            tasks.append(asyncio.create_task(digest_forever(session_factory, publisher)))
            log.info("attacks digest enabled")
        else:
            log.info("attacks digest disabled (DIGEST_ENABLED off)")
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

    if topics_enabled():
        tasks.append(asyncio.create_task(topics_forever(session_factory)))
        log.info("hot topics enabled")
    else:
        log.info("hot topics disabled (TOPICS_ENABLED off)")

    if taxonomy_merge_enabled():
        grouper = build_synonym_grouper_from_env()
        tasks.append(asyncio.create_task(taxonomy_merge_forever(session_factory, grouper)))
        log.info("taxonomy synonym merge enabled")
    else:
        log.info("taxonomy synonym merge disabled (TAXONOMY_MERGE_ENABLED off)")

    # A crashing task must NOT take the whole bot down with it: log its traceback and let
    # the other loops keep running (return_exceptions), so e.g. a Telethon collector failure
    # never stops publishing. (Each forever-loop already guards per-tick errors; this catches
    # a fatal one that ends the task.)
    def _on_task_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            name = getattr(task.get_coro(), "__qualname__", "task")
            log.error("background task crashed", extra=bind(task=name, error=repr(exc)), exc_info=exc)

    for t in tasks:
        t.add_done_callback(_on_task_done)
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_service())
