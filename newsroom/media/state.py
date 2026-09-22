"""Persistent media readiness for editorially-approved publications.

The collector records lightweight MediaAsset metadata for every source item, but
network downloads start only after a Telegram draft passes the editorial critic.
This module is the shared source of truth used by discovery, download, checking,
and publishing workers.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


DOWNLOAD_PENDING = "pending"
DOWNLOAD_RETRY = "retry"
DOWNLOAD_DOWNLOADING = "downloading"
DOWNLOAD_READY = "ready"
DOWNLOAD_FAILED = "failed"
DOWNLOAD_TOO_LARGE = "too_large"
DOWNLOAD_TERMINAL = frozenset({DOWNLOAD_FAILED, DOWNLOAD_TOO_LARGE})

PUBLICATION_MEDIA_NONE = "none"
PUBLICATION_MEDIA_DISCOVERING = "discovering"
PUBLICATION_MEDIA_PENDING = "pending"
PUBLICATION_MEDIA_CHECKING = "checking"
PUBLICATION_MEDIA_READY = "ready"
PUBLICATION_MEDIA_UNAVAILABLE = "unavailable"
PUBLICATION_MEDIA_BLOCKED = "blocked"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def approved_event_ids_query():
    """Event ids whose Telegram draft passed final text gate + prepublish dedup.

    Duplicate/digest events are excluded so media work is not spent on drafts that
    the publisher is already guaranteed to skip.
    """
    from sqlalchemy import select

    from newsroom.models import Event, Publication

    return (
        select(Publication.event_id)
        .join(Event, Event.id == Publication.event_id)
        .where(
            Publication.status == "draft",
            Publication.channel == "telegram",
            Publication.event_id.is_not(None),
            Publication.features["critic_ok"].as_boolean().is_(True),
            Publication.media_approved_at.is_not(None),
            Event.duplicate_of.is_(None),
            Event.curated.is_distinct_from("digest"),
            Event.curated.is_distinct_from("digested"),
        )
        .distinct()
    )


def approved_item_ids_query():
    from sqlalchemy import select

    from newsroom.models import EventItem

    return (
        select(EventItem.item_id)
        .where(EventItem.event_id.in_(approved_event_ids_query()))
        .distinct()
    )


@dataclass(frozen=True)
class MediaReadiness:
    event_id: int
    has_media: bool = False
    expected: int = 0
    ready: int = 0
    failed: int = 0
    pending: int = 0
    discovery_pending: bool = False

    @property
    def downloads_settled(self) -> bool:
        return not self.discovery_pending and self.pending == 0


def event_media_readiness(session, event_id: int) -> MediaReadiness:
    """Return live download/discovery state for one event.

    Only attachable image/video assets count. A legacy row with storage_key is
    treated as ready even if it predates download_status. RSS items without an
    asset remain in discovery until og:image records og_image or og_none.
    """
    from sqlalchemy import String, cast, select

    from newsroom.models import Decision, EventItem, Item, MediaAsset, Source

    item_ids = list(session.execute(
        select(EventItem.item_id).where(EventItem.event_id == event_id)
    ).scalars().all())
    if not item_ids:
        return MediaReadiness(event_id)

    rows = list(session.execute(
        select(MediaAsset.download_status, MediaAsset.storage_key, MediaAsset.purged_at,
               MediaAsset.url, MediaAsset.source_ref)
        .where(MediaAsset.item_id.in_(item_ids), MediaAsset.kind.in_(("image", "video")))
    ).all())

    ready = failed = pending = 0
    for status, storage_key, purged_at, url, source_ref in rows:
        if status in DOWNLOAD_TERMINAL or purged_at is not None or not (url or source_ref):
            failed += 1
        elif storage_key:
            ready += 1
        else:
            pending += 1

    # An RSS/site article without feed media may still expose og:image. The resolver
    # writes an explicit og_image/og_none decision, which closes this discovery phase.
    checked_items = set(session.execute(
        select(cast(Decision.entity_id, String)).where(
            Decision.entity_type == "item",
            Decision.stage == "media",
            Decision.decision.in_(("og_image", "og_none")),
        )
    ).scalars().all())
    rss_without_asset = list(session.execute(
        select(Item.id)
        .join(Source, Source.id == Item.source_id)
        .where(
            Item.id.in_(item_ids),
            Source.kind.in_(("rss", "site")),
            Item.url.is_not(None),
            ~Item.id.in_(select(MediaAsset.item_id)),
        )
    ).scalars().all())
    discovery_pending = any(str(item_id) not in checked_items for item_id in rss_without_asset)
    return MediaReadiness(
        event_id=event_id,
        has_media=bool(rows),
        expected=len(rows),
        ready=ready,
        failed=failed,
        pending=pending,
        discovery_pending=discovery_pending,
    )


def persist_publication_media_state(session, publication, readiness: MediaReadiness,
                                    *, checked: bool = False, blocked: bool = False) -> str:
    """Persist the current snapshot and return publication.media_status."""
    publication.has_media = readiness.has_media
    publication.media_expected_count = readiness.expected
    publication.media_ready_count = readiness.ready
    publication.media_failed_count = readiness.failed
    if readiness.discovery_pending:
        status = PUBLICATION_MEDIA_DISCOVERING
    elif not readiness.has_media:
        status = PUBLICATION_MEDIA_NONE
    elif not readiness.downloads_settled:
        status = PUBLICATION_MEDIA_PENDING
    elif readiness.ready == 0:
        status = PUBLICATION_MEDIA_UNAVAILABLE
    elif blocked:
        status = PUBLICATION_MEDIA_BLOCKED
    elif not checked:
        status = PUBLICATION_MEDIA_CHECKING
    else:
        status = PUBLICATION_MEDIA_READY
    publication.media_status = status
    return status
