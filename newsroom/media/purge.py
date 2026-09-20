"""Delete local media files after publication.

The stored bytes are sent to Telegram by URL and are never read locally again
(the pHash reuse-archive lives in the DB, moderation reads the URL — see
media/download.py). So once a post is out, its local media files are dead weight;
purging them right after publish keeps the store small. Non-destructive to the
record: storage_key is kept (so nothing re-downloads) and purged_at is stamped.
"""
from __future__ import annotations

import datetime as dt
import logging

log = logging.getLogger("newsroom.media.purge")


def purge_event_media(session_factory, store, event_id: int) -> int:
    """Delete the local stored files of a published event's media. Returns how many
    files were removed. Idempotent — assets already purged are skipped, and a missing
    file is a no-op. A store/IO error on one asset never raises: it is logged and the
    asset is still marked purged (the file is gone or unreachable either way)."""
    from sqlalchemy import select

    from newsroom.models import EventItem, MediaAsset

    if event_id is None:
        return 0
    removed = 0
    now = dt.datetime.now(dt.timezone.utc)
    with session_factory() as s:
        assets = list(s.execute(
            select(MediaAsset)
            .join(EventItem, EventItem.item_id == MediaAsset.item_id)
            .where(
                EventItem.event_id == event_id,
                MediaAsset.storage_key.is_not(None),
                MediaAsset.purged_at.is_(None),
            )
        ).scalars().all())
        for asset in assets:
            try:
                if store.delete(asset.storage_key):
                    removed += 1
            except Exception as exc:  # noqa: BLE001 — cleanup must never fail a publish
                log.warning("media purge failed", extra={"media_id": asset.id, "error": str(exc)})
            asset.purged_at = now
        s.commit()
    if removed:
        log.info("purged published media", extra={"event_id_": event_id, "removed": removed})
    return removed


def purge_stale_media(session_factory, store, *, older_than_hours: int = 48,
                      limit: int = 500) -> int:
    """Delete local files for media that will never be (re)published: stored, not yet purged,
    older than `older_than_hours`, and NOT tied to an in-flight publication (draft/publishing/
    review). Media is downloaded for every filter-passed item, but the post-publish purge only
    cleans the tiny fraction that actually publishes — the rest (held/duplicate/uncurated
    events) would otherwise grow the store forever. The pHash reuse-archive lives in the DB,
    so dropping files is safe: storage_key is kept (no re-download) and purged_at stamped.
    Runs on a slow cadence. Returns how many files were removed."""
    from sqlalchemy import exists, select

    from newsroom.models import EventItem, MediaAsset, Publication

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=older_than_hours)
    now = dt.datetime.now(dt.timezone.utc)
    # correlated: this asset's item belongs to an event with a not-yet-terminal publication
    in_flight = (
        select(1).select_from(EventItem)
        .join(Publication, Publication.event_id == EventItem.event_id)
        .where(EventItem.item_id == MediaAsset.item_id,
               Publication.status.in_(("draft", "publishing", "review")))
    )
    removed = 0
    with session_factory() as s:
        assets = list(s.execute(
            select(MediaAsset)
            .where(MediaAsset.storage_key.is_not(None), MediaAsset.purged_at.is_(None),
                   MediaAsset.first_seen_at < cutoff, ~exists(in_flight))
            .limit(limit)
        ).scalars().all())
        for asset in assets:
            try:
                if store.delete(asset.storage_key):
                    removed += 1
            except Exception as exc:  # noqa: BLE001 — cleanup must never crash the loop
                log.warning("stale media purge failed", extra={"media_id": asset.id, "error": str(exc)})
            asset.purged_at = now
        s.commit()
    if removed:
        log.info("purged stale media", extra={"removed": removed, "older_than_hours": older_than_hours})
    return removed
