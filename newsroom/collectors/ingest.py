"""Persistence for collected items: idempotent upsert + source-health updates.

Idempotence key is (source_id, external_id): re-ingesting the same item never
creates a duplicate (готовності §14). A changed body is versioned, not lost
(architecture §5.1 item_versions, principle "raw data is immutable").
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.collectors.base import RawItem
from newsroom.models import Item, ItemVersion, MediaAsset, Source


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def upsert_raw_item(session: Session, raw: RawItem) -> tuple[Item, bool]:
    """Insert a new item or update an existing one. Returns (item, created)."""
    existing = session.execute(
        select(Item).where(Item.source_id == raw.source_id, Item.external_id == raw.external_id)
    ).scalar_one_or_none()

    now = utc_now()
    if existing is None:
        item = Item(
            source_id=raw.source_id,
            external_id=raw.external_id,
            url=raw.url,
            title=raw.title,
            text=raw.text,
            lang=raw.lang,
            published_at=raw.published_at,
            fetched_at=raw.fetched_at or now,
            forwarded_from=raw.forwarded_from,
            grouped_id=raw.grouped_id,
            content_hash=raw.content_hash,
            simhash=raw.simhash,
            status="new",
            raw_payload=raw.raw_payload,
        )
        session.add(item)
        session.flush()
        for m in raw.media:
            session.add(MediaAsset(
                item_id=item.id, kind=m.kind, url=m.url,
                width=m.width, height=m.height, size_bytes=m.size_bytes,
            ))
        return item, True

    # Existing item: version and update only if the body actually changed.
    if existing.content_hash != raw.content_hash:
        session.add(ItemVersion(item_id=existing.id, text=existing.text))  # snapshot previous body
        existing.title = raw.title
        existing.text = raw.text
        existing.content_hash = raw.content_hash
        existing.simhash = raw.simhash
        existing.edited_at = now
    return existing, False


def mark_source_success(source: Source) -> None:
    source.last_success_at = utc_now()
    source.last_error = None
    source.consecutive_failures = 0


def mark_source_failure(source: Source, error: object) -> None:
    source.last_error = str(error)[:500]
    source.consecutive_failures = (source.consecutive_failures or 0) + 1
