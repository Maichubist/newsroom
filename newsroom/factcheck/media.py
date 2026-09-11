"""Media reuse check (architecture §9, step 4).

A recycled image — the same picture attached earlier to a different, unrelated
item — is a classic disinformation tell: old footage passed off as new. We catch
it with perceptual hashing: media_assets.phash (a 64-bit pHash stored hex) is
compared by Hamming distance against the archive; a near-identical hash first
seen earlier under another item is flagged.

phash is only populated once media is downloaded (post-filter, stage 1г), so this
runs clean until then — the comparison logic is the part that must be correct now.
The distance/selection core is pure and offline-tested; the archive query and the
event check are the DB parts. A media asset with no phash is skipped, never guessed.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Iterable

from newsroom.collectors.base import hamming_distance

log = logging.getLogger("newsroom.factcheck.media")

DEFAULT_PHASH_MAX_DISTANCE = 6   # <= ~6 of 64 bits differ -> visually near-identical


def phash_to_int(phash: str | None) -> int | None:
    if not phash:
        return None
    try:
        return int(phash.strip(), 16)
    except (ValueError, AttributeError):
        return None


def phash_distance(a: str | None, b: str | None) -> int | None:
    """Hamming distance between two hex pHashes; None if either is unusable."""
    ia, ib = phash_to_int(a), phash_to_int(b)
    if ia is None or ib is None:
        return None
    return hamming_distance(ia, ib)


@dataclass(frozen=True)
class MediaMatch:
    media_id: int
    item_id: int
    distance: int
    first_seen_at: dt.datetime | None = None


def select_reused(
    candidates: Iterable[tuple[int, int, str, dt.datetime | None]],
    target_phash: str, *, max_distance: int = DEFAULT_PHASH_MAX_DISTANCE,
    exclude_item_ids: Iterable[int] = (),
) -> list[MediaMatch]:
    """From (media_id, item_id, phash, first_seen_at) candidates, keep those whose
    pHash is within `max_distance` of the target and not from an excluded item,
    nearest first. This is the pure core of the archive search."""
    exclude = {int(i) for i in exclude_item_ids}
    out: list[MediaMatch] = []
    for media_id, item_id, phash, first_seen_at in candidates:
        if int(item_id) in exclude:
            continue
        dist = phash_distance(target_phash, phash)
        if dist is None or dist > max_distance:
            continue
        out.append(MediaMatch(media_id=int(media_id), item_id=int(item_id),
                              distance=dist, first_seen_at=first_seen_at))
    out.sort(key=lambda m: m.distance)
    return out


def find_reused_media(session_factory, target_phash: str, *,
                      max_distance: int = DEFAULT_PHASH_MAX_DISTANCE,
                      exclude_item_ids: Iterable[int] = (),
                      before: dt.datetime | None = None) -> list[MediaMatch]:
    """Search the media archive for near-identical pHashes. `before` restricts to
    media first seen earlier (recycled-image detection)."""
    from sqlalchemy import select

    from newsroom.models import MediaAsset

    if phash_to_int(target_phash) is None:
        return []
    query = select(MediaAsset.id, MediaAsset.item_id, MediaAsset.phash, MediaAsset.first_seen_at).where(
        MediaAsset.phash.is_not(None)
    )
    if before is not None:
        query = query.where(MediaAsset.first_seen_at < before)
    with session_factory() as s:
        rows = s.execute(query).all()
    return select_reused(rows, target_phash, max_distance=max_distance, exclude_item_ids=exclude_item_ids)


@dataclass(frozen=True)
class MediaCheckResult:
    event_id: int
    checked: int = 0        # media assets with a usable phash
    reused: int = 0         # assets found reused from earlier, other items
    skipped: bool = False


class MediaChecker:
    def __init__(self, session_factory, *, max_distance: int = DEFAULT_PHASH_MAX_DISTANCE,
                 charter_version: str = "0.2"):
        self.sf = session_factory
        self.max_distance = max_distance
        self.charter_version = charter_version

    def check_event(self, event_id: int) -> MediaCheckResult:
        from sqlalchemy import select

        from newsroom.models import Decision, EventItem, MediaAsset

        with self.sf() as s:
            if _already_checked(s, event_id):
                return MediaCheckResult(event_id, skipped=True)
            own_item_ids = list(s.execute(
                select(EventItem.item_id).where(EventItem.event_id == event_id)
            ).scalars().all())
            assets = list(s.execute(
                select(MediaAsset.id, MediaAsset.item_id, MediaAsset.phash, MediaAsset.first_seen_at)
                .where(MediaAsset.item_id.in_(own_item_ids), MediaAsset.phash.is_not(None))
            ).all()) if own_item_ids else []

        checked = 0
        flags: list[dict] = []
        for media_id, item_id, phash, first_seen_at in assets:
            checked += 1
            matches = find_reused_media(
                self.sf, phash, max_distance=self.max_distance,
                exclude_item_ids=own_item_ids, before=first_seen_at,
            )
            if matches:
                flags.append({"media_id": media_id, "item_id": item_id,
                              "matches": [{"media_id": m.media_id, "item_id": m.item_id,
                                           "distance": m.distance} for m in matches[:5]]})

        with self.sf() as s:
            s.add(Decision(
                entity_type="event", entity_id=str(event_id), stage="verify",
                decision="media_reuse" if flags else "media_clean",
                reason=f"{len(flags)} reused of {checked} checked" if checked else "no media with phash",
                details={"checked": checked, "flags": flags},
                charter_version=self.charter_version,
            ))
            s.commit()

        return MediaCheckResult(event_id, checked=checked, reused=len(flags))


def check_media_pending(session_factory, checker: "MediaChecker", *, limit: int = 25) -> dict[str, int]:
    """One media-check tick: check publishable events that own hashed media and
    have not been checked. Text-only events are ignored (nothing to hash), so
    this stays dormant until media is downloaded (stage 1г)."""
    from sqlalchemy import Integer, cast, select

    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset

    with session_factory() as s:
        checked = (
            select(cast(Decision.entity_id, Integer))
            .where(Decision.entity_type == "event", Decision.stage == "verify",
                   Decision.decision.in_(("media_clean", "media_reuse")))
        )
        ids = list(s.execute(
            select(Event.id)
            .join(EventItem, EventItem.event_id == Event.id)
            .join(Item, Item.id == EventItem.item_id)
            .join(MediaAsset, MediaAsset.item_id == Item.id)
            .where(
                Event.status.in_(("reported", "confirmed", "rumor")),
                MediaAsset.phash.is_not(None),
                Event.id.not_in(checked),
            )
            .order_by(Event.id).distinct().limit(limit)
        ).scalars().all())

    stats = {"events": 0, "reused": 0}
    for event_id in ids:
        result = checker.check_event(event_id)
        if result.skipped:
            continue
        stats["events"] += 1
        stats["reused"] += result.reused
    return stats


def _already_checked(session, event_id: int) -> bool:
    from sqlalchemy import func, select

    from newsroom.models import Decision

    return bool(session.scalar(
        select(func.count()).select_from(Decision).where(
            Decision.entity_type == "event",
            Decision.entity_id == str(event_id),
            Decision.stage == "verify",
            Decision.decision.in_(("media_clean", "media_reuse")),
        )
    ))
