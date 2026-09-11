"""Media send cascade (architecture §13).

A post leads with media when it can: try a video, else the best (widest) image,
else text only. Telegram fetches media by URL, so nothing needs downloading here;
size limits are enforced against known sizes (Telegram enforces the rest). The
selection is pure and offline-tested; the actual sending lives on the adapter.

Attaching media is conservative: the Publisher only offers assets from an event
whose media passed the §9.4 reuse check, because there is no image stop-list
(vision) yet and, in doubt, we do not publish (§3.5).
"""
from __future__ import annotations

from dataclasses import dataclass

# Telegram limits for media sent by URL.
PHOTO_MAX_BYTES = 5 * 1024 * 1024
VIDEO_MAX_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class MediaLimits:
    photo_max_bytes: int = PHOTO_MAX_BYTES
    video_max_bytes: int = VIDEO_MAX_BYTES


@dataclass(frozen=True)
class MediaItem:
    kind: str                       # image | video | embed
    url: str | None = None
    width: int | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class MediaChoice:
    method: str                     # sendPhoto | sendVideo
    param: str                      # photo | video
    url: str


def _within(size_bytes: int | None, limit: int) -> bool:
    # unknown size: allow and let Telegram enforce (asymmetry: dropping media is cheap,
    # but a missing size should not silently kill every attachment)
    return size_bytes is None or size_bytes <= limit


def choose_media(items: list[MediaItem], limits: MediaLimits | None = None) -> MediaChoice | None:
    """video → widest image → nothing. Only items with a URL and within the size
    limit are eligible; embeds are never sent."""
    limits = limits or MediaLimits()

    videos = [i for i in items if i.kind == "video" and i.url and _within(i.size_bytes, limits.video_max_bytes)]
    if videos:
        return MediaChoice(method="sendVideo", param="video", url=videos[0].url)  # type: ignore[arg-type]

    images = [i for i in items if i.kind == "image" and i.url and _within(i.size_bytes, limits.photo_max_bytes)]
    if images:
        widest = max(images, key=lambda i: (i.width or 0))
        return MediaChoice(method="sendPhoto", param="photo", url=widest.url)  # type: ignore[arg-type]

    return None
