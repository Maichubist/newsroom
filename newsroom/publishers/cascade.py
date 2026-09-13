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
    # Below this known width an image is a logo/icon/thumbnail, not a news photo —
    # attaching it looks worse than no image. Unknown width is still allowed
    # (Telegram fetches it), so we don't silently drop every sizeless asset.
    min_image_width: int = 400


@dataclass(frozen=True)
class MediaItem:
    kind: str                       # image | video | embed
    url: str | None = None
    width: int | None = None
    size_bytes: int | None = None
    storage_key: str | None = None  # local stored file (Telegram media, sent by upload)


@dataclass(frozen=True)
class MediaChoice:
    method: str                     # sendPhoto | sendVideo
    param: str                      # photo | video
    url: str | None = None          # public URL (Telegram fetches it) ...
    storage_key: str | None = None  # ... or a local file to upload (Telegram media)


def _within(size_bytes: int | None, limit: int) -> bool:
    # unknown size: allow and let Telegram enforce (asymmetry: dropping media is cheap,
    # but a missing size should not silently kill every attachment)
    return size_bytes is None or size_bytes <= limit


def _sendable(i: MediaItem) -> bool:
    # a URL Telegram can fetch, or a local file we can upload
    return bool(i.url or i.storage_key)


def choose_media(items: list[MediaItem], limits: MediaLimits | None = None) -> MediaChoice | None:
    """video → widest image → nothing. Only items sendable (a URL or a stored file) and
    within the size limit are eligible; embeds are never sent."""
    limits = limits or MediaLimits()

    videos = [i for i in items if i.kind == "video" and _sendable(i) and _within(i.size_bytes, limits.video_max_bytes)]
    if videos:
        v = videos[0]
        return MediaChoice(method="sendVideo", param="video", url=v.url, storage_key=v.storage_key)

    images = [
        i for i in items
        if i.kind == "image" and _sendable(i)
        and _within(i.size_bytes, limits.photo_max_bytes)
        and _wide_enough(i.width, limits.min_image_width)
    ]
    if images:
        widest = max(images, key=lambda i: (i.width or 0))
        return MediaChoice(method="sendPhoto", param="photo", url=widest.url, storage_key=widest.storage_key)

    return None


def _wide_enough(width: int | None, min_width: int) -> bool:
    # unknown width: allow (Telegram fetches it); known but tiny: reject as a logo/icon
    return width is None or width >= min_width
