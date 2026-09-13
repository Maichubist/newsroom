"""Media download pipeline (architecture §12, §13).

Media is downloaded only after an item passes the filter (accepted/clustered) —
never for noise. For each such media asset without a stored copy: fetch the bytes,
store them behind the MediaStore abstraction, record size, and for images compute
a pHash for the reuse check (§9.4). Fetch, decode and store are all injectable, so
the pipeline is offline-tested end to end. Idempotent: an asset with a storage_key
is skipped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from newsroom.media.phash import phash_bytes
from newsroom.media.store import MediaStore, media_key

log = logging.getLogger("newsroom.media.download")

DEFAULT_MAX_BYTES = 25 * 1024 * 1024
_DOWNLOADABLE_ITEM_STATUSES = ("accepted", "clustered")


_USER_AGENT = "Mozilla/5.0 (compatible; newsroom/0.1; +https://example.org)"


def _http_fetch(url: str, *, timeout: float = 20.0) -> bytes:  # pragma: no cover - network
    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                     headers={"User-Agent": _USER_AGENT})
    resp.raise_for_status()
    return resp.content


def persist_media_bytes(session_factory, store, decoder, *, media_id: int, item_id: int,
                        key_source: str, kind: str, data: bytes) -> tuple[str, bool]:
    """Store fetched bytes behind the MediaStore, set storage_key + size, and for an
    image compute + save the pHash (reuse check). Shared by the HTTP and Telethon
    downloaders. `key_source` is any stable string folded into the storage key (a URL
    for RSS, the tg message ref for Telegram). Returns (storage_key, hashed)."""
    from newsroom.models import MediaAsset

    key = store.put(media_key(item_id, media_id, key_source or ""), data)
    phash = phash_bytes(data, decoder) if (kind == "image" and decoder is not None) else None
    with session_factory() as s:
        asset = s.get(MediaAsset, media_id)
        if asset is not None:
            asset.storage_key = key
            asset.size_bytes = len(data)
            if phash is not None:
                asset.phash = phash
            s.commit()
    return key, phash is not None


@dataclass(frozen=True)
class DownloadResult:
    media_id: int
    stored: bool = False
    hashed: bool = False
    skipped: bool = False
    error: str | None = None


class MediaDownloader:
    def __init__(self, session_factory, *, store: MediaStore, decoder=None,
                 fetch: Callable[[str], bytes] | None = None, max_bytes: int = DEFAULT_MAX_BYTES):
        self.sf = session_factory
        self.store = store
        self.decoder = decoder
        self.fetch = fetch or _http_fetch
        self.max_bytes = max_bytes

    def download_asset(self, media_id: int) -> DownloadResult:
        from newsroom.models import MediaAsset

        with self.sf() as s:
            asset = s.get(MediaAsset, media_id)
            if asset is None or asset.storage_key or not asset.url:
                return DownloadResult(media_id, skipped=True)
            url, kind, item_id = asset.url, asset.kind, asset.item_id

        try:
            data = self.fetch(url)
        except Exception as exc:  # noqa: BLE001
            log.warning("media fetch failed", extra={"media_id": media_id, "error": str(exc)})
            return DownloadResult(media_id, error=str(exc))
        if not data or len(data) > self.max_bytes:
            return DownloadResult(media_id, skipped=True, error="too_large_or_empty")

        _key, hashed = persist_media_bytes(
            self.sf, self.store, self.decoder,
            media_id=media_id, item_id=item_id, key_source=url, kind=kind, data=data)
        return DownloadResult(media_id, stored=True, hashed=hashed)

    def download_pending(self, *, limit: int = 50) -> dict[str, int]:
        """Download media for filter-passed items that has not been stored yet."""
        from sqlalchemy import select

        from newsroom.models import Item, MediaAsset

        with self.sf() as s:
            ids = list(s.execute(
                select(MediaAsset.id)
                .join(Item, Item.id == MediaAsset.item_id)
                .where(
                    Item.status.in_(_DOWNLOADABLE_ITEM_STATUSES),
                    MediaAsset.storage_key.is_(None),
                    MediaAsset.url.is_not(None),
                )
                .order_by(MediaAsset.id)
                .limit(limit)
            ).scalars().all())

        stats = {"stored": 0, "hashed": 0, "errors": 0}
        for media_id in ids:
            result = self.download_asset(media_id)
            if result.stored:
                stats["stored"] += 1
                stats["hashed"] += 1 if result.hashed else 0
            elif result.error:
                stats["errors"] += 1
        return stats
