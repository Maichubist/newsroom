"""Approved-publication media download pipeline (architecture §12, §13).

Media is downloaded only after an item belongs to a Telegram publication whose
editorial critic approved it — never for noise or unpublished candidates. For each
such media asset without a stored copy: fetch the bytes,
store them behind the MediaStore abstraction, record size, and for images compute
a pHash for the reuse check (§9.4). Fetch, decode and store are all injectable, so
the pipeline is offline-tested end to end. Idempotent: an asset with a storage_key
is skipped.
"""
from __future__ import annotations

import logging
import datetime as dt
from dataclasses import dataclass
from typing import Callable

from newsroom.media.phash import phash_bytes
from newsroom.media.store import MediaStore, media_key

log = logging.getLogger("newsroom.media.download")

DEFAULT_MAX_BYTES = 25 * 1024 * 1024
MAX_DOWNLOAD_ATTEMPTS = 3
RETRY_BASE_SECONDS = 30


# A real browser UA — a bot-ish UA gets 403'd by many CDNs/sites (empty UA got 403s too).
_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def _http_fetch(url: str, *, timeout: float = 20.0) -> bytes:  # pragma: no cover - network
    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                     headers={"User-Agent": _USER_AGENT,
                              "Accept": "image/avif,image/webp,image/png,image/*,*/*;q=0.8"})
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
            hash_failed = kind == "image" and decoder is not None and phash is None
            asset.download_status = "failed" if hash_failed else "ready"
            asset.download_error = "phash_failed" if hash_failed else None
            asset.next_retry_at = None
            asset.downloaded_at = dt.datetime.now(dt.timezone.utc)
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


def record_download_failure(session_factory, media_id: int, error: str, *, terminal: bool = False) -> None:
    """Persist bounded retry/terminal state shared by HTTP and Telegram workers."""
    from newsroom.models import MediaAsset

    with session_factory() as s:
        asset = s.get(MediaAsset, media_id)
        if asset is None:
            return
        attempts = int(asset.download_attempts or 0)
        exhausted = terminal or attempts >= MAX_DOWNLOAD_ATTEMPTS
        asset.download_status = "too_large" if terminal else ("failed" if exhausted else "retry")
        asset.download_error = (error or "download_failed")[:2000]
        asset.next_retry_at = None if exhausted else (
            dt.datetime.now(dt.timezone.utc)
            + dt.timedelta(seconds=RETRY_BASE_SECONDS * (2 ** max(0, attempts - 1)))
        )
        s.commit()


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
            if asset is None or asset.storage_key or not asset.url or asset.download_status in {"failed", "too_large"}:
                return DownloadResult(media_id, skipped=True)
            now = dt.datetime.now(dt.timezone.utc)
            if asset.next_retry_at is not None and asset.next_retry_at > now:
                return DownloadResult(media_id, skipped=True)
            asset.download_attempts = int(asset.download_attempts or 0) + 1
            asset.download_status = "downloading"
            asset.download_error = None
            url, kind, item_id = asset.url, asset.kind, asset.item_id
            s.commit()

        try:
            data = self.fetch(url)
        except Exception as exc:  # noqa: BLE001
            log.warning("media fetch failed", extra={"media_id": media_id, "error": str(exc)})
            record_download_failure(self.sf, media_id, str(exc))
            return DownloadResult(media_id, error=str(exc))
        if not data or len(data) > self.max_bytes:
            record_download_failure(self.sf, media_id, "too_large_or_empty", terminal=True)
            return DownloadResult(media_id, skipped=True, error="too_large_or_empty")

        try:
            _key, hashed = persist_media_bytes(
                self.sf, self.store, self.decoder,
                media_id=media_id, item_id=item_id, key_source=url, kind=kind, data=data)
        except Exception as exc:  # noqa: BLE001 - one bad file/store write must not abort the batch
            log.warning("media persist failed", extra={"media_id": media_id, "error": str(exc)})
            record_download_failure(self.sf, media_id, str(exc))
            return DownloadResult(media_id, error=str(exc))
        return DownloadResult(media_id, stored=True, hashed=hashed)

    def download_pending(self, *, limit: int = 50) -> dict[str, int]:
        """Download media only for editorially-approved draft publications."""
        from sqlalchemy import or_, select

        from newsroom.media.state import approved_item_ids_query
        from newsroom.models import MediaAsset

        now = dt.datetime.now(dt.timezone.utc)
        with self.sf() as s:
            ids = list(s.execute(
                select(MediaAsset.id)
                .where(
                    MediaAsset.item_id.in_(approved_item_ids_query()),
                    MediaAsset.storage_key.is_(None),
                    MediaAsset.url.is_not(None),
                    MediaAsset.download_status.not_in(("failed", "too_large")),
                    or_(MediaAsset.next_retry_at.is_(None), MediaAsset.next_retry_at <= now),
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
