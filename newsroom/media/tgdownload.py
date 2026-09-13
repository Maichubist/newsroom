"""Telegram media download (architecture §12, §13).

Telegram media has no public URL, so the URL-based MediaDownloader skips it. Here we
fetch the bytes via Telethon — reusing the collector's single session — for
filter-passed items, then store them behind the MediaStore + pHash exactly like HTTP
media. After that the reuse check, vision moderation and the publish cascade treat
Telegram media uniformly (moderation reads the stored file, publishing uploads it).

The DB selection is pure/pg-tested; the Telethon orchestration (network) is not
unit-tested — it is thin wiring behind COLLECTOR_TELEGRAM_ENABLED.
"""
from __future__ import annotations

import logging

from newsroom.collectors.telegram import run_with_floodwait
from newsroom.media.download import DEFAULT_MAX_BYTES, persist_media_bytes

log = logging.getLogger("newsroom.media.tgdownload")

_DOWNLOADABLE_ITEM_STATUSES = ("accepted", "clustered")


def select_pending_tg_media(session_factory, *, limit: int = 50):
    """URL-less Telegram media of filter-passed items that has not been stored yet.
    Returns rows of (media_id, item_id, kind, source_ref, channel_handle)."""
    from sqlalchemy import select

    from newsroom.models import Item, MediaAsset, Source

    with session_factory() as s:
        rows = s.execute(
            select(MediaAsset.id, MediaAsset.item_id, MediaAsset.kind,
                   MediaAsset.source_ref, Source.handle_or_url)
            .join(Item, Item.id == MediaAsset.item_id)
            .join(Source, Source.id == Item.source_id)
            .where(
                Source.kind == "telegram",
                MediaAsset.url.is_(None),
                MediaAsset.storage_key.is_(None),
                MediaAsset.source_ref.is_not(None),
                Item.status.in_(_DOWNLOADABLE_ITEM_STATUSES),
            )
            .order_by(MediaAsset.id)
            .limit(limit)
        ).all()
    return [(int(mid), int(iid), kind, ref, handle) for mid, iid, kind, ref, handle in rows]


class TelegramMediaDownloader:
    """Downloads url-less Telegram media through a live Telethon client. Construction
    is side-effect-free; `download_pending` is the only network part."""

    def __init__(self, session_factory, *, store, decoder=None, max_bytes: int = DEFAULT_MAX_BYTES):
        self.sf = session_factory
        self.store = store
        self.decoder = decoder
        self.max_bytes = max_bytes
        self._entities: dict[str, object] = {}

    async def _entity(self, client, handle: str):  # pragma: no cover - network
        key = handle.lstrip("@").lower()
        if key not in self._entities:
            self._entities[key] = await client.get_entity(handle)
        return self._entities[key]

    async def download_pending(self, client, *, limit: int = 50, sleeper=None) -> dict[str, int]:  # pragma: no cover - network
        """Fetch + store one batch of pending Telegram media via the shared client.
        FloodWait is waited out, not fatal; one bad asset never aborts the batch."""
        import asyncio

        rows = await asyncio.to_thread(select_pending_tg_media, self.sf, limit=limit)
        stats = {"stored": 0, "errors": 0, "skipped": 0}
        for media_id, item_id, kind, ref, handle in rows:
            try:
                entity = await self._entity(client, handle)

                async def _fetch(entity=entity, ref=ref):
                    msg = await client.get_messages(entity, ids=int(ref))
                    if msg is None or getattr(msg, "media", None) is None:
                        return None
                    return await client.download_media(msg, file=bytes)

                data = await run_with_floodwait(_fetch, sleeper=sleeper)
            except Exception as exc:  # noqa: BLE001 — a dead asset must not abort the batch
                log.warning("tg media fetch failed", extra={"media_id": media_id, "error": str(exc)})
                stats["errors"] += 1
                continue
            if not data or len(data) > self.max_bytes:
                stats["skipped"] += 1
                continue
            await asyncio.to_thread(
                persist_media_bytes, self.sf, self.store, self.decoder,
                media_id=media_id, item_id=item_id,
                key_source=f"tg:{handle}:{ref}", kind=kind, data=data)
            stats["stored"] += 1
        if stats["stored"]:
            log.info("tg media stored", extra={"stored": stats["stored"]})
        return stats
