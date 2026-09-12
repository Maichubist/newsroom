"""Telegram collector (architecture §12).

Layered for testability:
  * PURE (stdlib only): message_to_raw_item, extract_*, AlbumBuffer, merge_album,
    floodwait_seconds, run_with_floodwait — unit-tested with fake objects, no network.
  * DB (lazy SQLAlchemy import): backfill_cursor, mark_item_deleted — pg-tested.
  * NETWORK (lazy Telethon import): TelegramCollector — thin wiring behind the
    COLLECTOR_TELEGRAM_ENABLED flag; not unit-tested (needs a live account).

Media is NOT downloaded in 1a: media_assets are recorded with url/metadata but
storage_key stays NULL until an item passes the filter (§12, §5.1).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from typing import Awaitable, Callable

from newsroom.collectors.base import RawItem, RawMedia

log = logging.getLogger("newsroom.collectors.telegram")


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# --------------------------------------------------------------------------- #
# PURE: message -> unified item
# --------------------------------------------------------------------------- #

def extract_forwarded_from(msg) -> str | None:
    """First-source of a forward, for the independence check (§9). Defensive:
    reads whichever of Telethon's forward shapes is present."""
    fwd = getattr(msg, "forward", None)
    if fwd is not None:
        for attr in ("chat", "sender"):
            obj = getattr(fwd, attr, None)
            if obj is not None:
                uname = getattr(obj, "username", None)
                if uname:
                    return f"@{uname}"
                title = getattr(obj, "title", None) or getattr(obj, "first_name", None)
                if title:
                    return str(title)
        from_name = getattr(fwd, "from_name", None)
        if from_name:
            return str(from_name)
    fh = getattr(msg, "fwd_from", None)
    if fh is not None:
        from_name = getattr(fh, "from_name", None)
        if from_name:
            return str(from_name)
    return None


def extract_media(msg) -> list[RawMedia]:
    """Record media presence/metadata only. No URLs, no download in 1a."""
    out: list[RawMedia] = []
    if getattr(msg, "photo", None) is not None:
        out.append(RawMedia(kind="image", url=None))
    video = getattr(msg, "video", None)
    doc = getattr(msg, "document", None)
    if video is not None:
        out.append(RawMedia(kind="video", url=None, size_bytes=getattr(video, "size", None)))
    elif doc is not None and str(getattr(doc, "mime_type", "") or "").startswith("video"):
        out.append(RawMedia(kind="video", url=None, size_bytes=getattr(doc, "size", None)))
    return out


def message_to_raw_item(msg, source_id: int, *, channel_username: str | None = None) -> RawItem:
    """Map one Telethon message to a RawItem. Album members map individually and
    are merged later by AlbumBuffer."""
    mid = getattr(msg, "id", None)
    grouped = getattr(msg, "grouped_id", None)
    text = getattr(msg, "message", None) or getattr(msg, "text", None)
    date = getattr(msg, "date", None)
    url = f"https://t.me/{channel_username}/{mid}" if channel_username and mid is not None else None
    return RawItem(
        source_id=source_id,
        external_id=str(mid),
        url=url,
        title=None,
        text=text,
        published_at=date,
        fetched_at=_utc_now(),
        forwarded_from=extract_forwarded_from(msg),
        grouped_id=int(grouped) if grouped else None,
        media=extract_media(msg),
        raw_payload={
            "id": mid,
            "grouped_id": int(grouped) if grouped else None,
            "date": date.isoformat() if isinstance(date, dt.datetime) else None,
            "has_forward": getattr(msg, "forward", None) is not None or getattr(msg, "fwd_from", None) is not None,
        },
    )


# --------------------------------------------------------------------------- #
# PURE: album aggregation (grouped_id -> one item)
# --------------------------------------------------------------------------- #

def merge_album(items: list[RawItem]) -> RawItem:
    """Merge album members into one item: media concatenated, caption kept,
    identity = the earliest message id (stable across restarts)."""
    ordered = sorted(items, key=lambda r: int(r.external_id))
    first = ordered[0]
    media = [m for r in ordered for m in r.media]
    text = next((r.text for r in ordered if r.text), None)
    return RawItem(
        source_id=first.source_id,
        external_id=first.external_id,
        url=first.url,
        title=None,
        text=text,
        published_at=first.published_at,
        fetched_at=first.fetched_at,
        forwarded_from=next((r.forwarded_from for r in ordered if r.forwarded_from), None),
        grouped_id=first.grouped_id,
        media=media,
        raw_payload={"album_size": len(ordered),
                     "member_ids": [int(r.external_id) for r in ordered]},
    )


class AlbumBuffer:
    """Buffers album members by grouped_id and flushes a group once no new member
    has arrived for `debounce_seconds` (the album is complete)."""

    def __init__(self, debounce_seconds: float = 2.0):
        self.debounce = debounce_seconds
        self._groups: dict[int, dict] = {}

    def add(self, raw: RawItem, now: dt.datetime | None = None) -> None:
        now = now or _utc_now()
        g = self._groups.setdefault(int(raw.grouped_id), {"items": [], "last": now})
        g["items"].append(raw)
        g["last"] = now

    def flush_ready(self, now: dt.datetime | None = None) -> list[RawItem]:
        now = now or _utc_now()
        ready, keep = [], {}
        for gid, g in self._groups.items():
            if (now - g["last"]).total_seconds() >= self.debounce:
                ready.append(merge_album(g["items"]))
            else:
                keep[gid] = g
        self._groups = keep
        return ready

    def flush_all(self) -> list[RawItem]:
        out = [merge_album(g["items"]) for g in self._groups.values()]
        self._groups = {}
        return out


# --------------------------------------------------------------------------- #
# PURE: FloodWait handling (§12 — wait, do not crash)
# --------------------------------------------------------------------------- #

def floodwait_seconds(exc: BaseException) -> int | None:
    """Return the wait seconds if `exc` is a Telethon FloodWaitError-like error,
    else None. Duck-typed so tests need no telethon."""
    if "floodwait" in type(exc).__name__.lower():
        secs = getattr(exc, "seconds", None)
        if secs is not None:
            try:
                return int(secs)
            except (TypeError, ValueError):
                return None
    return None


async def run_with_floodwait(
    fn: Callable[[], Awaitable],
    *,
    retries: int = 5,
    sleeper: Callable[[float], Awaitable] | None = None,
):
    """Call an async op, waiting out FloodWait errors instead of failing."""
    sleeper = sleeper or asyncio.sleep
    last: BaseException | None = None
    for attempt in range(retries):
        try:
            return await fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised unless it is a FloodWait
            secs = floodwait_seconds(exc)
            if secs is None:
                raise
            last = exc
            if attempt == retries - 1:
                break
            log.warning("floodwait, waiting", extra={"seconds": secs, "attempt": attempt + 1})
            await sleeper(secs)
    assert last is not None
    raise last


# --------------------------------------------------------------------------- #
# DB: backfill cursor + deletions
# --------------------------------------------------------------------------- #

def backfill_cursor(session, source_id: int) -> int:
    """Highest stored message id for a Telegram source. Backfill fetches ids above
    it after downtime (§12). 0 when the channel has never been collected."""
    from sqlalchemy import select

    from newsroom.models import Item

    rows = session.scalars(select(Item.external_id).where(Item.source_id == source_id)).all()
    ids = [int(x) for x in rows if str(x).isdigit()]
    return max(ids) if ids else 0


def mark_item_deleted(session, source_id: int, external_id: str) -> bool:
    """Flag a deleted channel post (§12: deletions → deleted_at). Returns True if found."""
    from sqlalchemy import select

    from newsroom.models import Item

    item = session.execute(
        select(Item).where(Item.source_id == source_id, Item.external_id == str(external_id))
    ).scalar_one_or_none()
    if item is None:
        return False
    if item.deleted_at is None:
        item.deleted_at = _utc_now()
    return True


# --------------------------------------------------------------------------- #
# NETWORK: Telethon-backed collector (behind flag; not unit-tested)
# --------------------------------------------------------------------------- #

def telegram_enabled() -> bool:
    return os.getenv("COLLECTOR_TELEGRAM_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


class TelegramCollector:
    """Realtime + backfill reader.

    Construction is side-effect-free (no flag, no network) so the orchestration —
    backfill, new/edit/delete handlers, album flushing — is testable through the
    small `client` interface (`iter_messages`) with a fake. `start()` is the only
    part that touches Telethon and is not unit-tested.
    """

    def __init__(self, session_factory, *, album_debounce: float = 2.0,
                 floodwait_sleeper: Callable[[float], Awaitable] | None = None):
        self.session_factory = session_factory
        self.buffer = AlbumBuffer(album_debounce)
        self._sleeper = floodwait_sleeper
        self._client = None  # telethon.TelegramClient, created in start()
        self._client_ready = asyncio.Event()  # set once the client is connected

    async def wait_client(self):  # pragma: no cover - shared with the metrics loop
        """Block until start() has connected, then hand back the live client so a
        second consumer (metrics) reuses this single session (architecture §11)."""
        await self._client_ready.wait()
        return self._client

    # ---- persistence ----
    def _persist(self, raw: RawItem) -> int:
        from newsroom.collectors.ingest import upsert_raw_item

        with self.session_factory() as s:
            item, _ = upsert_raw_item(s, raw)
            s.commit()
            return item.id

    # ---- realtime handlers (called by Telethon callbacks or directly by tests) ----
    def on_new_message(self, source_id: int, msg, *, channel_username: str | None = None) -> RawItem | None:
        """Persist a standalone post immediately; buffer album members for merge."""
        raw = message_to_raw_item(msg, source_id, channel_username=channel_username)
        if raw.grouped_id is not None:
            self.buffer.add(raw)
            return None
        self._persist(raw)
        return raw

    def flush_albums(self, now: dt.datetime | None = None) -> list[RawItem]:
        """Persist albums whose debounce window has elapsed (called on a ticker)."""
        flushed = self.buffer.flush_ready(now)
        for raw in flushed:
            self._persist(raw)
        return flushed

    def on_edit(self, source_id: int, msg, *, channel_username: str | None = None) -> None:
        # upsert versions the previous body when content_hash changes (§12).
        self._persist(message_to_raw_item(msg, source_id, channel_username=channel_username))

    def on_delete(self, source_id: int, message_ids: list[int]) -> int:
        marked = 0
        with self.session_factory() as s:
            for mid in message_ids:
                if mark_item_deleted(s, source_id, str(mid)):
                    marked += 1
            s.commit()
        return marked

    # ---- backfill (client-agnostic; fake client in tests) ----
    async def backfill_source(self, client, source_id: int, *, peer=None,
                              channel_username: str | None = None,
                              first_seed_limit: int = 30, max_backfill: int = 500) -> int:
        """Ingest recent messages for a channel. FloodWait is waited out, not fatal.

        A channel seen for the FIRST time (no stored cursor) is *seeded* with only the
        most recent `first_seed_limit` posts — never its whole history, or a busy
        aggregator would pour tens of thousands of years-old posts into the pipeline
        (§12: we track from now on, we are not an archive). A channel with a cursor
        catches up on posts above it, capped at `max_backfill` so a long downtime
        cannot flood in one tick (the cursor advances, the next tick continues)."""
        with self.session_factory() as s:
            cursor = backfill_cursor(s, source_id)

        if cursor == 0:
            async def fetch() -> list:
                # newest-first, then chronological, so albums/order stay consistent
                msgs = [m async for m in client.iter_messages(peer, limit=first_seed_limit, reverse=False)]
                return list(reversed(msgs))
        else:
            async def fetch() -> list:
                return [m async for m in client.iter_messages(
                    peer, min_id=cursor, reverse=True, limit=max_backfill)]

        messages = await run_with_floodwait(fetch, sleeper=self._sleeper)
        for msg in messages:
            self.on_new_message(source_id, msg, channel_username=channel_username)
        # albums buffered during backfill are complete now — force-flush them.
        for raw in self.buffer.flush_all():
            self._persist(raw)
        return len(messages)

    # ---- live wiring (Telethon; not unit-tested) ----
    def _connect(self):  # pragma: no cover
        from telethon import TelegramClient

        api_id = int(os.environ["TELEGRAM_API_ID"])
        api_hash = os.environ["TELEGRAM_API_HASH"]
        session = os.environ["TELEGRAM_SESSION_PATH"]
        return TelegramClient(session, api_id, api_hash)

    def _telegram_sources(self) -> list[tuple[int, str]]:  # pragma: no cover
        from sqlalchemy import select

        from newsroom.models import Source

        with self.session_factory() as s:
            rows = s.execute(
                select(Source.id, Source.handle_or_url).where(
                    Source.kind == "telegram", Source.active.is_(True)
                )
            ).all()
        return [(sid, handle) for sid, handle in rows]

    async def start(self, *, album_tick_seconds: float = 2.0) -> None:  # pragma: no cover
        """Connect, backfill each channel, then stream realtime until disconnected."""
        from telethon import events

        if not telegram_enabled():
            raise RuntimeError("COLLECTOR_TELEGRAM_ENABLED is off")

        client = self._connect()
        await client.start()
        self._client = client
        self._client_ready.set()   # unblock the metrics loop (shared session)

        sources = self._telegram_sources()
        by_username = {handle.lstrip("@").lower(): sid for sid, handle in sources}

        async def _sid_for(chat) -> int | None:
            uname = (getattr(chat, "username", None) or "").lower()
            return by_username.get(uname)

        @client.on(events.NewMessage(chats=[h for _, h in sources]))
        async def _new(event):
            sid = await _sid_for(await event.get_chat())
            if sid is not None:
                self.on_new_message(sid, event.message, channel_username=getattr(await event.get_chat(), "username", None))

        @client.on(events.MessageEdited(chats=[h for _, h in sources]))
        async def _edited(event):
            sid = await _sid_for(await event.get_chat())
            if sid is not None:
                self.on_edit(sid, event.message)

        @client.on(events.MessageDeleted(chats=[h for _, h in sources]))
        async def _deleted(event):
            sid = await _sid_for(await event.get_chat())
            if sid is not None:
                self.on_delete(sid, list(event.deleted_ids))

        seed = int(os.getenv("TELEGRAM_SEED_LIMIT", "30"))
        max_backfill = int(os.getenv("TELEGRAM_MAX_BACKFILL", "500"))
        for sid, handle in sources:
            entity = await client.get_entity(handle)
            await self.backfill_source(client, sid, peer=entity,
                                       channel_username=handle.lstrip("@"),
                                       first_seed_limit=seed, max_backfill=max_backfill)

        async def _album_ticker():
            while True:
                await asyncio.sleep(album_tick_seconds)
                self.flush_albums()

        asyncio.ensure_future(_album_ticker())
        log.info("telegram collector started", extra={"channels": len(sources)})
        await client.run_until_disconnected()
