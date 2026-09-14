"""Auto-discover Telegram subscriptions and register them as sources.

The reading account is subscribed to competitor channels; instead of hand-editing
config/sources.yaml for each one, this reads the account's channel dialogs and adds any
not-yet-tracked public channel as a `telegram` source. Discovered sources persist (the
YAML sync only upserts YAML entries, it never deactivates others), so once added they stay.

Note: the realtime collector registers its per-channel handlers once at start(), so a
newly-added channel is only *collected* after the next restart — this just registers it.

`sync_subscriptions` is pure/DB (a fake channel list in tests); listing the live dialogs is
a thin Telethon call kept behind `# pragma`.
"""
from __future__ import annotations

import logging

log = logging.getLogger("newsroom.sources.discovery")

# defaults for a freshly discovered competitor channel (admin can retune via /sql)
_DEFAULT_ORIGIN = "ua"
_DEFAULT_LANG = "uk"
_DEFAULT_TIER = "media"
_DEFAULT_POLL_INTERVAL = 300


def _norm_username(username: str | None) -> str:
    return (username or "").lstrip("@").strip().lower()


def sync_subscriptions(session_factory, channels) -> dict:
    """Register any not-yet-tracked public channels as telegram sources.

    `channels` is an iterable of dicts with at least {"username", "title"} (and optionally
    "id"). Channels without a username are skipped (the collector matches realtime events by
    username and cannot resolve a private channel). Dedup is by handle_or_url == "@username"
    (case-insensitive) across existing telegram sources. Returns {added, skipped, existing}.
    """
    from sqlalchemy import select

    from newsroom.models import Source

    with session_factory() as s:
        existing = {
            _norm_username(h) for h in s.execute(
                select(Source.handle_or_url).where(Source.kind == "telegram")
            ).scalars().all()
        }

    added = 0
    skipped = 0
    seen: set[str] = set()
    with session_factory() as s:
        for ch in channels:
            uname = _norm_username(ch.get("username"))
            if not uname:
                skipped += 1
                continue
            if uname in existing or uname in seen:
                continue
            seen.add(uname)
            title = (ch.get("title") or uname).strip()
            s.add(Source(
                kind="telegram", handle_or_url=f"@{uname}", name=title[:200],
                origin=_DEFAULT_ORIGIN, lang=_DEFAULT_LANG, region=None,
                tier=_DEFAULT_TIER, is_official=False,
                active=True, poll_interval=_DEFAULT_POLL_INTERVAL,
            ))
            added += 1
        s.commit()

    return {"added": added, "skipped": skipped, "existing": len(existing)}


async def list_subscribed_channels(client) -> list[dict]:  # pragma: no cover - network
    """The account's public broadcast-channel dialogs as [{id, username, title}]."""
    out: list[dict] = []
    async for dialog in client.iter_dialogs():
        entity = dialog.entity
        # broadcast channels only (not private chats, users, or megagroup discussions)
        if getattr(entity, "broadcast", False) and getattr(entity, "username", None):
            out.append({"id": entity.id, "username": entity.username, "title": dialog.name or entity.username})
    return out
