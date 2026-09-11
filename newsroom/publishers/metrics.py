"""Publication and channel metrics (architecture §5.3, §10).

After a post goes out we track how it does — views, reactions, forwards — and the
channel's subscriber count over time, to feed ranking and reputation later. These
numbers come from the Telethon reading account (the Bot API cannot read channel
message views/reactions), so the stats source is pluggable: a real Telethon source
in production (`# pragma: no cover`), a fake in tests.

Metrics are time series: each tick appends a fresh snapshot (measured_at differs),
so there is no dedup. Only recent posts are polled — old ones stop changing. The
recording and selection are pure/DB and pg-tested.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger("newsroom.publishers.metrics")


@dataclass(frozen=True)
class MessageStats:
    views: int | None = None
    reactions: dict | None = None      # {emoji: count}
    forwards: int | None = None

    def is_empty(self) -> bool:
        return self.views is None and self.forwards is None and not self.reactions


class MetricsSource(Protocol):
    def message_stats(self, channel_ref: str) -> MessageStats | None: ...

    def channel_subscribers(self, channel: str) -> int | None: ...


def record_publication_metric(session, publication_id: int, stats: MessageStats) -> int:
    """Append one metrics snapshot for a publication. Flushes; caller commits."""
    from newsroom.models import PublicationMetric

    row = PublicationMetric(
        publication_id=publication_id,
        views=stats.views,
        reactions=stats.reactions,
        forwards=stats.forwards,
    )
    session.add(row)
    session.flush()
    return row.id


def record_channel_metric(session, channel: str, subscribers: int | None) -> int:
    """Append one subscriber-count snapshot for a channel. Flushes; caller commits."""
    from newsroom.models import ChannelMetric

    row = ChannelMetric(channel=channel, subscribers=subscribers)
    session.add(row)
    session.flush()
    return row.id


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class MetricsCollector:
    def __init__(self, session_factory, source: MetricsSource, *, channel: str = "telegram",
                 max_age_hours: int = 48):
        self.sf = session_factory
        self.source = source
        self.channel = channel
        self.max_age_hours = max_age_hours

    def collect_publications(self, *, limit: int = 50) -> dict[str, int]:
        """Snapshot metrics for recently published posts on this channel."""
        from sqlalchemy import select

        from newsroom.models import Publication

        cutoff = _utc_now() - dt.timedelta(hours=self.max_age_hours)
        with self.sf() as s:
            rows = list(s.execute(
                select(Publication.id, Publication.channel_ref)
                .where(
                    Publication.status == "published",
                    Publication.channel == self.channel,
                    Publication.channel_ref.is_not(None),
                    Publication.published_at >= cutoff,
                )
                .order_by(Publication.id)
                .limit(limit)
            ).all())

        stats = {"polled": 0, "recorded": 0}
        for pub_id, channel_ref in rows:
            stats["polled"] += 1
            try:
                message_stats = self.source.message_stats(channel_ref)
            except Exception:  # noqa: BLE001 - one bad read must not sink the batch
                log.warning("message stats read failed", extra={"channel_ref": channel_ref})
                continue
            if message_stats is None or message_stats.is_empty():
                continue
            with self.sf() as s:
                record_publication_metric(s, pub_id, message_stats)
                s.commit()
            stats["recorded"] += 1
        return stats

    def collect_channel(self) -> bool:
        """Snapshot the channel's subscriber count."""
        try:
            subscribers = self.source.channel_subscribers(self.channel)
        except Exception:  # noqa: BLE001
            log.warning("channel subscribers read failed")
            return False
        if subscribers is None:
            return False
        with self.sf() as s:
            record_channel_metric(s, self.channel, subscribers)
            s.commit()
        return True


class TelethonMetricsSource:  # pragma: no cover - network / MTProto
    """Reads channel message stats and subscriber counts via the Telethon account
    (the same reading account as the collector; not the publishing bot).

    The collector is being driven from `loop`, so this source — called from the
    metrics worker thread — schedules Telethon's async calls back onto that loop
    with run_coroutine_threadsafe rather than opening a second session (§11).
    """

    def __init__(self, client, channel_entity, loop, *, timeout: float = 30.0):
        self._client = client
        self._entity = channel_entity
        self._loop = loop
        self._timeout = timeout

    def _run(self, coro):
        import asyncio

        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=self._timeout)

    def message_stats(self, channel_ref: str) -> MessageStats | None:
        try:
            msg = self._run(self._client.get_messages(self._entity, ids=int(channel_ref)))
        except Exception:
            return None
        if msg is None:
            return None
        reactions = None
        if getattr(msg, "reactions", None) and getattr(msg.reactions, "results", None):
            reactions = {}
            for r in msg.reactions.results:
                emoticon = getattr(getattr(r, "reaction", None), "emoticon", None) or "?"
                reactions[emoticon] = getattr(r, "count", 0)
        return MessageStats(views=getattr(msg, "views", None),
                            reactions=reactions or None,
                            forwards=getattr(msg, "forwards", None))

    def channel_subscribers(self, channel: str) -> int | None:
        from telethon.tl.functions.channels import GetFullChannelRequest

        try:
            full = self._run(self._client(GetFullChannelRequest(self._entity)))
            return int(full.full_chat.participants_count)
        except Exception:
            return None
