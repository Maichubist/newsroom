"""Demand-intelligence collection (docs/demand-intelligence.md).

Watch how the ecosystem reacts to source posts: snapshot views/reactions/forwards
of monitored Telegram channels' recent posts over time (item_metrics) and each
channel's subscriber count (source_metrics), so later we can normalise engagement
by reach and age and derive a per-theme demand signal.

This is the *collection* layer only — it accumulates data. Theme clustering and the
demand score come later. Reads come from the shared Telethon reading account (Bot
API can't see views/reactions), so the stats source is pluggable: Telethon in prod
(`# pragma`), a fake in tests. Recording and selection are pure/DB and pg-tested.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Protocol

from newsroom.publishers.metrics import MessageStats

log = logging.getLogger("newsroom.analyze.demand")

# forwards are active sharing ("worth passing on"), a better value signal than an
# emotional reaction — weight them higher in the engagement rate.
DEFAULT_FORWARD_WEIGHT = 2.0


def engagement_rate(stats: MessageStats, subscribers: int | None,
                    *, forward_weight: float = DEFAULT_FORWARD_WEIGHT) -> float | None:
    """Reach-normalised engagement: (reactions + weighted forwards) / subscribers.
    None when subscribers is unknown/zero — a raw count without reach is misleading
    (10k views means opposite things on a 50k vs a 2M channel)."""
    if not subscribers or subscribers <= 0:
        return None
    reacts = sum((stats.reactions or {}).values())
    forwards = stats.forwards or 0
    return (reacts + forward_weight * forwards) / subscribers


class DemandSource(Protocol):
    def message_stats(self, handle: str, message_id: str) -> MessageStats | None: ...

    def subscribers(self, handle: str) -> int | None: ...


def record_item_metric(session, item_id: int, stats: MessageStats) -> int:
    """Append one engagement snapshot for a source item. Flushes; caller commits."""
    from newsroom.models import ItemMetric

    row = ItemMetric(item_id=item_id, views=stats.views,
                     reactions=stats.reactions, forwards=stats.forwards)
    session.add(row)
    session.flush()
    return row.id


def record_source_metric(session, source_id: int, subscribers: int | None) -> int:
    """Append one subscriber-count snapshot for a source. Flushes; caller commits."""
    from newsroom.models import SourceMetric

    row = SourceMetric(source_id=source_id, subscribers=subscribers)
    session.add(row)
    session.flush()
    return row.id


def select_items_to_measure(session, *, max_age_hours: int = 48, limit: int = 100) -> list[tuple[int, str, str]]:
    """Recent Telegram source posts still worth polling (older posts stop changing).
    Returns (item_id, source_handle, external_id)."""
    from sqlalchemy import select

    from newsroom.models import Item, Source

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=max_age_hours)
    rows = session.execute(
        select(Item.id, Source.handle_or_url, Item.external_id)
        .join(Source, Source.id == Item.source_id)
        .where(
            Source.kind == "telegram",
            Item.external_id.is_not(None),
            Item.published_at >= cutoff,
        )
        .order_by(Item.id.desc())
        .limit(limit)
    ).all()
    return [(int(i), str(h), str(e)) for i, h, e in rows]


class DemandCollector:
    def __init__(self, session_factory, source: DemandSource, *, max_age_hours: int = 48):
        self.sf = session_factory
        self.source = source
        self.max_age_hours = max_age_hours

    def collect_items(self, *, limit: int = 100) -> dict[str, int]:
        """Snapshot engagement for recent source posts."""
        with self.sf() as s:
            rows = select_items_to_measure(s, max_age_hours=self.max_age_hours, limit=limit)

        stats = {"polled": 0, "recorded": 0}
        for item_id, handle, external_id in rows:
            stats["polled"] += 1
            try:
                ms = self.source.message_stats(handle, external_id)
            except Exception:  # noqa: BLE001 - one bad read must not sink the batch
                log.warning("item stats read failed", extra={"item_id": item_id})
                continue
            if ms is None or ms.is_empty():
                continue
            with self.sf() as s:
                record_item_metric(s, item_id, ms)
                s.commit()
            stats["recorded"] += 1
        return stats

    def collect_sources(self) -> dict[str, int]:
        """Snapshot subscriber counts for active Telegram sources (for normalisation)."""
        from sqlalchemy import select

        from newsroom.models import Source

        with self.sf() as s:
            rows = list(s.execute(
                select(Source.id, Source.handle_or_url)
                .where(Source.kind == "telegram", Source.active.is_(True))
            ).all())

        stats = {"sources": 0, "recorded": 0}
        for source_id, handle in rows:
            stats["sources"] += 1
            try:
                subs = self.source.subscribers(str(handle))
            except Exception:  # noqa: BLE001
                log.warning("source subscribers read failed", extra={"source_id": source_id})
                continue
            if subs is None:
                continue
            with self.sf() as s:
                record_source_metric(s, source_id, subs)
                s.commit()
            stats["recorded"] += 1
        return stats


class TelethonDemandSource:  # pragma: no cover - network / MTProto
    """Reads engagement + subscriber counts for arbitrary channels via the shared
    Telethon reading account, scheduling async calls onto the collector's loop (§11).
    Resolved entities are cached by handle."""

    def __init__(self, client, loop, *, timeout: float = 30.0):
        self._client = client
        self._loop = loop
        self._timeout = timeout
        self._entities: dict[str, object] = {}

    def _run(self, coro):
        import asyncio

        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=self._timeout)

    def _entity(self, handle: str):
        if handle not in self._entities:
            self._entities[handle] = self._run(self._client.get_entity(handle))
        return self._entities[handle]

    def message_stats(self, handle: str, message_id: str) -> MessageStats | None:
        try:
            msg = self._run(self._client.get_messages(self._entity(handle), ids=int(message_id)))
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

    def subscribers(self, handle: str) -> int | None:
        from telethon.tl.functions.channels import GetFullChannelRequest

        try:
            full = self._run(self._client(GetFullChannelRequest(self._entity(handle))))
            return int(full.full_chat.participants_count)
        except Exception:
            return None
