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
import math
from typing import Protocol

from newsroom.publishers.metrics import MessageStats

log = logging.getLogger("newsroom.analyze.demand")

# forwards are active sharing ("worth passing on"), a better value signal than an
# emotional reaction — weight them higher in the engagement rate.
DEFAULT_FORWARD_WEIGHT = 2.0
DEFAULT_COMMENT_WEIGHT = 1.5

# Manipulation discount: aggregator/anonymous channels are the most botted/IPSO-prone,
# so their engagement counts for less when we learn what the audience genuinely wants.
TIER_WEIGHT = {"official": 1.0, "media": 1.0, "aggregator": 0.5, "leak": 0.3, "anonymous": 0.3}

DEMAND_STATE_KEY = "demand_by_rubric"


def engagement_rate(stats: MessageStats, subscribers: int | None,
                    *, forward_weight: float = DEFAULT_FORWARD_WEIGHT,
                    comment_weight: float = DEFAULT_COMMENT_WEIGHT,
                    post_age_hours: float | None = None) -> float | None:
    """Age/reach-normalised audience response for one Telegram post.

    Views measure reach; reactions/forwards/comments measure active response per viewer.
    When a snapshot age is known, reach is divided by a bounded Telegram growth curve so a
    five-minute post is not compared directly with a three-hour post.  None means there is
    no subscriber denominator and the raw counts would be misleading.
    """
    if not subscribers or subscribers <= 0:
        return None
    reacts = sum((stats.reactions or {}).values())
    forwards = stats.forwards or 0
    comments = stats.comments or 0
    views = max(int(stats.views or 0), 0)
    reach_rate = views / subscribers
    if post_age_hours is not None:
        age = max(float(post_age_hours), 0.0)
        # Most Telegram reach accumulates in the first hours.  The 10% floor avoids
        # exploding a just-published post while still making age a first-class signal.
        maturity = max(1.0 - math.exp(-age / 6.0), 0.10)
        reach_rate /= maturity
    actions = reacts + forward_weight * forwards + comment_weight * comments
    action_rate = actions / (views if views > 0 else subscribers)
    # Reach and active response are deliberately separate; forwards/comments cannot be
    # hidden by a giant subscriber denominator once actual views are known.
    return 0.65 * reach_rate + 0.35 * action_rate


class DemandSource(Protocol):
    def message_stats(self, handle: str, message_id: str) -> MessageStats | None: ...

    def subscribers(self, handle: str) -> int | None: ...


def record_item_metric(session, item_id: int, stats: MessageStats) -> int:
    """Append one engagement snapshot for a source item. Flushes; caller commits."""
    from newsroom.models import ItemMetric

    row = ItemMetric(item_id=item_id, views=stats.views,
                     reactions=stats.reactions, forwards=stats.forwards,
                     comments=stats.comments)
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


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def compute_rubric_demand(session, *, window_days: int = 7,
                          forward_weight: float = DEFAULT_FORWARD_WEIGHT) -> dict[str, float]:
    """Learn per-rubric demand from competitor engagement (docs/demand-intelligence.md).

    For each measured source post in the window: reach-normalised engagement
    (forwards weighted), discounted by source tier (bots/IPSO on aggregators count
    less). Group by the rubric of the post's event, take the MEDIAN (robust to
    spikes), then min-max normalise across rubrics to a comparative 0..1 index —
    comparison in a batch, not an absolute score (CLAUDE.md). {} until data exists."""
    import datetime as dt

    from sqlalchemy import func, select

    from newsroom.models import Event, EventItem, Item, ItemMetric, Source, SourceMetric

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)

    # latest subscriber count per source
    subs: dict[int, int] = {}
    sub_rows = session.execute(
        select(SourceMetric.source_id, SourceMetric.subscribers, SourceMetric.measured_at)
        .order_by(SourceMetric.source_id, SourceMetric.measured_at.desc())
    ).all()
    for source_id, subscribers, _at in sub_rows:
        if source_id not in subs and subscribers:
            subs[source_id] = int(subscribers)

    # latest engagement snapshot per measured item in the window, with tier + rubric
    rows = session.execute(
        select(ItemMetric.item_id, ItemMetric.views, ItemMetric.reactions, ItemMetric.forwards,
               ItemMetric.comments, ItemMetric.measured_at, Item.published_at,
               Source.id, Source.tier, Event.rubric)
        .join(Item, Item.id == ItemMetric.item_id)
        .join(Source, Source.id == Item.source_id)
        .join(EventItem, EventItem.item_id == Item.id)
        .join(Event, Event.id == EventItem.event_id)
        .where(ItemMetric.measured_at >= cutoff, Event.rubric.is_not(None))
        .order_by(ItemMetric.item_id, ItemMetric.measured_at.desc())
    ).all()

    seen: set[int] = set()
    by_rubric: dict[str, list[float]] = {}
    for item_id, views, reactions, forwards, comments, measured_at, published_at, source_id, tier, rubric in rows:
        if item_id in seen:
            continue                        # keep only the latest snapshot per item
        seen.add(item_id)
        age_hours = None
        if measured_at is not None and published_at is not None:
            age_hours = max((measured_at - published_at).total_seconds() / 3600.0, 0.0)
        rate = engagement_rate(
            MessageStats(views=views, reactions=reactions, forwards=forwards, comments=comments),
            subs.get(source_id), forward_weight=forward_weight, post_age_hours=age_hours)
        if rate is None:
            continue
        weighted = rate * TIER_WEIGHT.get((tier or "").lower(), 0.5)
        by_rubric.setdefault(rubric, []).append(weighted)

    medians = {rubric: _median(vals) for rubric, vals in by_rubric.items() if vals}
    if not medians:
        return {}
    lo, hi = min(medians.values()), max(medians.values())
    if hi <= lo:
        return {r: 0.5 for r in medians}    # single level — neutral
    return {r: (m - lo) / (hi - lo) for r, m in medians.items()}


def store_demand(session, by_rubric: dict[str, float]) -> None:
    """Persist the learned demand index for curation to read. Flushes; caller commits."""
    import datetime as dt

    from newsroom.models import SystemState

    value = {"by_rubric": by_rubric, "at": dt.datetime.now(dt.timezone.utc).isoformat()}
    row = session.get(SystemState, DEMAND_STATE_KEY)
    if row is None:
        session.add(SystemState(key=DEMAND_STATE_KEY, value=value))
    else:
        row.value = value
    session.flush()


def load_demand(session) -> dict[str, float]:
    """Read the learned per-rubric demand index ({} if none yet)."""
    from newsroom.models import SystemState

    row = session.get(SystemState, DEMAND_STATE_KEY)
    if row and isinstance(row.value, dict):
        by_rubric = row.value.get("by_rubric")
        if isinstance(by_rubric, dict):
            return {str(k): float(v) for k, v in by_rubric.items()}
    return {}


def refresh_demand(session_factory, *, window_days: int = 7) -> dict[str, float]:
    """One analytics tick: recompute per-rubric demand and store it. Returns the index."""
    with session_factory() as s:
        by_rubric = compute_rubric_demand(s, window_days=window_days)
        # Persist even an empty snapshot: absence and a computed empty result are different
        # operational states, and the timestamp proves the analytics loop is alive.
        store_demand(s, by_rubric)
        s.commit()
    return by_rubric


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
        replies = getattr(getattr(msg, "replies", None), "replies", None)
        return MessageStats(views=getattr(msg, "views", None),
                            reactions=reactions or None,
                            forwards=getattr(msg, "forwards", None),
                            comments=replies)

    def subscribers(self, handle: str) -> int | None:
        from telethon.tl.functions.channels import GetFullChannelRequest

        try:
            full = self._run(self._client(GetFullChannelRequest(self._entity(handle))))
            return int(full.full_chat.participants_count)
        except Exception:
            return None
