"""Data-driven hot topics (docs/demand-intelligence.md follow-up).

The fixed rubric is a coarse risk/floor label. The *interesting* signal — what the
ecosystem is actually talking about right now — is learned from the corpus instead of
hardcoded: the classifier tags each event with 5–10 keywords, and here we aggregate
those keywords across the recent window of source (competitor) posts, weighted by how
much they were posted (volume across channels) and how much engagement they drew
(reach-normalised, tier-discounted — same as demand). The result is an always-current
ranked list of hot topics, stored for the editor (curation) and the admin console.

Keeps the charter floor intact: rubric still decides risk/sourcing; this only feeds
selection/interest. Aggregation is pure/DB and pg-tested.
"""
from __future__ import annotations

import datetime as dt
import logging

from newsroom.analyze.demand import DEFAULT_FORWARD_WEIGHT, TIER_WEIGHT, engagement_rate
from newsroom.publishers.metrics import MessageStats

log = logging.getLogger("newsroom.analyze.topics")

HOT_TOPICS_STATE_KEY = "hot_topics"


def normalize_keyword(kw: str) -> str:
    """Fold a keyword for aggregation: lowercase + collapse whitespace. The LLM is
    asked for base forms, so exact-match grouping is usually enough."""
    return " ".join((kw or "").lower().split())


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi <= lo:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def compute_hot_topics(session, *, window_hours: int = 48, min_events: int = 2,
                       forward_weight: float = DEFAULT_FORWARD_WEIGHT, top: int = 40) -> dict:
    """Rank keywords across the recent window by volume (posts across channels) and
    engagement. Returns {"topics": [detail…], "heat": {kw: 0..1}, "at": iso}. {} until
    events carry keywords."""
    from collections import defaultdict

    from sqlalchemy import func, select

    from newsroom.models import Event, EventItem, Item, ItemMetric, Source, SourceMetric

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)

    ev_rows = session.execute(
        select(Event.id, Event.keywords).where(
            Event.first_seen_at >= cutoff, Event.keywords.is_not(None))
    ).all()
    event_keywords: dict[int, list[str]] = {}
    for eid, kw in ev_rows:
        norm = sorted({normalize_keyword(k) for k in (kw or []) if str(k).strip()})
        if norm:
            event_keywords[eid] = norm
    if not event_keywords:
        return {}
    event_ids = list(event_keywords)

    # posts per event = how many channels carried it (volume across the ecosystem)
    volume = dict(session.execute(
        select(EventItem.event_id, func.count()).where(EventItem.event_id.in_(event_ids))
        .group_by(EventItem.event_id)
    ).all())

    # engagement per event = tier-weighted, reach-normalised (latest metric per item)
    subs: dict[int, int] = {}
    for sid, n, _at in session.execute(
        select(SourceMetric.source_id, SourceMetric.subscribers, SourceMetric.measured_at)
        .order_by(SourceMetric.source_id, SourceMetric.measured_at.desc())
    ).all():
        if sid not in subs and n:
            subs[sid] = int(n)
    eng_by_event: dict[int, float] = defaultdict(float)
    seen_items: set[int] = set()
    for item_id, views, reactions, forwards, _at, event_id, source_id, tier in session.execute(
        select(ItemMetric.item_id, ItemMetric.views, ItemMetric.reactions, ItemMetric.forwards,
               ItemMetric.measured_at, EventItem.event_id, Source.id, Source.tier)
        .join(Item, Item.id == ItemMetric.item_id)
        .join(EventItem, EventItem.item_id == Item.id)
        .join(Source, Source.id == Item.source_id)
        .where(ItemMetric.measured_at >= cutoff, EventItem.event_id.in_(event_ids))
        .order_by(ItemMetric.item_id, ItemMetric.measured_at.desc())
    ).all():
        if item_id in seen_items:
            continue
        seen_items.add(item_id)
        rate = engagement_rate(MessageStats(views=views, reactions=reactions, forwards=forwards),
                               subs.get(source_id), forward_weight=forward_weight)
        if rate is not None:
            eng_by_event[event_id] += rate * TIER_WEIGHT.get((tier or "").lower(), 0.5)

    # accumulate per keyword (unique per event, so one story isn't double-counted within itself)
    agg: dict[str, list[float]] = {}   # kw -> [events, posts, engagement]
    for eid, kws in event_keywords.items():
        vol = int(volume.get(eid, 1) or 1)
        eng = float(eng_by_event.get(eid, 0.0))
        for kw in kws:
            a = agg.setdefault(kw, [0, 0, 0.0])
            a[0] += 1
            a[1] += vol
            a[2] += eng
    agg = {k: v for k, v in agg.items() if v[0] >= min_events}
    if not agg:
        return {}

    posts_norm = _minmax({k: v[1] for k, v in agg.items()})
    eng_norm = _minmax({k: v[2] for k, v in agg.items()})
    heat = {k: round(0.6 * posts_norm[k] + 0.4 * eng_norm[k], 4) for k in agg}
    order = sorted(agg, key=lambda k: heat[k], reverse=True)[:top]
    topics = [{"topic": k, "events": agg[k][0], "posts": agg[k][1],
               "engagement": round(agg[k][2], 5), "heat": heat[k]} for k in order]
    return {"topics": topics, "heat": {k: heat[k] for k in order},
            "at": dt.datetime.now(dt.timezone.utc).isoformat()}


def store_hot_topics(session, result: dict) -> None:
    """Persist the hot-topics result for curation + the admin console. Flushes."""
    from newsroom.models import SystemState

    row = session.get(SystemState, HOT_TOPICS_STATE_KEY)
    if row is None:
        session.add(SystemState(key=HOT_TOPICS_STATE_KEY, value=result))
    else:
        row.value = result
    session.flush()


def load_hot_topics(session) -> dict[str, float]:
    """The keyword→heat map (0..1) for curation ({} if none yet)."""
    from newsroom.models import SystemState

    row = session.get(SystemState, HOT_TOPICS_STATE_KEY)
    if row and isinstance(row.value, dict):
        heat = row.value.get("heat")
        if isinstance(heat, dict):
            return {str(k): float(v) for k, v in heat.items()}
    return {}


def load_hot_topics_detail(session) -> list[dict]:
    """The ranked topic detail (for /topics)."""
    from newsroom.models import SystemState

    row = session.get(SystemState, HOT_TOPICS_STATE_KEY)
    if row and isinstance(row.value, dict) and isinstance(row.value.get("topics"), list):
        return list(row.value["topics"])
    return []


def topic_heat(keywords, heat_map: dict[str, float]) -> float:
    """Heat of an event = the hottest of its keywords (0 when none are trending)."""
    if not keywords or not heat_map:
        return 0.0
    return max((heat_map.get(normalize_keyword(k), 0.0) for k in keywords), default=0.0)


def refresh_hot_topics(session_factory, *, window_hours: int = 48) -> dict:
    """One analytics tick: recompute hot topics and store them. Returns the result."""
    with session_factory() as s:
        result = compute_hot_topics(s, window_hours=window_hours)
        if result:
            store_hot_topics(s, result)
            s.commit()
    return result
