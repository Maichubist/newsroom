"""Faceted topic model and engagement analytics.

The editorial rubric remains a stable top-level domain.  Everything below it is
represented as typed facets instead of forcing geography, actors, impact and event type
into one artificial path.  Values may be learned from data, while dimensions are fixed.

This module is intentionally deterministic/DB-only.  The classifier proposes assignments;
the functions here normalise, persist and score them.  Existing ``topic_path`` data remains
untouched so rollout and replay can compare both models.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from newsroom.analyze.taxonomy import normalize_label


# Fixed axes.  ``controlled`` means the vocabulary should eventually be backed by an
# editorial dictionary; unknown values are still stored as provisional so information is
# not discarded during the shadow rollout.
FACET_DIMENSIONS: dict[str, tuple[str, bool]] = {
    "domain": ("Редакційний домен", True),
    "event_type": ("Тип події", True),
    "geography": ("Географія", True),
    "actor": ("Дійова особа", False),
    "target": ("Ціль або об'єкт", False),
    "entity": ("Сутність", False),
    "sector": ("Сектор", True),
    "impact": ("Наслідок", True),
    "means": ("Засіб або тактика", True),
    "audience_scope": ("Масштаб аудиторії", True),
    "story": ("Сюжет або кампанія", False),
}

# Only meaningful cross-axis pairs become analytical edges.  Unrestricted N² pair creation
# is both noisy and expensive.
PAIR_DIMENSIONS = frozenset({
    frozenset(("event_type", "geography")),
    frozenset(("event_type", "target")),
    frozenset(("event_type", "impact")),
    frozenset(("actor", "event_type")),
    frozenset(("sector", "impact")),
    frozenset(("entity", "event_type")),
})

DECISION_DIMENSIONS = frozenset({
    "event_type", "geography", "target", "sector", "impact", "means", "audience_scope",
})


@dataclass(frozen=True)
class FacetAssignment:
    dimension: str
    path: tuple[str, ...] = ()
    confidence: float = 1.0
    evidence: str = ""

    @property
    def value(self) -> str:
        return self.path[-1] if self.path else ""


def normalise_assignment(value: FacetAssignment) -> FacetAssignment | None:
    dimension = str(value.dimension or "").strip().lower()
    if dimension not in FACET_DIMENSIONS:
        return None
    path: list[str] = []
    seen: set[str] = set()
    for raw in value.path[:4]:
        label = str(raw or "").strip()
        slug = normalize_label(label)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        path.append(label[:200])
    if not path:
        return None
    try:
        confidence = min(1.0, max(0.0, float(value.confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    return FacetAssignment(
        dimension=dimension,
        path=tuple(path),
        confidence=confidence,
        evidence=str(value.evidence or "").strip()[:500],
    )


def seed_dimensions(session) -> dict[str, int]:
    """Create fixed dimensions idempotently and return key -> id. Caller commits."""
    from sqlalchemy import select

    from newsroom.models import FacetDimension

    existing = {d.key: d for d in session.execute(select(FacetDimension)).scalars()}
    for key, (label, controlled) in FACET_DIMENSIONS.items():
        row = existing.get(key)
        if row is None:
            row = FacetDimension(key=key, label=label, controlled=controlled)
            session.add(row)
            session.flush()
            existing[key] = row
        else:
            row.label = label
            row.controlled = controlled
    return {key: row.id for key, row in existing.items() if key in FACET_DIMENSIONS}


def _get_or_create_value(session, *, dimension_id: int, path: tuple[str, ...], status: str) -> int:
    from sqlalchemy import select

    from newsroom.models import FacetValue

    parent_id: int | None = None
    leaf_id: int | None = None
    for label in path:
        slug = normalize_label(label)
        row = session.execute(select(FacetValue).where(
            FacetValue.dimension_id == dimension_id, FacetValue.slug == slug,
        )).scalar_one_or_none()
        if row is None:
            row = FacetValue(
                dimension_id=dimension_id, parent_id=parent_id, slug=slug,
                label=label[:200], status=status,
            )
            session.add(row)
            session.flush()
        elif row.parent_id is None and parent_id is not None:
            # A previously learned leaf may later arrive with a useful hierarchy.
            row.parent_id = parent_id
        parent_id = row.id
        leaf_id = row.id
    assert leaf_id is not None
    return leaf_id


def ingest_event_facets(session, event_id: int, assignments,
                        *, source_item_id: int | None = None,
                        primary_rubric: str | None = None,
                        secondary_rubrics=()) -> int:
    """Persist evidence-backed leaf assignments. Idempotent per event/value.

    Domain assignments are derived from the stable rubric spine rather than trusted to the
    model.  Learned values start provisional; domain values are active immediately.
    """
    from sqlalchemy import select

    from newsroom.models import EventFacet, FacetValue

    clean: list[FacetAssignment] = []
    for raw in assignments or ():
        if isinstance(raw, FacetAssignment):
            item = normalise_assignment(raw)
            if item is not None and item.dimension != "domain":
                clean.append(item)
    rubrics = [str(x).strip().lower() for x in [primary_rubric, *(secondary_rubrics or ())]
               if str(x or "").strip()]
    for rubric in dict.fromkeys(rubrics):
        clean.append(FacetAssignment("domain", (rubric,), 1.0, "stable editorial rubric"))

    if not clean:
        return 0
    dims = seed_dimensions(session)
    existing = set(session.execute(select(EventFacet.facet_value_id).where(
        EventFacet.event_id == event_id)).scalars())
    added = 0
    seen: set[tuple[str, str]] = set()
    for item in clean:
        leaf_slug = normalize_label(item.value)
        key = (item.dimension, leaf_slug)
        if key in seen:
            continue
        seen.add(key)
        leaf_id = _get_or_create_value(
            session, dimension_id=dims[item.dimension], path=item.path,
            status="active" if item.dimension == "domain" else "provisional",
        )
        if leaf_id in existing:
            continue
        session.add(EventFacet(
            event_id=event_id, facet_value_id=leaf_id, confidence=item.confidence,
            evidence_text=item.evidence or None, source_item_id=source_item_id, role="leaf",
        ))
        node = session.get(FacetValue, leaf_id)
        guard = 0
        while node is not None and guard < 16:
            node.event_count = (node.event_count or 0) + 1
            node = session.get(FacetValue, node.parent_id) if node.parent_id is not None else None
            guard += 1
        existing.add(leaf_id)
        added += 1
    session.flush()
    return added


def load_event_assignments(session, event_id: int) -> list[FacetAssignment]:
    """Rebuild cached leaf assignments for a classified event without another LLM call."""
    from sqlalchemy import select

    from newsroom.models import EventFacet, FacetDimension, FacetValue

    dims = {row.id: row.key for row in session.execute(select(FacetDimension)).scalars()}
    values = {row.id: row for row in session.execute(select(FacetValue)).scalars()}
    out: list[FacetAssignment] = []
    rows = session.execute(select(EventFacet).where(EventFacet.event_id == event_id)).scalars()
    for link in rows:
        leaf = values.get(link.facet_value_id)
        if leaf is None:
            continue
        path: list[str] = []
        current = leaf
        guard = 0
        while current is not None and guard < 16:
            path.append(current.label)
            current = values.get(current.parent_id) if current.parent_id is not None else None
            guard += 1
        out.append(FacetAssignment(
            dimension=dims.get(leaf.dimension_id, ""), path=tuple(reversed(path)),
            confidence=link.confidence or 0.0, evidence=link.evidence_text or "",
        ))
    return [item for item in out if item.dimension]


def _median(values) -> float:
    rows = sorted(float(v) for v in values)
    if not rows:
        return 0.0
    mid = len(rows) // 2
    return rows[mid] if len(rows) % 2 else (rows[mid - 1] + rows[mid]) / 2.0


def _minmax_grouped(raw: dict[int, float], group_of: dict[int, object]) -> dict[int, float]:
    grouped: dict[object, dict[int, float]] = defaultdict(dict)
    for key, score in raw.items():
        grouped[group_of.get(key)][key] = score
    out: dict[int, float] = {}
    for scores in grouped.values():
        lo, hi = min(scores.values()), max(scores.values())
        if hi <= lo:
            out.update({key: 0.5 for key in scores})
        else:
            out.update({key: (score - lo) / (hi - lo) for key, score in scores.items()})
    return out


def _ancestors(value_id: int, parents: dict[int, int | None]) -> tuple[int, ...]:
    out: list[int] = []
    current: int | None = value_id
    guard = 0
    while current is not None and guard < 16:
        out.append(current)
        current = parents.get(current)
        guard += 1
    return tuple(out)


def _pair_allowed(a_dim: str, b_dim: str) -> bool:
    return a_dim != b_dim and frozenset((a_dim, b_dim)) in PAIR_DIMENSIONS


def compute_facet_metrics(session, *, window_hours: int = 24, baseline_days: int = 7,
                          halflife_hours: float = 3.0,
                          min_pair_events: int = 3,
                          min_pair_sources: int = 2) -> dict:
    """Compute facet and meaningful-pair heat/demand without mutating the DB.

    Heat is a burst signal: recency-weighted event activity, independent-source breadth and
    acceleration over the preceding baseline. Demand is a robust median of age-normalised
    post performance, first within each source and then across sources, so one prolific
    aggregator cannot dominate a topic.
    """
    from sqlalchemy import select

    from newsroom.analyze.demand import engagement_rate
    from newsroom.models import (
        Event, EventFacet, EventItem, FacetDimension, FacetValue, Item, ItemMetric,
        SourceMetric,
    )
    from newsroom.publishers.metrics import MessageStats

    now = dt.datetime.now(dt.timezone.utc)
    recent_cutoff = now - dt.timedelta(hours=window_hours)
    baseline_cutoff = now - dt.timedelta(days=baseline_days)

    values = {row.id: row for row in session.execute(select(FacetValue)).scalars()}
    if not values:
        return {"values": {}, "pairs": {}}
    dimensions = {row.id: row.key for row in session.execute(select(FacetDimension)).scalars()}
    parents = {vid: row.parent_id for vid, row in values.items()}
    dimension_of = {vid: dimensions.get(row.dimension_id, "") for vid, row in values.items()}

    event_meta = {eid: (seen, duplicate_of) for eid, seen, duplicate_of in session.execute(
        select(Event.id, Event.first_seen_at, Event.duplicate_of).where(
            Event.first_seen_at >= baseline_cutoff, Event.status != "filtered_out",
        )).all()}
    if not event_meta:
        return {"values": {}, "pairs": {}}

    leaf_by_event: dict[int, list[int]] = defaultdict(list)
    confidence_by_event_value: dict[tuple[int, int], float] = {}
    for eid, value_id, confidence in session.execute(select(
        EventFacet.event_id, EventFacet.facet_value_id, EventFacet.confidence,
    ).where(EventFacet.event_id.in_(event_meta))).all():
        if event_meta[eid][1] is not None or value_id not in values:
            continue
        leaf_by_event[eid].append(value_id)
        confidence_by_event_value[(eid, value_id)] = max(0.0, min(1.0, confidence or 0.0))

    sources_by_event: dict[int, set[int]] = defaultdict(set)
    for eid, source_id in session.execute(select(EventItem.event_id, Item.source_id)
                                           .join(Item, Item.id == EventItem.item_id)
                                           .where(EventItem.event_id.in_(event_meta))).all():
        sources_by_event[eid].add(source_id)

    recent_weight: dict[int, float] = defaultdict(float)
    recent_events: dict[int, set[int]] = defaultdict(set)
    baseline_events: dict[int, set[int]] = defaultdict(set)
    sources_by_value: dict[int, set[int]] = defaultdict(set)
    pair_recent_weight: dict[tuple[int, int], float] = defaultdict(float)
    pair_events: dict[tuple[int, int], set[int]] = defaultdict(set)
    pair_sources: dict[tuple[int, int], set[int]] = defaultdict(set)
    pairs_by_event: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for eid, leaves in leaf_by_event.items():
        seen_at = event_meta[eid][0]
        if seen_at is None:
            continue
        age_h = max((now - seen_at).total_seconds() / 3600.0, 0.0)
        is_recent = seen_at >= recent_cutoff
        recency = math.exp(-age_h / halflife_hours) if is_recent and halflife_hours > 0 else 0.0
        expanded: set[int] = set()
        for leaf in leaves:
            confidence = confidence_by_event_value.get((eid, leaf), 0.0)
            # Evidence-poor speculative tags stay recorded but do not steer ranking.
            if confidence < 0.5:
                continue
            expanded.update(_ancestors(leaf, parents))
        for value_id in expanded:
            baseline_events[value_id].add(eid)
            if is_recent:
                recent_weight[value_id] += recency
                recent_events[value_id].add(eid)
                sources_by_value[value_id].update(sources_by_event.get(eid, ()))

        decision_leaves = sorted({leaf for leaf in leaves
                                  if confidence_by_event_value.get((eid, leaf), 0.0) >= 0.5})
        for a, b in combinations(decision_leaves, 2):
            if not _pair_allowed(dimension_of.get(a, ""), dimension_of.get(b, "")):
                continue
            pair = (min(a, b), max(a, b))
            pairs_by_event[eid].append(pair)
            if is_recent:
                pair_recent_weight[pair] += recency
                pair_events[pair].add(eid)
                pair_sources[pair].update(sources_by_event.get(eid, ()))

    baseline_hours = max(baseline_days * 24 - window_hours, 1)
    value_heat_raw: dict[int, float] = {}
    for value_id, weighted in recent_weight.items():
        old_count = max(len(baseline_events[value_id]) - len(recent_events[value_id]), 0)
        expected = old_count * window_hours / baseline_hours
        burst = len(recent_events[value_id]) / (expected + 1.0)
        breadth = len(sources_by_value[value_id])
        value_heat_raw[value_id] = weighted * (1.0 + 0.25 * math.log1p(breadth)) * (
            1.0 + 0.5 * math.log1p(burst))

    # Latest snapshots and latest subscriber counts.
    subscribers: dict[int, int] = {}
    for sid, count, _at in session.execute(select(
        SourceMetric.source_id, SourceMetric.subscribers, SourceMetric.measured_at,
    ).order_by(SourceMetric.source_id, SourceMetric.measured_at.desc())).all():
        if sid not in subscribers and count:
            subscribers[sid] = int(count)

    latest_items: set[int] = set()
    demand_by_value_source: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    demand_by_pair_source: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    rows = session.execute(select(
        ItemMetric.item_id, ItemMetric.views, ItemMetric.reactions, ItemMetric.forwards,
        ItemMetric.comments, ItemMetric.measured_at, Item.published_at, Item.source_id,
        EventItem.event_id,
    ).join(Item, Item.id == ItemMetric.item_id)
     .join(EventItem, EventItem.item_id == Item.id)
     .where(ItemMetric.measured_at >= baseline_cutoff, EventItem.event_id.in_(leaf_by_event))
     .order_by(ItemMetric.item_id, ItemMetric.measured_at.desc())).all()
    for item_id, views, reactions, forwards, comments, measured_at, published_at, source_id, eid in rows:
        if item_id in latest_items:
            continue
        latest_items.add(item_id)
        age_hours = None
        if measured_at is not None and published_at is not None:
            age_hours = max((measured_at - published_at).total_seconds() / 3600.0, 0.0)
        score = engagement_rate(
            MessageStats(views=views, reactions=reactions, forwards=forwards, comments=comments),
            subscribers.get(source_id), post_age_hours=age_hours,
        )
        if score is None:
            continue
        for leaf in leaf_by_event.get(eid, ()):
            if confidence_by_event_value.get((eid, leaf), 0.0) >= 0.5:
                for value_id in _ancestors(leaf, parents):
                    demand_by_value_source[value_id][source_id].append(score)
        for pair in pairs_by_event.get(eid, ()):
            demand_by_pair_source[pair][source_id].append(score)

    value_demand_raw = {
        value_id: _median(_median(scores) for scores in by_source.values())
        for value_id, by_source in demand_by_value_source.items() if by_source
    }
    value_heat = _minmax_grouped(value_heat_raw, dimension_of)
    value_demand = _minmax_grouped(value_demand_raw, dimension_of)

    eligible_pairs = {
        pair for pair in pair_events
        if len(pair_events[pair]) >= min_pair_events and len(pair_sources[pair]) >= min_pair_sources
    }
    pair_group = {pair: tuple(sorted((dimension_of.get(pair[0], ""),
                                     dimension_of.get(pair[1], "")))) for pair in eligible_pairs}
    pair_heat_raw = {
        pair: pair_recent_weight[pair] * (1.0 + 0.25 * math.log1p(len(pair_sources[pair])))
        for pair in eligible_pairs
    }
    pair_demand_raw = {
        pair: _median(_median(scores) for scores in demand_by_pair_source.get(pair, {}).values())
        for pair in eligible_pairs if demand_by_pair_source.get(pair)
    }
    pair_heat = _minmax_grouped(pair_heat_raw, pair_group)
    pair_demand = _minmax_grouped(pair_demand_raw, pair_group)

    return {
        "values": {
            value_id: {
                "heat": round(value_heat.get(value_id, 0.0), 4),
                "demand": round(value_demand.get(value_id, 0.0), 4),
                "events": len(recent_events[value_id]),
                "sources": len(sources_by_value[value_id]),
            }
            for value_id in set(value_heat_raw) | set(value_demand_raw)
        },
        "pairs": {
            pair: {
                "heat": round(pair_heat.get(pair, 0.0), 4),
                "demand": round(pair_demand.get(pair, 0.0), 4),
                "events": len(pair_events[pair]),
                "sources": len(pair_sources[pair]),
            }
            for pair in eligible_pairs
        },
    }


def refresh_facet_metrics(session_factory, *, window_hours: int = 24,
                          baseline_days: int = 7) -> dict[str, int]:
    """Recompute and persist facet + pair signals. Safe to call repeatedly."""
    from sqlalchemy import delete, select

    from newsroom.models import FacetPairMetric, FacetValue

    with session_factory() as session:
        result = compute_facet_metrics(
            session, window_hours=window_hours, baseline_days=baseline_days)
        now = dt.datetime.now(dt.timezone.utc)
        active = set(result["values"])
        for value_id, metrics in result["values"].items():
            row = session.get(FacetValue, value_id)
            if row is not None:
                row.heat = metrics["heat"]
                row.demand = metrics["demand"]
                row.metric_at = now
                if row.status == "provisional" and row.event_count >= 3:
                    row.status = "active"
        for row in session.execute(select(FacetValue).where(FacetValue.metric_at.is_not(None))).scalars():
            if row.id not in active:
                row.heat = 0.0
                row.demand = 0.0
                row.metric_at = now

        # Pair rows are derived cache, so replace only this window's cache atomically.
        session.execute(delete(FacetPairMetric).where(FacetPairMetric.window_hours == window_hours))
        for (a, b), metrics in result["pairs"].items():
            session.add(FacetPairMetric(
                facet_a_id=a, facet_b_id=b, window_hours=window_hours,
                event_count=metrics["events"], independent_source_count=metrics["sources"],
                heat=metrics["heat"], demand=metrics["demand"], measured_at=now,
            ))
        session.commit()
    return {"values": len(result["values"]), "pairs": len(result["pairs"])}


def load_event_facet_signals(session, event_ids) -> dict[int, tuple[float, float]]:
    """Return robust event popularity from supported pairs, then useful singleton facets."""
    from sqlalchemy import select

    from newsroom.models import EventFacet, FacetDimension, FacetPairMetric, FacetValue

    ids = list(dict.fromkeys(event_ids))
    if not ids:
        return {}
    dims = {row.id: row.key for row in session.execute(select(FacetDimension)).scalars()}
    values = {row.id: (row.heat or 0.0, row.demand or 0.0,
                       dims.get(row.dimension_id, ""), row.metric_at)
              for row in session.execute(select(FacetValue)).scalars()}
    by_event: dict[int, list[int]] = defaultdict(list)
    for eid, value_id, confidence in session.execute(select(
        EventFacet.event_id, EventFacet.facet_value_id, EventFacet.confidence,
    ).where(EventFacet.event_id.in_(ids))).all():
        if (confidence or 0.0) >= 0.5 and value_id in values:
            by_event[eid].append(value_id)
    if not by_event:
        return {}

    pair_rows = {(row.facet_a_id, row.facet_b_id): row
                 for row in session.execute(select(FacetPairMetric).where(
                     FacetPairMetric.window_hours == 24)).scalars()}
    out: dict[int, tuple[float, float]] = {}
    for eid, leaves in by_event.items():
        pair_scores: list[tuple[float, float]] = []
        for a, b in combinations(sorted(set(leaves)), 2):
            row = pair_rows.get((min(a, b), max(a, b)))
            if row is not None:
                pair_scores.append((row.heat or 0.0, row.demand or 0.0))
        if pair_scores:
            # Median of up to the three strongest supported relations: robust to one hot tag.
            strongest = sorted(pair_scores, key=lambda x: x[0] + x[1], reverse=True)[:3]
            out[eid] = (round(_median(x[0] for x in strongest), 4),
                        round(_median(x[1] for x in strongest), 4))
            continue
        singleton = [(values[v][0], values[v][1]) for v in leaves
                     if values[v][2] in DECISION_DIMENSIONS and values[v][3] is not None]
        if singleton:
            strongest = sorted(singleton, key=lambda x: x[0] + x[1], reverse=True)[:3]
            out[eid] = (round(_median(x[0] for x in strongest), 4),
                        round(_median(x[1] for x in strongest), 4))
    return out


def load_top_facets(session, *, limit: int = 25) -> list[dict]:
    """Operational view of the strongest current facet values for admin/shadow review."""
    from sqlalchemy import select

    from newsroom.models import FacetDimension, FacetValue

    rows = session.execute(
        select(FacetDimension.key, FacetValue.label, FacetValue.status,
               FacetValue.event_count, FacetValue.heat, FacetValue.demand)
        .join(FacetValue, FacetValue.dimension_id == FacetDimension.id)
        .where(FacetValue.metric_at.is_not(None))
        .order_by((FacetValue.heat + FacetValue.demand).desc(), FacetValue.event_count.desc())
        .limit(limit)
    ).all()
    return [{
        "dimension": dim, "label": label, "status": status, "events": events,
        "heat": round(heat or 0.0, 3), "demand": round(demand or 0.0, 3),
    } for dim, label, status, events, heat, demand in rows]
