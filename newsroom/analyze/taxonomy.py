"""Learned taxonomy pyramid (charter v0.3 §3.1).

The tree is built from the topic paths the classifier emits per event — broad->specific
labels like ["війна", "атака рф", "удар бпла", "одеса"], not a hand-authored list.
`ingest_path` walks/creates the node chain for a path and returns the leaf node id, and
bumps `event_count` on every node on the path, so a node's count is the number of events
whose topic passes through it — the roll-up the engagement layer (Phase 2) builds on.

Normalization is deterministic (lowercase, trim, drop punctuation, collapse spaces) so the
same topic always maps to the same node chain. Parsing/normalization is pure and offline-
tested; ingest is pg-tested.
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import re

log = logging.getLogger("newsroom.analyze.taxonomy")

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s-]", re.UNICODE)   # keep letters/digits/underscore/space/hyphen


def normalize_label(label: str | None) -> str:
    """Deterministic match key for a topic label: lowercase, no punctuation, single spaces."""
    s = (label or "").strip().lower()
    s = _PUNCT.sub("", s)
    return _WS.sub(" ", s).strip()


def normalize_path(path, *, max_depth: int = 5) -> list[tuple[str, str]]:
    """(slug, display_label) pairs for a broad->specific path, dropping empties, capped."""
    out: list[tuple[str, str]] = []
    for raw in (path or [])[:max_depth]:
        slug = normalize_label(raw)
        if slug:
            out.append((slug, str(raw).strip()))
    return out


def ingest_path(session, path, *, max_depth: int = 5) -> int | None:
    """Upsert the node chain for a topic path and return the leaf node id (None if empty).
    Bumps event_count on every node on the path. Caller commits."""
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError

    from newsroom.models import TaxonomyNode

    labels = normalize_path(path, max_depth=max_depth)
    if not labels:
        return None

    parent_id: int | None = None
    leaf_id: int | None = None
    for depth, (slug, label) in enumerate(labels):
        cond = TaxonomyNode.parent_id.is_(None) if parent_id is None else TaxonomyNode.parent_id == parent_id
        node = session.execute(
            select(TaxonomyNode).where(cond, TaxonomyNode.slug == slug)
        ).scalar_one_or_none()
        if node is None:
            try:
                with session.begin_nested():    # savepoint: only this insert rolls back on a race
                    node = TaxonomyNode(parent_id=parent_id, slug=slug, label=label[:200],
                                        depth=depth, event_count=0)
                    session.add(node)
                    session.flush()             # assign id; may hit the unique constraint
            except IntegrityError:
                node = session.execute(
                    select(TaxonomyNode).where(cond, TaxonomyNode.slug == slug)
                ).scalar_one()
        node.event_count = (node.event_count or 0) + 1
        parent_id = node.id
        leaf_id = node.id
    return leaf_id


# --- Phase 2: engagement heat (charter v0.3 §3.2) -----------------------------

def _minmax(values: dict) -> dict:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi <= lo:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def compute_node_heat(session, *, window_hours: int = 24, halflife_hours: float = 3.0,
                      forward_weight: float | None = None) -> dict[int, dict]:
    """Recency-weighted Telegram engagement heat per taxonomy node, rolled up the tree.

    Each event in the window contributes `recency * (1 + engagement)` (recency = an
    exponential decay by age, so fresh posts weigh more; engagement is the tier-weighted,
    reach-normalised rate — same as demand/topics) to its leaf AND every ancestor, so a
    node's raw score is the rolled-up activity of everything beneath it. Scores are then
    min-max normalized WITHIN each depth level, so a specific leaf can be 'hot' relative to
    its peers, not drowned by the always-large roots. Returns {node_id: {heat, events, depth}}.
    """
    from collections import defaultdict

    from sqlalchemy import select

    from newsroom.analyze.demand import DEFAULT_FORWARD_WEIGHT, TIER_WEIGHT, engagement_rate
    from newsroom.models import Event, EventItem, Item, ItemMetric, Source, SourceMetric, TaxonomyNode
    from newsroom.publishers.metrics import MessageStats

    fw = DEFAULT_FORWARD_WEIGHT if forward_weight is None else forward_weight
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=window_hours)

    ev_rows = session.execute(
        select(Event.id, Event.topic_leaf_id, Event.first_seen_at)
        .where(Event.first_seen_at >= cutoff, Event.topic_leaf_id.is_not(None))
    ).all()
    if not ev_rows:
        return {}

    # latest known subscriber count per source (reach normalisation)
    subs: dict[int, int] = {}
    for sid, n, _at in session.execute(
        select(SourceMetric.source_id, SourceMetric.subscribers, SourceMetric.measured_at)
        .order_by(SourceMetric.source_id, SourceMetric.measured_at.desc())
    ).all():
        if sid not in subs and n:
            subs[sid] = int(n)

    # engagement per event = tier-weighted, reach-normalised (latest metric per item)
    eng_by_event: dict[int, float] = defaultdict(float)
    seen_items: set[int] = set()
    for item_id, views, reactions, forwards, _at, event_id, source_id, tier in session.execute(
        select(ItemMetric.item_id, ItemMetric.views, ItemMetric.reactions, ItemMetric.forwards,
               ItemMetric.measured_at, EventItem.event_id, Source.id, Source.tier)
        .join(Item, Item.id == ItemMetric.item_id)
        .join(EventItem, EventItem.item_id == Item.id)
        .join(Source, Source.id == Item.source_id)
        .where(ItemMetric.measured_at >= cutoff)
        .order_by(ItemMetric.item_id, ItemMetric.measured_at.desc())
    ).all():
        if item_id in seen_items:
            continue
        seen_items.add(item_id)
        rate = engagement_rate(MessageStats(views=views, reactions=reactions, forwards=forwards),
                               subs.get(source_id), forward_weight=fw)
        if rate is not None:
            eng_by_event[event_id] += rate * TIER_WEIGHT.get((tier or "").lower(), 0.5)

    parent = dict(session.execute(select(TaxonomyNode.id, TaxonomyNode.parent_id)).all())
    depth = dict(session.execute(select(TaxonomyNode.id, TaxonomyNode.depth)).all())

    raw: dict[int, float] = defaultdict(float)
    ev_count: dict[int, int] = defaultdict(int)
    for eid, leaf_id, seen_at in ev_rows:
        if leaf_id not in parent:
            continue
        age_h = max((now - seen_at).total_seconds() / 3600.0, 0.0) if seen_at else float(window_hours)
        recency = math.exp(-age_h / halflife_hours) if halflife_hours > 0 else 1.0
        contribution = recency * (1.0 + eng_by_event.get(eid, 0.0))
        node = leaf_id
        guard = 0
        while node is not None and guard < 32:      # guard against a cycle
            raw[node] += contribution
            ev_count[node] += 1
            node = parent.get(node)
            guard += 1

    # normalize within each depth level
    by_level: dict[int, dict[int, float]] = defaultdict(dict)
    for nid, score in raw.items():
        by_level[depth.get(nid, 0)][nid] = score
    heat: dict[int, float] = {}
    for scores in by_level.values():
        heat.update(_minmax(scores))

    return {nid: {"heat": round(heat.get(nid, 0.0), 4), "events": ev_count[nid], "depth": depth.get(nid, 0)}
            for nid in raw}


def refresh_taxonomy_heat(session_factory, *, window_hours: int = 24,
                          halflife_hours: float = 3.0) -> dict:
    """One analytics tick: recompute node heat and write it onto taxonomy_nodes. Nodes with
    no activity in the window are reset to 0 so stale heat never lingers. Returns a summary."""
    from sqlalchemy import select

    from newsroom.models import TaxonomyNode

    with session_factory() as s:
        result = compute_node_heat(s, window_hours=window_hours, halflife_hours=halflife_hours)
        now = dt.datetime.now(dt.timezone.utc)
        active = set(result)
        for nid, d in result.items():
            node = s.get(TaxonomyNode, nid)
            if node is not None:
                node.heat = d["heat"]
                node.heat_events = d["events"]
                node.heat_at = now
        # clear stale heat on nodes that dropped out of the window
        for node in s.execute(select(TaxonomyNode).where(TaxonomyNode.heat > 0)).scalars():
            if node.id not in active:
                node.heat = 0.0
                node.heat_events = 0
                node.heat_at = now
        s.commit()
    return {"nodes": len(result)}


def node_path_labels(nodes_by_id: dict, node_id: int) -> list[str]:
    """Broad->specific display labels from the root down to node_id, via a preloaded map."""
    chain: list[str] = []
    nid = node_id
    guard = 0
    while nid is not None and guard < 32:
        node = nodes_by_id.get(nid)
        if node is None:
            break
        chain.append(node.label)
        nid = node.parent_id
        guard += 1
    return list(reversed(chain))


def load_event_heat(session, event_ids) -> dict[int, float]:
    """Per-event topic heat for curation: the MAX heat along the event's path (a hot broad
    topic or a hot specific leaf both count), from taxonomy_nodes. 0.0 when unplaced/cold.
    This is the pyramid's popularity signal that replaces the flat keyword hot-topics."""
    from sqlalchemy import select

    from newsroom.models import Event, TaxonomyNode

    event_ids = list(event_ids)
    if not event_ids:
        return {}
    leaf_of = dict(session.execute(
        select(Event.id, Event.topic_leaf_id).where(Event.id.in_(event_ids))
    ).all())
    nodes = {nid: (heat or 0.0, parent) for nid, heat, parent in session.execute(
        select(TaxonomyNode.id, TaxonomyNode.heat, TaxonomyNode.parent_id)
    ).all()}

    out: dict[int, float] = {}
    for eid in event_ids:
        h = 0.0
        nid = leaf_of.get(eid)
        guard = 0
        while nid is not None and guard < 32:
            rec = nodes.get(nid)
            if rec is None:
                break
            h = max(h, rec[0])
            nid = rec[1]
            guard += 1
        out[eid] = round(h, 4)
    return out


def load_top_nodes(session, *, limit: int = 25, min_depth: int = 1) -> list[dict]:
    """Hottest taxonomy nodes with their full path, for the admin console (/topics)."""
    from sqlalchemy import select

    from newsroom.models import TaxonomyNode

    all_nodes = {n.id: n for n in session.execute(select(TaxonomyNode)).scalars()}
    hot = sorted((n for n in all_nodes.values() if n.heat and n.depth >= min_depth),
                 key=lambda n: (n.heat, n.heat_events), reverse=True)[:limit]
    return [{"path": node_path_labels(all_nodes, n.id), "heat": round(n.heat, 3),
             "events": n.heat_events, "depth": n.depth} for n in hot]
