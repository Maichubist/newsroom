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
import hashlib
import logging
import math
import re
from typing import Protocol

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
    its peers, not drowned by the always-large roots. `demand` is the same rollup but on
    engagement ALONE (no recency) — a steadier "how much does the audience care about this
    topic" signal. Returns {node_id: {heat, demand, events, depth}}.
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

    raw: dict[int, float] = defaultdict(float)          # heat: recency × (1 + engagement)
    eng_raw: dict[int, float] = defaultdict(float)      # demand: engagement only (no recency)
    ev_count: dict[int, int] = defaultdict(int)
    for eid, leaf_id, seen_at in ev_rows:
        if leaf_id not in parent:
            continue
        age_h = max((now - seen_at).total_seconds() / 3600.0, 0.0) if seen_at else float(window_hours)
        recency = math.exp(-age_h / halflife_hours) if halflife_hours > 0 else 1.0
        eng = eng_by_event.get(eid, 0.0)
        contribution = recency * (1.0 + eng)
        node = leaf_id
        guard = 0
        while node is not None and guard < 32:      # guard against a cycle
            raw[node] += contribution
            eng_raw[node] += eng
            ev_count[node] += 1
            node = parent.get(node)
            guard += 1

    def _norm_by_level(scores_map: dict[int, float]) -> dict[int, float]:
        by_level: dict[int, dict[int, float]] = defaultdict(dict)
        for nid, score in scores_map.items():
            by_level[depth.get(nid, 0)][nid] = score
        out: dict[int, float] = {}
        for scores in by_level.values():
            out.update(_minmax(scores))
        return out

    heat = _norm_by_level(raw)
    # demand is pure audience engagement; when NO engagement exists yet, leave it empty
    # (so curation falls back to the L1 rubric-demand index) rather than a flat 0.5.
    demand = _norm_by_level(eng_raw) if any(v > 0 for v in eng_raw.values()) else {}

    return {nid: {"heat": round(heat.get(nid, 0.0), 4), "demand": round(demand.get(nid, 0.0), 4),
                  "events": ev_count[nid], "depth": depth.get(nid, 0)}
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
                node.demand = d["demand"]
                node.heat_events = d["events"]
                node.heat_at = now
        # clear stale heat/demand on nodes that dropped out of the window
        for node in s.execute(select(TaxonomyNode).where(TaxonomyNode.heat > 0)).scalars():
            if node.id not in active:
                node.heat = 0.0
                node.demand = 0.0
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


def load_event_signals(session, event_ids, *, prefer_facets: bool = False) -> dict[int, tuple[float, float]]:
    """Per-event (heat, demand) from the event's L2 (depth-1) topic node — the two-top-
    levels sweet spot for popularity (L1 too coarse: all war; L3+ too sparse/noisy). Falls
    back to the L1 (depth-0) node, else (0.0, 0.0). Keying off the SPECIFIC level is what
    lets a routine sub-topic (війна>втрати) stay 'cold' even under a hot broad rubric,
    instead of inheriting the hottest ancestor as the old max-along-path did."""
    from sqlalchemy import select

    from newsroom.models import Event, TaxonomyNode

    event_ids = list(event_ids)
    if not event_ids:
        return {}
    leaf_of = dict(session.execute(
        select(Event.id, Event.topic_leaf_id).where(Event.id.in_(event_ids))
    ).all())
    nodes = {nid: (heat or 0.0, demand or 0.0, parent, dep)
             for nid, heat, demand, parent, dep in session.execute(
                 select(TaxonomyNode.id, TaxonomyNode.heat, TaxonomyNode.demand,
                        TaxonomyNode.parent_id, TaxonomyNode.depth)).all()}

    out: dict[int, tuple[float, float]] = {}
    for eid in event_ids:
        nid = leaf_of.get(eid)
        l1: tuple[float, float] | None = None
        l2: tuple[float, float] | None = None
        guard = 0
        while nid is not None and guard < 32:
            rec = nodes.get(nid)
            if rec is None:
                break
            heat, demand, parent, dep = rec
            if dep == 0:
                l1 = (heat, demand)
            elif dep == 1:
                l2 = (heat, demand)
            nid = parent
            guard += 1
        sig = l2 if l2 is not None else (l1 if l1 is not None else (0.0, 0.0))
        out[eid] = (round(sig[0], 4), round(sig[1], 4))
    if prefer_facets:
        from newsroom.analyze.facets import load_event_facet_signals

        # Facets are additive and may only exist for newly classified events.  Override
        # legacy path signals where available; retain the old pyramid as a rollout/backfill
        # fallback rather than turning missing facet data into zeros.
        out.update(load_event_facet_signals(session, event_ids))
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


# --- synonym-node merge (charter v0.3: keep the learned tree from fragmenting) --

class SynonymGrouper(Protocol):
    model: str

    def group(self, nodes: list[tuple[int, str]]) -> list[list[int]]: ...


DEFAULT_MERGE_PROMPT = """Нижче — дочірні теми ОДНОГО батьківського вузла дерева тем
(id і назва). Згрупуй ТІ, ЩО ОЗНАЧАЮТЬ ОДНУ Й ТУ САМУ ТЕМУ (синоніми чи різні
формулювання одного: «удар бпла» / «атака дронів» / «дронова атака»; «смартфон» /
«телефон»). НЕ групуй просто пов'язані чи сусідні теми: «удар бпла» і «удар каб» —
РІЗНІ; «одеса» і «львів» — РІЗНІ.

Поверни лише JSON: {"groups": [[id, id, ...], ...]} — лише групи з 2+ синонімів.

ТЕМИ:
{nodes}"""


def _render_nodes(nodes: list[tuple[int, str]]) -> str:
    return "\n".join(f"[{nid}] {(label or '').strip()}" for nid, label in nodes)


def taxonomy_children_hash(nodes: list[tuple[int, str]]) -> str:
    """Stable identity of a sibling set. Counts/heat do not affect synonymy."""
    payload = "\n".join(
        f"{nid}:{normalize_label(label)}" for nid, label in sorted(nodes, key=lambda row: row[0])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMSynonymGrouper:  # pragma: no cover - network
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini",
                 prompt: str = DEFAULT_MERGE_PROMPT):
        import os

        self.model = model
        self.prompt = prompt
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def group(self, nodes: list[tuple[int, str]]) -> list[list[int]]:
        from newsroom.editorial.dedup import parse_groups
        from newsroom.promptutil import fill_prompt

        valid = {nid for nid, _ in nodes}
        if len(valid) < 2:
            return []
        content = fill_prompt(self.prompt, nodes=_render_nodes(nodes))
        try:
            from newsroom.llmutil import chat_json

            raw = chat_json(self._ensure_client(), model=self.model,
                            messages=[{"role": "user", "content": content}],
                            op="taxonomy_merge", max_tokens=1024)
            return parse_groups(raw, valid)
        except Exception as exc:  # noqa: BLE001
            log.warning("synonym group failed", extra={"error": str(exc)})
            return []


def merge_node(session, victim_id: int, canonical_id: int) -> None:
    """Merge sibling node `victim` into `canonical`: re-point victim's children (recursing
    on a slug collision so grandchildren fold together too) and its events, add up
    event_count, then delete victim. Caller commits."""
    from sqlalchemy import select, update

    from newsroom.models import Event, TaxonomyNode

    if victim_id == canonical_id:
        return
    victim = session.get(TaxonomyNode, victim_id)
    canonical = session.get(TaxonomyNode, canonical_id)
    if victim is None or canonical is None:
        return

    children = list(session.execute(
        select(TaxonomyNode).where(TaxonomyNode.parent_id == victim_id)).scalars())
    canon_children = {c.slug: c for c in session.execute(
        select(TaxonomyNode).where(TaxonomyNode.parent_id == canonical_id)).scalars()}
    for child in children:
        existing = canon_children.get(child.slug)
        if existing is not None and existing.id != child.id:
            merge_node(session, child.id, existing.id)      # same-slug grandchild -> recurse
        else:
            child.parent_id = canonical_id
            canon_children[child.slug] = child
    session.flush()

    session.execute(update(Event).where(Event.topic_leaf_id == victim_id)
                    .values(topic_leaf_id=canonical_id))
    canonical.event_count = (canonical.event_count or 0) + (victim.event_count or 0)
    session.flush()
    session.delete(victim)
    session.flush()


def merge_synonym_nodes(session_factory, grouper: "SynonymGrouper", *, limit_parents: int = 500) -> dict:
    """One merge tick: for each parent with 2+ children, ask the grouper which children are
    the SAME topic and fold synonyms into a canonical node (the one with the most events).
    Keeps the learned tree from fragmenting into near-duplicate topics (analog of the dedup
    story-merge). Returns {merged, groups}."""
    from collections import defaultdict

    from sqlalchemy import select

    from newsroom.models import TaxonomyNode

    with session_factory() as s:
        rows = s.execute(select(TaxonomyNode.id, TaxonomyNode.parent_id,
                                TaxonomyNode.label, TaxonomyNode.event_count)).all()
    by_parent: dict[int | None, list[tuple[int, str, int]]] = defaultdict(list)
    for nid, pid, label, ec in rows:
        by_parent[pid].append((nid, label, ec or 0))

    merged = 0
    groups_found = 0
    cached = 0
    for parent_id, children in list(by_parent.items())[:limit_parents]:
        if len(children) < 2:
            continue
        from newsroom.models import SystemState

        cache_key = f"taxonomy_merge:{parent_id if parent_id is not None else 'root'}"
        input_hash = taxonomy_children_hash([(nid, label) for nid, label, _ in children])
        with session_factory() as s:
            state = s.get(SystemState, cache_key)
            if state is not None and (state.value or {}).get("children_hash") == input_hash:
                cached += 1
                continue

        from newsroom.llmutil import llm_context

        with llm_context(stage="taxonomy_merge", taxonomy_parent_id=parent_id,
                         children_hash=input_hash):
            groups = grouper.group([(nid, label) for nid, label, _ in children])
        ec_map = {nid: ec for nid, _, ec in children}
        with session_factory() as s:
            for group in groups:
                canonical = max(group, key=lambda nid: (ec_map.get(nid, 0), -nid))  # most events, tie->lowest id
                for nid in group:
                    if nid != canonical:
                        merge_node(s, nid, canonical)
                        merged += 1
                groups_found += 1
            s.flush()
            remaining = list(s.execute(
                select(TaxonomyNode.id, TaxonomyNode.label)
                .where(TaxonomyNode.parent_id == parent_id)
            ).all())
            final_hash = taxonomy_children_hash(remaining)
            state = s.get(SystemState, cache_key)
            value = {"children_hash": final_hash, "child_count": len(remaining)}
            if state is None:
                s.add(SystemState(key=cache_key, value=value))
            else:
                state.value = value
            s.commit()
    return {"merged": merged, "groups": groups_found, "cached": cached}
