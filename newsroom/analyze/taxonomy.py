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

import re

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
