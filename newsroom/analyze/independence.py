"""Source-independence counting (charter §3.4).

"Передруки й переписування одного матеріалу рахуються як одне джерело." Given a
set of items already judged to be about the SAME event (event clustering is a
separate step), collapse them into distinct independent sources so the risk
matrix (§3.1) can ask "2+ independent sources?".

Two items are NOT independent when any of these holds:
  * same source (a channel is not independent from itself);
  * identical text (exact content_hash);
  * near-duplicate text (simhash Hamming distance <= threshold — verbatim/lightly
    edited reposts);
  * a forwarding relationship (same origin, or one is a forward of the other's
    source).

Pure and offline. The simhash threshold is a shadow-mode-calibrated parameter;
heavier rewrites are additionally collapsed by semantic event clustering (1б.5).
"""
from __future__ import annotations

from dataclasses import dataclass

from newsroom.collectors.base import hamming_distance

# Conservative default: collapses verbatim and lightly-edited reposts only.
# Raising it collapses heavier rewrites too (safer for the gate) at the risk of
# merging genuinely independent reports — tuned in shadow mode.
DEFAULT_SIMHASH_THRESHOLD = 6


@dataclass(frozen=True)
class SourceItem:
    source_id: int
    content_hash: str | None = None
    simhash: int | None = None
    forwarded_from: str | None = None
    source_name: str | None = None
    item_id: int | None = None


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


def _same_source(a: SourceItem, b: SourceItem, threshold: int) -> bool:
    if a.source_id == b.source_id:
        return True
    if a.content_hash and b.content_hash and a.content_hash == b.content_hash:
        return True
    if a.simhash is not None and b.simhash is not None and hamming_distance(a.simhash, b.simhash) <= threshold:
        return True
    fa, fb = _norm(a.forwarded_from), _norm(b.forwarded_from)
    if fa and fb and fa == fb:                       # forwards of the same origin
        return True
    an, bn = _norm(a.source_name), _norm(b.source_name)
    if fa and bn and fa == bn:                       # a is a forward of b's source
        return True
    if fb and an and fb == an:                       # b is a forward of a's source
        return True
    return False


def group_materials(items: list[SourceItem], *, simhash_threshold: int = DEFAULT_SIMHASH_THRESHOLD) -> list[list[int]]:
    """Union-find grouping. Returns lists of item indices, one list per distinct
    independent source."""
    n = len(items)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if _same_source(items[i], items[j], simhash_threshold):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)
    return list(groups.values())


def independent_source_count(items: list[SourceItem], *, simhash_threshold: int = DEFAULT_SIMHASH_THRESHOLD) -> int:
    """Number of distinct independent sources among same-event items."""
    if not items:
        return 0
    return len(group_materials(items, simhash_threshold=simhash_threshold))
