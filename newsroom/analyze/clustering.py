"""Incremental event clustering (architecture §7).

An item's embedding is compared (cosine) to the centroids of recent open events;
if the best match clears the threshold the item joins that event and the centroid
is updated as a running mean, otherwise a new event opens. Vector similarity, not
LLM-generated keys, defines identity (CLAUDE.md).

The math is pure/offline; EventClusterer persists to events/event_items. Threshold
and window are shadow-mode-calibrated parameters.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

DEFAULT_THRESHOLD = 0.83
DEFAULT_WINDOW_HOURS = 48


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def update_centroid(centroid, count: int, vec) -> list[float]:
    """Running mean: new = (centroid*count + vec) / (count+1)."""
    c = np.asarray(centroid, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    return ((c * count + v) / (count + 1)).tolist()


def best_match(vec, centroids: list, threshold: float = DEFAULT_THRESHOLD) -> tuple[int | None, float]:
    """Index of the nearest centroid at/above threshold and its similarity;
    (None, best_sim) when nothing clears the threshold."""
    best_i, best_sim = None, -1.0
    for i, c in enumerate(centroids):
        sim = cosine(vec, c)
        if sim > best_sim:
            best_i, best_sim = i, sim
    if best_i is None or best_sim < threshold:
        return None, (best_sim if best_sim >= 0 else 0.0)
    return best_i, best_sim


@dataclass(frozen=True)
class AssignResult:
    event_id: int
    created_new: bool
    similarity: float


class EventClusterer:
    def __init__(self, session_factory, *, threshold: float = DEFAULT_THRESHOLD,
                 window_hours: int = DEFAULT_WINDOW_HOURS):
        self.session_factory = session_factory
        self.threshold = threshold
        self.window_hours = window_hours

    def assign(self, item_id: int, vec) -> AssignResult:
        from sqlalchemy import func, select

        from newsroom.models import Event, EventItem

        vec_list = np.asarray(vec, dtype=np.float64).tolist()
        now = dt.datetime.now(dt.timezone.utc)
        cutoff = now - dt.timedelta(hours=self.window_hours)

        with self.session_factory() as s:
            events = list(s.execute(
                select(Event).where(
                    Event.centroid.is_not(None),
                    Event.updated_at >= cutoff,
                    Event.duplicate_of.is_(None),   # a merged duplicate is inert — never a cluster target
                )
            ).scalars().all())

            idx, sim = best_match(vec_list, [e.centroid for e in events], self.threshold)

            if idx is not None:
                event = events[idx]
                n = int(s.scalar(select(func.count()).select_from(EventItem).where(EventItem.event_id == event.id)) or 0)
                event.centroid = update_centroid(event.centroid, n, vec_list)
                s.add(EventItem(event_id=event.id, item_id=item_id, role=None, similarity=float(sim)))
                s.commit()
                return AssignResult(event.id, False, float(sim))

            event = Event(centroid=vec_list, status="signal", first_seen_at=now)
            s.add(event)
            s.flush()
            s.add(EventItem(event_id=event.id, item_id=item_id, role="origin", similarity=1.0))
            s.commit()
            return AssignResult(event.id, True, float(sim))
