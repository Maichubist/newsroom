"""Source reputation as a journal of events (architecture §4, §9).

Reputation is not a stored number but a log of facts about each source — it was
first with a story, it turned out confirmed or refuted, it merely copied — from
which a score is computed and can always be recomputed (§4, principle 4). This
module writes those facts and derives the score:

  * origins  — when an event forms, its origin source is credited "first"; items
    explicitly marked as copies get "copy".
  * outcome  — when an event settles as confirmed/refuted, its origin source gets
    "confirmed"/"refuted" (leak-tier sources get the leak_* variants).

Writes are idempotent (deduped on source+event+kind) so the batch runner can
re-scan safely. Pure score weights are offline-tested; the DB writes are pg-tested.
"""
from __future__ import annotations

import logging

log = logging.getLogger("newsroom.reputation")

# reputation_events.kind (architecture §5.4)
REP_FIRST = "first"
REP_CONFIRMED = "confirmed"
REP_REFUTED = "refuted"
REP_COPY = "copy"
REP_LEAK_CONFIRMED = "leak_confirmed"
REP_LEAK_REFUTED = "leak_refuted"

_OUTCOME_KINDS = (REP_CONFIRMED, REP_REFUTED, REP_LEAK_CONFIRMED, REP_LEAK_REFUTED)

# score weights: being first and confirmed is good, being refuted is costly
# (asymmetry of errors, §3.5), copying is mildly negative.
SCORE_WEIGHTS = {
    REP_FIRST: 2.0,
    REP_CONFIRMED: 3.0,
    REP_LEAK_CONFIRMED: 2.0,
    REP_COPY: -0.5,
    REP_REFUTED: -4.0,
    REP_LEAK_REFUTED: -3.0,
}

# events for which we credit an origin at all
_ORIGIN_STATUSES = ("reported", "confirmed", "rumor", "refuted")


def score_from_counts(counts: dict[str, int]) -> float:
    """Reputation score from a {kind: count} tally. Pure — the same math the
    read helper applies to the journal, so it is what tests pin down."""
    return float(sum(SCORE_WEIGHTS.get(kind, 0.0) * n for kind, n in counts.items()))


def record_reputation(session, source_id: int, kind: str, *, event_id: int | None = None,
                      item_id: int | None = None) -> int | None:
    """Append one reputation fact, deduped on (source_id, event_id, kind). Returns
    the new row id, or None if that fact was already recorded. Flushes, no commit."""
    from sqlalchemy import select

    from newsroom.models import ReputationEvent

    exists = session.scalar(
        select(ReputationEvent.id).where(
            ReputationEvent.source_id == source_id,
            ReputationEvent.event_id == event_id,
            ReputationEvent.kind == kind,
        )
    )
    if exists is not None:
        return None
    row = ReputationEvent(source_id=source_id, event_id=event_id, item_id=item_id, kind=kind)
    session.add(row)
    session.flush()
    return row.id


def _origin_source_id(session, event) -> int | None:
    from sqlalchemy import select

    from newsroom.models import EventItem, Item

    if event.first_source_id is not None:
        return event.first_source_id
    return session.scalar(
        select(Item.source_id)
        .join(EventItem, EventItem.item_id == Item.id)
        .where(EventItem.event_id == event.id, EventItem.role == "origin")
    )


def record_event_origins(session_factory, event_id: int) -> dict[str, int]:
    """Credit the origin source "first" and any explicit copies "copy"."""
    from sqlalchemy import select

    from newsroom.models import Event, EventItem, Item

    counts = {"first": 0, "copy": 0}
    with session_factory() as s:
        event = s.get(Event, event_id)
        if event is None:
            return counts
        origin = _origin_source_id(s, event)
        if origin is not None and record_reputation(s, origin, REP_FIRST, event_id=event_id):
            counts["first"] += 1
        copies = s.execute(
            select(Item.source_id)
            .join(EventItem, EventItem.item_id == Item.id)
            .where(EventItem.event_id == event_id, EventItem.role == "copy")
            .distinct()
        ).scalars().all()
        for sid in copies:
            if sid != origin and record_reputation(s, sid, REP_COPY, event_id=event_id):
                counts["copy"] += 1
        s.commit()
    return counts


def record_event_outcome(session_factory, event_id: int) -> str | None:
    """When an event settles confirmed/refuted, credit its origin source. Returns
    the kind written, or None if nothing applied / already recorded."""
    from newsroom.models import Event, Source

    with session_factory() as s:
        event = s.get(Event, event_id)
        if event is None or event.status not in ("confirmed", "refuted"):
            return None
        origin = _origin_source_id(s, event)
        if origin is None:
            return None
        source = s.get(Source, origin)
        is_leak = bool(source and source.tier == "leak")
        if event.status == "confirmed":
            kind = REP_LEAK_CONFIRMED if is_leak else REP_CONFIRMED
        else:
            kind = REP_LEAK_REFUTED if is_leak else REP_REFUTED
        wrote = record_reputation(s, origin, kind, event_id=event_id)
        s.commit()
        return kind if wrote else None


def reputation_score(session, source_id: int) -> dict:
    """Recompute a source's reputation from its journal: {counts, score}."""
    from sqlalchemy import func, select

    from newsroom.models import ReputationEvent

    rows = session.execute(
        select(ReputationEvent.kind, func.count())
        .where(ReputationEvent.source_id == source_id)
        .group_by(ReputationEvent.kind)
    ).all()
    counts = {kind: int(n) for kind, n in rows}
    return {"counts": counts, "score": score_from_counts(counts)}


def record_pending(session_factory, *, limit: int = 50) -> dict[str, int]:
    """One reputation tick: credit origins for newsworthy events that have none,
    and outcomes for settled events that have none. Idempotent via the dedup."""
    from sqlalchemy import select

    from newsroom.models import Event, ReputationEvent

    with session_factory() as s:
        have_any = select(ReputationEvent.event_id).where(ReputationEvent.event_id.is_not(None)).distinct()
        origin_ids = list(s.execute(
            select(Event.id)
            .where(Event.status.in_(_ORIGIN_STATUSES), Event.id.not_in(have_any))
            .order_by(Event.id).limit(limit)
        ).scalars().all())
        have_outcome = select(ReputationEvent.event_id).where(ReputationEvent.kind.in_(_OUTCOME_KINDS)).distinct()
        outcome_ids = list(s.execute(
            select(Event.id)
            .where(Event.status.in_(("confirmed", "refuted")), Event.id.not_in(have_outcome))
            .order_by(Event.id).limit(limit)
        ).scalars().all())

    stats = {"origins": 0, "outcomes": 0}
    for event_id in origin_ids:
        counts = record_event_origins(session_factory, event_id)
        if counts["first"] or counts["copy"]:
            stats["origins"] += 1
    for event_id in outcome_ids:
        if record_event_outcome(session_factory, event_id):
            stats["outcomes"] += 1
    return stats
