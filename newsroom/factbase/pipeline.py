"""Fact-base orchestration (architecture §8).

For one event: pull each source's material separately, extract that source's
facts and reactions, embed the facts, merge them into the shared base (confirmed
across sources, number divergence) and store it on events.fact_base. Extractor
and embedder are pluggable (LLM in prod, fakes in tests). Idempotent: an event
that already has a fact base is skipped.
"""
from __future__ import annotations

from dataclasses import dataclass

from newsroom.factbase.builder import (
    DEFAULT_FACT_THRESHOLD,
    FactExtractor,
    VectorFact,
    fact_base_json,
    merge_facts,
)

# events worth a fact base: the same ones that head toward a draft.
FACTBASE_STATUSES = ("reported", "confirmed", "rumor")


@dataclass(frozen=True)
class FactBaseResult:
    event_id: int
    facts: int = 0
    sources: int = 0
    reactions: int = 0
    skipped: bool = False


class FactBaseBuilder:
    def __init__(self, session_factory, *, extractor: FactExtractor, embedder,
                 threshold: float = DEFAULT_FACT_THRESHOLD, max_facts_per_source: int = 20):
        self.sf = session_factory
        self.extractor = extractor
        self.embedder = embedder
        self.threshold = threshold
        self.max_facts_per_source = max_facts_per_source

    def build_event(self, event_id: int) -> FactBaseResult:
        from sqlalchemy import select

        from newsroom.models import Event, EventItem, Item, Source

        with self.sf() as s:
            event = s.get(Event, event_id)
            if event is None or event.fact_base:
                return FactBaseResult(event_id, skipped=True)
            rows = list(s.execute(
                select(Item.source_id, Source.name, Item.title, Item.text)
                .join(Source, Source.id == Item.source_id)
                .join(EventItem, EventItem.item_id == Item.id)
                .where(EventItem.event_id == event_id)
            ).all())

        # group each source's material together (one viewpoint per source, §8.1)
        per_source: dict[int, list] = {}
        for source_id, name, title, text in rows:
            entry = per_source.setdefault(source_id, [name, []])
            entry[1].append(f"{title or ''}\n{text or ''}".strip())
        if not per_source:
            return FactBaseResult(event_id, skipped=True)

        vector_facts: list[VectorFact] = []
        reactions: list[tuple[int, object]] = []
        for source_id, (name, texts) in per_source.items():
            facts = self.extractor.extract(name, None, "\n".join(texts))[: self.max_facts_per_source]
            for fact in facts:
                if fact.kind == "reaction":
                    reactions.append((source_id, fact))
                    continue
                vec = self.embedder.embed(fact.text)
                vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
                vector_facts.append(VectorFact(fact=fact, source_id=source_id, vector=vec_list))

        merged = merge_facts(vector_facts, threshold=self.threshold)
        base = fact_base_json(merged, reactions)

        with self.sf() as s:
            event = s.get(Event, event_id)
            if event is not None:
                event.fact_base = base
                s.commit()

        return FactBaseResult(event_id, facts=len(merged), sources=len(per_source),
                              reactions=len(reactions))


def build_pending(session_factory, builder: "FactBaseBuilder", *, limit: int = 25,
                  significance_threshold: float | None = None,
                  classify_grace_seconds: float = 180.0,
                  require_dedup_settled: bool = False,
                  dedup_grace_seconds: float = 300.0) -> dict[str, int]:
    """One fact-base tick: build the shared base for publishable events that have
    none yet. Once built, the event has a fact_base and is skipped next tick. When a
    significance threshold is given, low-significance events are skipped — no tokens
    spent on news that will not be posted. With require_dedup_settled, an event also
    waits for the ingest-dedup verdict (or the grace) so a duplicate is merged before
    a fact base is built for it (Phase B savings)."""
    import datetime as dt

    from sqlalchemy import select

    from newsroom.analyze.ingest_dedup import dedup_settled_clause
    from newsroom.analyze.significance import significance_ready_clause
    from newsroom.models import Event

    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(seconds=classify_grace_seconds)
    # a duplicate event (ingest/publish dedup) is never posted — don't spend a fact base on it
    conditions = [Event.status.in_(FACTBASE_STATUSES), Event.fact_base.is_(None),
                  Event.duplicate_of.is_(None)]
    clause = significance_ready_clause(significance_threshold, cutoff)
    if clause is not None:
        conditions.append(clause)
    dedup_clause = dedup_settled_clause(require_dedup_settled,
                                        now - dt.timedelta(seconds=dedup_grace_seconds))
    if dedup_clause is not None:
        conditions.append(dedup_clause)

    with session_factory() as s:
        ids = list(s.execute(
            select(Event.id).where(*conditions).order_by(Event.id).limit(limit)
        ).scalars().all())

    stats = {"events": 0, "facts": 0}
    for event_id in ids:
        result = builder.build_event(event_id)
        if result.skipped:
            continue
        stats["events"] += 1
        stats["facts"] += result.facts
    return stats
