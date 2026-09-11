"""Verification pass (architecture §4, stage 1б integration).

Wires the deterministic analyzers together over a collected item:

    noise filter -> classify (event? rubric? side?) -> embed + cluster into an
    event -> gather the event's evidence (independent sources, official/reputable
    sources) -> risk gate -> set item/event status; every decision is journalled.

The two judgment steps (event-vs-noise and rubric) are a pluggable Classifier —
an LLM in production (comparative, not absolute — CLAUDE.md), a fake in tests.
Embedding is a pluggable Embedder. Everything else is the deterministic layer
built in 1б.1–1б.5. Nothing is published here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from newsroom.analyze.clustering import DEFAULT_THRESHOLD, DEFAULT_WINDOW_HOURS, EventClusterer
from newsroom.analyze.independence import SourceItem, independent_source_count
from newsroom.analyze.risk import RiskMatrix, decide
from newsroom.analyze.signal import FiltersConfig, classify_noise, ipso_markers
from newsroom.analyze.stoplist import StopRule, check as stoplist_check, worst_action


@dataclass(frozen=True)
class Classification:
    is_event: bool
    rubrics: list[str] = field(default_factory=list)
    side: str = "unknown"           # ua | ru | unknown (for scoped stop-list rules)
    is_first_source: bool = False   # document / court ruling / party's own statement
    is_rumor: bool = False          # leak-channel rumor (charter 3.7)


class Classifier(Protocol):
    model: str

    def classify(self, title: str | None, text: str | None) -> Classification: ...


class Embedder(Protocol):
    model: str

    def embed(self, text: str): ...


@dataclass(frozen=True)
class VerifyResult:
    item_status: str
    event_id: int | None = None
    event_status: str | None = None
    reason: str | None = None


class Verifier:
    def __init__(
        self,
        session_factory,
        *,
        classifier: Classifier,
        embedder: Embedder,
        risk_matrix: RiskMatrix,
        filters: FiltersConfig,
        stoplist_rules: list[StopRule],
        threshold: float = DEFAULT_THRESHOLD,
        window_hours: int = DEFAULT_WINDOW_HOURS,
        charter_version: str = "0.2",
    ):
        self.sf = session_factory
        self.classifier = classifier
        self.embedder = embedder
        self.risk_matrix = risk_matrix
        self.filters = filters
        self.stoplist_rules = stoplist_rules
        self.clusterer = EventClusterer(session_factory, threshold=threshold, window_hours=window_hours)
        self.charter_version = charter_version

    # ------------------------------------------------------------------
    def verify_item(self, item_id: int) -> VerifyResult:
        from newsroom.models import Item

        with self.sf() as s:
            item = s.get(Item, item_id)
            if item is None:
                return VerifyResult("missing", reason="item not found")
            title, text = item.title, item.text

        # 1. obvious noise -> drop
        noise = classify_noise(title, text, self.filters)
        if noise.is_noise:
            self._set_item_status(item_id, "filtered_out")
            self._journal("item", item_id, "filter", "noise", ",".join(noise.reasons))
            return VerifyResult("filtered_out", reason="noise:" + ",".join(noise.reasons))

        # 2. event vs noise + rubric (LLM in prod)
        cls = self.classifier.classify(title, text)
        if not cls.is_event:
            self._set_item_status(item_id, "filtered_out")
            self._journal("item", item_id, "filter", "not_event", "")
            return VerifyResult("filtered_out", reason="not_event")

        # 3. embed + persist + cluster into an event
        vec = self.embedder.embed(f"{title or ''}\n{text or ''}")
        self._store_embedding(item_id, vec)
        assign = self.clusterer.assign(item_id, vec)
        self._set_item_status(item_id, "clustered")

        # 4. event evidence + risk gate
        event_status, level, indep = self._verify_event(assign.event_id, cls)

        markers = ipso_markers(title, text, self.filters)
        violations = stoplist_check(title, text, self.stoplist_rules, side=cls.side)
        self._journal(
            "item", item_id, "verify", event_status,
            reason=f"level={level} independent_sources={indep}",
            details={
                "event_id": assign.event_id,
                "created_new_event": assign.created_new,
                "similarity": round(assign.similarity, 4),
                "rubrics": cls.rubrics,
                "ipso_markers": markers,
                "stoplist": worst_action(violations),
                "stoplist_rules": [v.rule_id for v in violations],
            },
            model=getattr(self.classifier, "model", None),
        )
        return VerifyResult("clustered", assign.event_id, event_status)

    # ------------------------------------------------------------------
    def _verify_event(self, event_id: int, cls: Classification) -> tuple[str, str, int]:
        from sqlalchemy import select

        from newsroom.models import Event, EventItem, Item, Source

        with self.sf() as s:
            rows = s.execute(
                select(Item, Source)
                .join(EventItem, EventItem.item_id == Item.id)
                .join(Source, Source.id == Item.source_id)
                .where(EventItem.event_id == event_id)
            ).all()

            src_items = [
                SourceItem(source_id=it.source_id, content_hash=it.content_hash, simhash=it.simhash,
                           forwarded_from=it.forwarded_from, source_name=src.name)
                for it, src in rows
            ]
            indep = independent_source_count(src_items)
            has_official = any(src.is_official or src.tier == "official" for _, src in rows)
            high_rep = any(src.tier in ("official", "media") for _, src in rows)

            level = self.risk_matrix.level_for(cls.rubrics)
            gate = decide(
                level,
                independent_sources=indep,
                has_official=has_official,
                has_first_source=cls.is_first_source,
                high_reputation=high_rep,
                is_rumor=cls.is_rumor,
            )

            event = s.get(Event, event_id)
            event.rubric = cls.rubrics[0] if cls.rubrics else None
            event.risk_level = level
            event.status = gate.status
            event.independent_source_count = indep
            s.commit()

        self._journal("event", event_id, "verify", gate.status, reason=gate.reason,
                      details={"level": level, "independent_sources": indep,
                               "has_official": has_official})
        return gate.status, level, indep

    # ------------------------------------------------------------------
    def _set_item_status(self, item_id: int, status: str) -> None:
        from newsroom.models import Item

        with self.sf() as s:
            item = s.get(Item, item_id)
            if item is not None:
                item.status = status
            s.commit()

    def _store_embedding(self, item_id: int, vec) -> None:
        import numpy as np
        from sqlalchemy import select

        from newsroom.models import ItemEmbedding

        model = getattr(self.embedder, "model", "unknown")
        with self.sf() as s:
            exists = s.scalar(
                select(ItemEmbedding.id).where(ItemEmbedding.item_id == item_id, ItemEmbedding.model == model)
            )
            if exists is None:
                s.add(ItemEmbedding(item_id=item_id, model=model,
                                    vector=np.asarray(vec, dtype=np.float64).tolist()))
                s.commit()

    def _journal(self, entity_type: str, entity_id: int, stage: str, decision: str,
                 reason: str | None = None, details: dict | None = None, model: str | None = None) -> None:
        from newsroom.models import Decision

        with self.sf() as s:
            s.add(Decision(
                entity_type=entity_type, entity_id=str(entity_id), stage=stage,
                decision=decision, reason=reason, details=details,
                charter_version=self.charter_version, model=model,
            ))
            s.commit()


def verify_pending(session_factory, verifier: Verifier, *, limit: int = 50) -> dict[str, int]:
    """One verification tick: take a batch of `new` items and run each through the
    verifier. Uses SELECT … FOR UPDATE SKIP LOCKED so parallel workers don't grab
    the same rows (architecture §4). verify_item moves each item off `new`, so the
    next tick sees fresh items."""
    from sqlalchemy import select

    from newsroom.models import Item

    with session_factory() as s:
        ids = list(s.execute(
            select(Item.id)
            .where(Item.status == "new")
            .order_by(Item.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).scalars().all())

    stats: dict[str, int] = {"processed": 0, "errors": 0}
    for item_id in ids:
        try:
            result = verifier.verify_item(item_id)
        except Exception:  # noqa: BLE001 - one bad item must not block the whole queue
            _mark_item_error(session_factory, item_id)
            stats["errors"] += 1
            continue
        stats["processed"] += 1
        stats[result.item_status] = stats.get(result.item_status, 0) + 1
    return stats


def _mark_item_error(session_factory, item_id: int) -> None:
    """Move a poison item off `new` so it is not retried forever. It can be reset
    to `new` for replay once the cause is fixed (architecture §3, replay)."""
    import logging

    from newsroom.models import Item

    logging.getLogger("newsroom.analyze.verify").warning(
        "verify_item failed; marking item error", extra={"item_id_": item_id})
    try:
        with session_factory() as s:
            item = s.get(Item, item_id)
            if item is not None and item.status == "new":
                item.status = "error"
                s.commit()
    except Exception:  # noqa: BLE001 - never let the error handler itself break the tick
        logging.getLogger("newsroom.analyze.verify").exception("could not mark item error")
