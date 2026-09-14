"""Verification pass (architecture §4, stage 1б integration).

Wires the deterministic analyzers together over a collected item:

    noise filter -> embed + cluster into an event -> classify the event ONCE
    (event? rubric? side?) -> gather the event's evidence (independent sources,
    official/reputable sources) -> risk gate -> set item/event status; every
    decision is journalled.

Order matters for cost: embedding + clustering are cheap and run first, so the
LLM classifier fires once per *event* rather than once per *item*. Reprint-heavy
feeds (aggregators repost the same news) collapse many items into one event, and
each reprint that joins an already-classified event reuses the cached
classification and pays nothing. The deterministic noise filter still runs first,
so obvious junk is never embedded.

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
from newsroom.analyze.signal import FiltersConfig, classify_noise, ipso_markers, is_air_alert
from newsroom.analyze.stoplist import StopRule, check as stoplist_check, worst_action


@dataclass(frozen=True)
class Classification:
    is_event: bool
    rubrics: list[str] = field(default_factory=list)
    side: str = "unknown"           # ua | ru | unknown (for scoped stop-list rules)
    is_first_source: bool = False   # document / court ruling / party's own statement
    is_rumor: bool = False          # leak-channel rumor (charter 3.7)
    keywords: list[str] = field(default_factory=list)   # 5-10 topic keywords (hot-topics layer)


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
        require_official_for_critical: bool = True,
    ):
        self.sf = session_factory
        self.classifier = classifier
        self.embedder = embedder
        self.risk_matrix = risk_matrix
        self.filters = filters
        self.stoplist_rules = stoplist_rules
        self.clusterer = EventClusterer(session_factory, threshold=threshold, window_hours=window_hours)
        self.charter_version = charter_version
        # test-channel: waive the critical→official rule (charter floor otherwise)
        self.require_official_for_critical = bool(require_official_for_critical)

    # ------------------------------------------------------------------
    def verify_item(self, item_id: int) -> VerifyResult:
        from newsroom.models import Item

        with self.sf() as s:
            item = s.get(Item, item_id)
            if item is None:
                return VerifyResult("missing", reason="item not found")
            title, text = item.title, item.text

        # 1. obvious noise -> drop (deterministic, no LLM — junk is never embedded)
        noise = classify_noise(title, text, self.filters)
        if noise.is_noise:
            self._set_item_status(item_id, "filtered_out")
            self._journal("item", item_id, "filter", "noise", ",".join(noise.reasons))
            return VerifyResult("filtered_out", reason="noise:" + ",".join(noise.reasons))

        # 1b. transient air-situation drone alert ("БпЛА над містом", "курсом на…",
        #     "в укриття") -> drop. High-volume real-time monitoring, not news; a strike
        #     WITH a consequence or a nightly summary is excluded and kept (see signal.py).
        if is_air_alert(title, text, self.filters):
            self._set_item_status(item_id, "filtered_out")
            self._journal("item", item_id, "filter", "air_alert", "transient_drone_alert")
            return VerifyResult("filtered_out", reason="air_alert")

        # 2. embed + persist + cluster into an event (cheap; runs before the LLM so
        #    reprints collapse into one event before we pay to classify)
        vec = self.embedder.embed(f"{title or ''}\n{text or ''}")
        self._store_embedding(item_id, vec)
        assign = self.clusterer.assign(item_id, vec)
        self._set_item_status(item_id, "clustered")

        # 3. classify the event ONCE (LLM in prod). Items that join an already-
        #    classified event reuse the cached classification and cost nothing.
        cls, newly_classified = self._classify_event(assign.event_id, title, text)
        if not cls.is_event:
            # the whole cluster is not a news event: drop its items, retire the event
            self._retire_non_event(assign.event_id)
            self._journal("item", item_id, "filter", "not_event", "")
            return VerifyResult("filtered_out", reason="not_event")

        # 4. event evidence + risk gate (re-runs per item so corroboration updates)
        event_status, level, indep = self._verify_event(assign.event_id, cls)

        markers = ipso_markers(title, text, self.filters)
        violations = stoplist_check(title, text, self.stoplist_rules, side=cls.side)
        self._journal(
            "item", item_id, "verify", event_status,
            reason=f"level={level} independent_sources={indep}",
            details={
                "event_id": assign.event_id,
                "created_new_event": assign.created_new,
                "newly_classified": newly_classified,
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
    def _classify_event(self, event_id: int, title: str | None, text: str | None) -> tuple[Classification, bool]:
        """Classify an event once (LLM) and cache the result on the event row.

        Returns (classification, newly_classified). If the event was already
        classified — a reprint joining it — the cached fields are rebuilt into a
        Classification with no LLM call (this is the whole cost saving). A retired
        non-event (status filtered_out) reports is_event=False from the cache.
        """
        from newsroom.models import Event

        with self.sf() as s:
            event = s.get(Event, event_id)
            if event is not None and event.classifier_model:
                if event.status == "filtered_out":
                    return Classification(is_event=False), False
                return Classification(
                    is_event=True,
                    rubrics=[event.rubric] if event.rubric else [],
                    side=event.side or "unknown",
                    is_first_source=bool(event.is_first_source),
                    is_rumor=bool(event.is_rumor),
                    keywords=list(event.keywords or []),
                ), False

        cls = self.classifier.classify(title, text)

        with self.sf() as s:
            event = s.get(Event, event_id)
            if event is not None:
                event.side = cls.side
                event.is_first_source = cls.is_first_source
                event.is_rumor = cls.is_rumor
                event.keywords = list(cls.keywords) or None
                event.classifier_model = getattr(self.classifier, "model", "unknown")
                s.commit()
        self._journal(
            "event", event_id, "classify", "event" if cls.is_event else "not_event",
            reason=f"rubrics={','.join(cls.rubrics)} side={cls.side}",
            details={"is_first_source": cls.is_first_source, "is_rumor": cls.is_rumor},
            model=getattr(self.classifier, "model", None),
        )
        return cls, True

    # ------------------------------------------------------------------
    def _retire_non_event(self, event_id: int) -> None:
        """The cluster classified as non-news: drop its items and take the event out
        of the clustering pool (centroid=None) so it attracts nothing further. Rows
        are kept, not deleted — the analytical layer is recomputable (architecture §3)."""
        from sqlalchemy import select

        from newsroom.models import Event, EventItem, Item

        with self.sf() as s:
            event = s.get(Event, event_id)
            if event is not None:
                event.status = "filtered_out"
                event.centroid = None
            item_ids = list(s.execute(
                select(EventItem.item_id).where(EventItem.event_id == event_id)
            ).scalars().all())
            for iid in item_ids:
                it = s.get(Item, iid)
                if it is not None:
                    it.status = "filtered_out"
            s.commit()

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
                require_official_for_critical=self.require_official_for_critical,
            )

            event = s.get(Event, event_id)
            event.rubric = cls.rubrics[0] if cls.rubrics else None
            event.risk_level = level
            event.status = gate.status
            event.independent_source_count = indep
            if not (event.title and event.title.strip()):
                # Telegram posts often have no title (only a caption), so the event
                # would stay title-less — breaking dedup/digest (which key off the
                # title) and curation/log readability. Derive one from the items.
                event.title = _derive_event_title(rows)
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


def _derive_event_title(rows, *, max_len: int = 200) -> str | None:
    """A human-readable event title from its items: the first non-empty item title,
    else the first line/snippet of the first non-empty item text (Telegram captions).
    `rows` is a list of (Item, Source). Pure — offline-tested."""
    for it, _src in rows:
        title = (getattr(it, "title", None) or "").strip()
        if title:
            return title[:max_len]
    for it, _src in rows:
        text = (getattr(it, "text", None) or "").strip()
        if text:
            snippet = text.splitlines()[0].strip()
            return (snippet or text)[:max_len]
    return None
