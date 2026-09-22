"""Fact-check orchestration (architecture §9, steps 1 -> 2 -> 6).

Ties the atomic pieces together for one event: extract claims from the event's
material, gather evidence for each claim (own corpus for now; official registry
and refutation DBs plug into the same EvidenceSearcher later), then judge a
verdict from that evidence. Extraction, search and judging are pluggable (LLM /
searcher in prod, fakes in tests); persistence and idempotency live here.

Idempotency mirrors the editorial pipeline: an event that already has claims is
skipped, so a re-run does not duplicate work. Nothing here publishes.
"""
from __future__ import annotations

from dataclasses import dataclass

from newsroom.factcheck.claims import ClaimExtractor, store_claims
from newsroom.factcheck.evidence import EvidenceRef, EvidenceSearcher, store_evidence
from newsroom.factcheck.verdict import VerdictJudge, apply_verdict


@dataclass(frozen=True)
class FactCheckResult:
    event_id: int
    extracted: int = 0
    checked: int = 0
    skipped: bool = False


def _combined_text(title: str | None, item_texts: list[str]) -> str:
    return "\n".join([title or "", *item_texts]).strip()


def _evidence_snippet(ref: EvidenceRef, item_text: str | None) -> str:
    if ref.item_id is not None and item_text:
        return item_text.strip()[:600]
    return (ref.external_ref or "").strip()


class FactChecker:
    def __init__(self, session_factory, *, extractor: ClaimExtractor, searcher: EvidenceSearcher,
                 judge: VerdictJudge, max_claims: int = 12, max_evidence: int = 5):
        self.sf = session_factory
        self.extractor = extractor
        self.searcher = searcher
        self.judge = judge
        self.max_claims = max_claims
        self.max_evidence = max_evidence

    def check_event(self, event_id: int) -> FactCheckResult:
        from sqlalchemy import func, select

        from newsroom.models import Claim, Event, EventItem, Item

        # 1. load the event's material; skip if already fact-checked
        with self.sf() as s:
            event = s.get(Event, event_id)
            if event is None:
                return FactCheckResult(event_id, skipped=True)
            if s.scalar(select(func.count()).select_from(Claim).where(Claim.event_id == event_id)):
                return FactCheckResult(event_id, skipped=True)
            rows = list(s.execute(
                select(Item.id, Item.title, Item.text)
                .join(EventItem, EventItem.item_id == Item.id)
                .where(EventItem.event_id == event_id)
            ).all())
            title = event.title
        own_item_ids = [r[0] for r in rows]
        item_texts = [f"{r[1] or ''}\n{r[2] or ''}".strip() for r in rows]

        # 2. extract atomic claims
        from newsroom.llmutil import llm_context

        with llm_context(event_id=event_id, stage="factcheck_claims"):
            claims = self.extractor.extract(title, _combined_text(title, item_texts))[: self.max_claims]
        if not claims:
            return FactCheckResult(event_id, extracted=0, checked=0)

        with self.sf() as s:
            claim_ids = store_claims(s, event_id, claims)
            s.commit()

        # 3. evidence + verdict per claim
        checked = 0
        for claim, claim_id in zip(claims, claim_ids):
            refs = self.searcher.search(claim.text, exclude_item_ids=own_item_ids)[: self.max_evidence]
            with self.sf() as s:
                ev_ids = store_evidence(s, claim_id, refs)
                texts = self._evidence_item_texts(s, refs)
                s.commit()
            snippets = [_evidence_snippet(ref, texts.get(ref.item_id)) for ref in refs]
            with llm_context(event_id=event_id, claim_id=claim_id,
                             stage="factcheck_verdict"):
                result = self.judge.judge(claim.text, snippets)
            with self.sf() as s:
                apply_verdict(s, claim_id, ev_ids, result)
                s.commit()
            checked += 1

        return FactCheckResult(event_id, extracted=len(claim_ids), checked=checked)

    @staticmethod
    def _evidence_item_texts(session, refs: list[EvidenceRef]) -> dict[int, str]:
        from sqlalchemy import select

        from newsroom.models import Item

        ids = [r.item_id for r in refs if r.item_id is not None]
        if not ids:
            return {}
        rows = session.execute(
            select(Item.id, Item.title, Item.text).where(Item.id.in_(ids))
        ).all()
        return {iid: f"{title or ''}\n{text or ''}".strip() for iid, title, text in rows}


# statuses worth fact-checking: the same events that head toward a draft, plus rumors.
CHECKABLE_STATUSES = ("reported", "confirmed", "rumor")


def check_pending(session_factory, checker: "FactChecker", *, limit: int = 25,
                  risk_levels: tuple[str, ...] | None = None,
                  require_dedup_settled: bool = False,
                  dedup_grace_seconds: float = 300.0) -> dict[str, int]:
    """One fact-check tick: check publishable events that have no claims yet.
    Once checked, the event has claims and is skipped next tick. Fact-checking (claims +
    a verdict per claim) is the priciest step.
    When `risk_levels` is given, only events at those risk levels are checked (plus
    unknown-risk events, checked conservatively): low-risk topics — economy, tech,
    sport, culture — need no deep verdicts, which cuts fact-check cost further."""
    import datetime as dt

    from sqlalchemy import or_, select

    from newsroom.analyze.ingest_dedup import dedup_settled_clause
    from newsroom.models import Claim, Event

    now = dt.datetime.now(dt.timezone.utc)
    have_claims = select(Claim.event_id).distinct()
    # a duplicate event (ingest/publish dedup) is never posted — don't fact-check it
    conditions = [Event.status.in_(CHECKABLE_STATUSES), Event.id.not_in(have_claims),
                  Event.duplicate_of.is_(None)]
    dedup_clause = dedup_settled_clause(require_dedup_settled,
                                        now - dt.timedelta(seconds=dedup_grace_seconds))
    if dedup_clause is not None:
        conditions.append(dedup_clause)
    if risk_levels:
        # unknown risk (NULL) is checked too — when the level is uncertain, treat it
        # as check-worthy (asymmetry of errors, §3.5)
        conditions.append(or_(Event.risk_level.in_(risk_levels), Event.risk_level.is_(None)))

    with session_factory() as s:
        ids = list(s.execute(
            select(Event.id).where(*conditions).order_by(Event.id).limit(limit)
        ).scalars().all())

    stats = {"events": 0, "claims": 0, "checked": 0}
    for event_id in ids:
        result = checker.check_event(event_id)
        if result.skipped:
            continue
        stats["events"] += 1
        stats["claims"] += result.extracted
        stats["checked"] += result.checked
    return stats
