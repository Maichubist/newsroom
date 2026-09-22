"""Ingest-time deduplication (Phase B carcass) — merge duplicate events EARLY.

Phase A stops a duplicate at publish time (after it already spent a fact base, a
fact-check and a draft). Phase B moves the same decision UPSTREAM: right after
clustering, a freshly-formed event is compared against other recent events and, if
it is the same story, MERGED into the earliest one (its items are reassigned, so the
canonical event gains corroboration) instead of living on as a separate event. A
merged duplicate is inert everywhere downstream (it carries `duplicate_of`, which
clustering, story-linking, fact base, fact-check, curation and editorial all skip),
so nothing is spent on it.

This reuses Phase A's signal machinery verbatim (`predup.candidate_signals` /
`classify_signals` / the pairwise `TwinJudge`): only an exact content_hash merges
without the LLM; anything else in the candidate net (forwarded_from, URL, SimHash,
cosine) asks the arbiter, and ONLY a "duplicate" verdict merges (an "update"/"separate"
stays a distinct event for story-linking to relate).

Carcass status: shipped OFF (INGEST_DEDUP_ENABLED) and observe-first
(INGEST_DEDUP_ENFORCE). The full item-first rewrite and the per-item fact-fingerprint
wait until Phase A observe logs show which signals actually drive false pos/neg.
Thresholds are shared with / mirror Phase A and are calibration-pending.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from newsroom.publishers.predup import (
    ACTION_HOLD_REVIEW,
    ACTION_DUPLICATE,
    CLASS_AUTO_DUPLICATE,
    CandidateSignals,
    EventSig,
    TwinJudge,
    TwinPair,
    candidate_signals,
    classify_signals,
    in_candidate_net,
    load_event_sig,
    load_event_sigs,
)

log = logging.getLogger("newsroom.analyze.ingest_dedup")

MERGE_SEPARATE = "separate"
MERGE_DUPLICATE = "duplicate"
MERGE_UNRESOLVED = "unresolved"

# decisions.decision values written per verdict (stage = "ingest_dedup")
_DECISION_DUPLICATE = "ingest_duplicate"
_DECISION_SEPARATE = "ingest_separate"
_DECISION_UNRESOLVED = "ingest_unresolved"
# a verdict is "settled" only when it actually resolved — an LLM outage writes
# `ingest_unresolved`, which must NOT count as done (it is retried, not skipped forever).
_RESOLVED_DECISIONS = (_DECISION_DUPLICATE, _DECISION_SEPARATE)
DEFAULT_MAX_UNRESOLVED_ATTEMPTS = 3

# events young enough to still be merging (mirrors the clustering window)
_MERGEABLE_STATUSES = ("signal", "rumor", "reported", "confirmed")


class MergeConfigError(ValueError):
    """Raised when the merge section of dedup.yaml is malformed."""


@dataclass(frozen=True)
class MergeConfig:
    window_hours: int = 48               # how far back to look for the same event (≈ clustering window)
    vector_candidate: float = 0.55       # centroid cosine that makes an event a candidate (a net, not a verdict)
    simhash_max: int = 6                 # near-duplicate SimHash Hamming bound
    top_candidates: int = 3              # at most this many nearest candidates go to the arbiter


def load_merge_config(path: str | Path) -> MergeConfig:
    path = Path(path)
    if not path.exists():
        raise MergeConfigError(f"dedup config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = data.get("merge", {}) if isinstance(data, dict) else {}
    try:
        config = MergeConfig(
            window_hours=int(section.get("window_hours", 48)),
            vector_candidate=float(section.get("vector_candidate", 0.55)),
            simhash_max=int(section.get("simhash_max", 6)),
            top_candidates=int(section.get("top_candidates", 3)),
        )
        _validate_merge_config(config)
        return config
    except (TypeError, ValueError) as exc:
        raise MergeConfigError(f"bad merge value: {exc}") from exc


def _validate_merge_config(config: MergeConfig) -> None:
    if config.window_hours < 1:
        raise MergeConfigError("window_hours must be >= 1")
    if not -1.0 <= config.vector_candidate <= 1.0:
        raise MergeConfigError("vector_candidate must be between -1 and 1")
    if not 0 <= config.simhash_max <= 64:
        raise MergeConfigError("simhash_max must be between 0 and 64")
    if config.top_candidates < 1:
        raise MergeConfigError("top_candidates must be >= 1")


@dataclass(frozen=True)
class MergeVerdict:
    action: str = MERGE_SEPARATE                 # separate | duplicate
    mode: str = "none"                           # none | auto | llm
    canonical_event_id: int | None = None         # the EARLIER event this one duplicates
    confidence: float | None = None
    reason: str = ""
    model: str | None = None
    signals: dict = field(default_factory=dict)
    pair_checks: tuple[dict, ...] = ()


@dataclass(frozen=True)
class DedupOneResult:
    event_id: int
    verdict: MergeVerdict
    merged: bool = False
    items_moved: int = 0


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def dedup_settled_clause(enabled: bool, cutoff):
    """SQLAlchemy condition that gates an expensive LLM stage on the ingest-dedup verdict
    being IN — so a duplicate is merged (→ `duplicate_of`, which those stages skip) before
    they spend tokens on it. An event qualifies when it has an ingest_dedup decision, OR it
    has waited out the grace (so nothing stalls if ingest dedup is off or momentarily behind).
    Returns None when `enabled` is False (the default) — the caller then gates on nothing, so
    the whole mechanism is a no-op unless INGEST_DEDUP_ENABLED is on."""
    if not enabled:
        return None
    from sqlalchemy import Integer, cast, or_, select

    from newsroom.models import Decision, Event

    # only a RESOLVED verdict counts as settled — an unresolved (LLM outage) event waits for
    # the grace so its retries have a chance, then proceeds rather than stalling forever.
    settled = select(cast(Decision.entity_id, Integer)).where(
        Decision.entity_type == "event", Decision.stage == "ingest_dedup",
        Decision.decision.in_(_RESOLVED_DECISIONS))
    return or_(Event.id.in_(settled), Event.first_seen_at < cutoff)


def find_merge_candidates(session, event_id: int, *, config: MergeConfig):
    """(incoming_sig, [(candidate_sig, signals), ...]) — EARLIER recent, non-duplicate events
    in the candidate net, earliest first. An event only ever merges INTO an earlier one, so
    the earliest stays canonical and a pair is never processed from both ends."""
    from sqlalchemy import select

    from newsroom.models import Event

    incoming = load_event_sig(session, event_id)
    if incoming is None:
        return None, []

    cutoff = _utc_now() - dt.timedelta(hours=config.window_hours)
    cand_ids = list(session.execute(
        select(Event.id).where(
            Event.id < event_id,                    # only earlier events (canonical survives)
            Event.duplicate_of.is_(None),
            Event.status.in_(_MERGEABLE_STATUSES),
            Event.first_seen_at >= cutoff,
        )
    ).scalars().all())

    scored: list[tuple[EventSig, CandidateSignals]] = []
    sigs = load_event_sigs(session, cand_ids)          # 2 queries for all candidates, not 1 per
    for cid in cand_ids:
        csig = sigs.get(cid)
        if csig is None:
            continue
        sig = candidate_signals(incoming, csig)
        if in_candidate_net(sig, config):
            scored.append((csig, sig))

    scored.sort(key=lambda t: (
        0 if (t[1].exact_content_hash or t[1].forwarded_from_match) else 1,
        t[0].event_id,                                  # earliest event first (canonical survivor)
    ))
    return incoming, scored[:config.top_candidates]


class IngestDedup:
    """Early event merge check. `check(event_id)` returns a MergeVerdict; it NEVER
    mutates state (the tick applies it, and only when enforcing)."""

    def __init__(self, session_factory, *, judge: "TwinJudge | None" = None,
                 config: MergeConfig | None = None):
        self.sf = session_factory
        self.judge = judge
        self.config = config or MergeConfig()

    def check(self, event_id: int | None) -> MergeVerdict:
        if event_id is None:
            return MergeVerdict()
        with self.sf() as s:
            incoming, candidates = find_merge_candidates(s, event_id, config=self.config)
            if incoming is None or not candidates:
                return MergeVerdict()
            checks: list[dict] = []

            for csig, sig in candidates:
                if classify_signals(sig, self.config) == CLASS_AUTO_DUPLICATE:
                    checks.append({"candidate_event_id": csig.event_id,
                                   "decision": MERGE_DUPLICATE, "mode": "auto",
                                   "confidence": 1.0, "reason": "exact content_hash",
                                   **sig.as_details()})
                    return MergeVerdict(action=MERGE_DUPLICATE, mode="auto",
                                        canonical_event_id=csig.event_id,
                                        reason="exact content_hash",
                                        signals=sig.as_details(), pair_checks=tuple(checks))

            if self.judge is None:
                checks.extend({"candidate_event_id": csig.event_id, "decision": "unjudged",
                               "mode": "unavailable", **sig.as_details()}
                              for csig, sig in candidates)
                return MergeVerdict(action=MERGE_UNRESOLVED, mode="unavailable",
                                    reason="dedup_judge_unavailable", pair_checks=tuple(checks))

            for csig, sig in candidates:
                pair = self._build_pair(s, incoming, csig)
                judgment = self.judge.judge(pair)
                checks.append({"candidate_event_id": csig.event_id,
                               "decision": judgment.decision, "mode": "llm",
                               "confidence": judgment.confidence, "reason": judgment.reason,
                               **sig.as_details()})
                if judgment.decision == ACTION_HOLD_REVIEW:
                    return MergeVerdict(action=MERGE_UNRESOLVED, mode="llm",
                                        canonical_event_id=csig.event_id,
                                        confidence=judgment.confidence, reason=judgment.reason,
                                        model=getattr(self.judge, "model", None),
                                        signals=sig.as_details(), pair_checks=tuple(checks))
                # only a straight duplicate merges; update/separate stay distinct events
                # (story-linking relates an update, it must not be swallowed here)
                if judgment.decision != ACTION_DUPLICATE:
                    continue
                return MergeVerdict(action=MERGE_DUPLICATE, mode="llm",
                                    canonical_event_id=csig.event_id,
                                    confidence=judgment.confidence, reason=judgment.reason,
                                    model=getattr(self.judge, "model", None),
                                    signals=sig.as_details(), pair_checks=tuple(checks))
        return MergeVerdict(reason="all_candidates_separate",
                            model=getattr(self.judge, "model", None), pair_checks=tuple(checks))

    def _build_pair(self, s, incoming: EventSig, candidate: EventSig) -> TwinPair:
        from sqlalchemy import select

        from newsroom.models import Event, EventItem, Item

        inc_ev = s.get(Event, incoming.event_id)
        cand_ev = s.get(Event, candidate.event_id)
        inc_texts = list(s.execute(
            select(Item.text).join(EventItem, EventItem.item_id == Item.id)
            .where(EventItem.event_id == incoming.event_id).limit(2)
        ).scalars().all())
        return TwinPair(
            incoming_event_id=incoming.event_id,
            incoming_title=(inc_ev.title if inc_ev else "") or "",
            incoming_texts=tuple(t for t in inc_texts if t),
            incoming_facts=tuple(_event_facts(inc_ev)),
            incoming_sources=int(inc_ev.independent_source_count or 0) if inc_ev else 0,
            candidate_event_id=candidate.event_id,
            candidate_title=(cand_ev.title if cand_ev else "") or "",
            candidate_facts=tuple(_event_facts(cand_ev)),
            candidate_published=False,
        )


def merge_events(session, *, keep: int, drop: int) -> int:
    """Fold event `drop` into `keep`: reassign its EventItems to `keep` (so the canonical
    event gains the extra sources → corroboration, §8) and mark `drop` a duplicate of
    `keep` so it is inert everywhere downstream. Returns the number of items moved.
    Idempotent — a `drop` already marked is left alone. Caller commits."""
    from sqlalchemy import select, update

    from newsroom.models import Event, EventItem

    drop_ev = session.get(Event, drop)
    keep_ev = session.get(Event, keep)
    if drop_ev is None or keep_ev is None or drop == keep or drop_ev.duplicate_of is not None:
        return 0

    existing = set(session.execute(
        select(EventItem.item_id).where(EventItem.event_id == keep)
    ).scalars().all())
    moved = 0
    for ei in session.execute(select(EventItem).where(EventItem.event_id == drop)).scalars().all():
        if ei.item_id in existing:
            session.delete(ei)          # keep already has this item — avoid a duplicate link
        else:
            ei.event_id = keep
            ei.role = ei.role or "copy"
            moved += 1
    drop_ev.duplicate_of = keep
    _refresh_merged_event_evidence(session, keep_ev)
    keep_ev.updated_at = _utc_now()
    return moved


def _refresh_merged_event_evidence(session, event) -> None:
    """Refresh derived evidence after moving sources into the canonical event.

    Verification ran before the two events were merged. Its prior source count and
    fact base are therefore stale once corroborating material is reassigned.
    """
    from sqlalchemy import select

    from newsroom.analyze.independence import SourceItem, independent_source_count
    from newsroom.analyze.risk import LEVELS, decide
    from newsroom.models import EventItem, Item, Source

    rows = session.execute(
        select(Item, Source)
        .join(EventItem, EventItem.item_id == Item.id)
        .join(Source, Source.id == Item.source_id)
        .where(EventItem.event_id == event.id)
    ).all()
    source_items = [
        SourceItem(source_id=item.source_id, content_hash=item.content_hash,
                   simhash=item.simhash, forwarded_from=item.forwarded_from,
                   source_name=source.name)
        for item, source in rows
    ]
    event.independent_source_count = independent_source_count(source_items)
    event.fact_base = None

    if event.risk_level in LEVELS:
        has_official = any(source.is_official or source.tier == "official" for _, source in rows)
        high_rep = any(source.tier in ("official", "media") for _, source in rows)
        gate = decide(event.risk_level, independent_sources=event.independent_source_count,
                      has_official=has_official, has_first_source=bool(event.is_first_source),
                      high_reputation=high_rep, is_rumor=bool(event.is_rumor))
        event.status = gate.status


# --- official-first canonical choice (docs: official-first sourcing) -------------------
# tier → "officialness" rank (lower = more authoritative); is_official also ⇒ 0.
_TIER_RANK = {"official": 0, "media": 1, "aggregator": 2, "leak": 3, "anonymous": 4}
_DEFAULT_TIER_RANK = 5


def _tier_rank(tier: str | None, is_official: bool = False) -> int:
    if is_official:
        return 0
    return _TIER_RANK.get((tier or "").strip().lower(), _DEFAULT_TIER_RANK)


def event_tier_rank(session, event_id: int) -> int:
    """Most-official rank among an event's sources (lower = more official). An event
    inherits its best source's rank, so an official-sourced event outranks an aggregator
    one. Source-less/unknown events get the least-official rank (they never win a tie)."""
    from sqlalchemy import select

    from newsroom.models import EventItem, Item, Source

    rows = session.execute(
        select(Source.tier, Source.is_official)
        .join(Item, Item.source_id == Source.id)
        .join(EventItem, EventItem.item_id == Item.id)
        .where(EventItem.event_id == event_id)
    ).all()
    if not rows:
        return _DEFAULT_TIER_RANK
    return min(_tier_rank(tier, bool(is_official)) for tier, is_official in rows)


def choose_canonical(incoming_id: int, incoming_rank: int,
                     candidate_id: int, candidate_rank: int) -> tuple[int, int]:
    """(keep, drop) for a confirmed duplicate pair: keep the MORE-official event; on a tie
    keep the earlier one. `candidate` is always earlier than `incoming`
    (find_merge_candidates guarantees candidate_id < incoming_id), so the earlier event wins
    every tie — official-first only ever FLIPS the canonical to a later, more-official one."""
    if incoming_rank < candidate_rank:
        return incoming_id, candidate_id          # later event is more official → it is canonical
    return candidate_id, incoming_id              # earlier canonical wins ties and when >= official


def dedup_one_event(session_factory, dedup: "IngestDedup", event_id: int, *,
                    enforce: bool = False, canonical_by_tier: bool = False,
                    charter_version: str = "0.3") -> DedupOneResult:
    """Resolve and journal one newly-created event before its first classifier call.

    The returned event_id is the surviving canonical event when an enforced duplicate
    merge occurred; otherwise it is the original event. This is also used by the
    background sweep, keeping online and catch-up semantics identical.
    """
    from newsroom.models import Decision

    try:
        verdict = dedup.check(event_id)
    except Exception as exc:  # noqa: BLE001 — telemetry + later stages must survive one bad pair
        log.warning("ingest dedup check failed", extra={"event_id": event_id, "error": str(exc)})
        verdict = MergeVerdict(action=MERGE_UNRESOLVED, mode="error", reason=str(exc)[:200])

    details = {
        "action": verdict.action, "mode": verdict.mode, "enforced": enforce,
        "candidate_event_id": verdict.canonical_event_id,
        "confidence": verdict.confidence, **(verdict.signals or {}),
    }
    keep = drop = None
    moved = 0
    with session_factory() as s:
        for pair in verdict.pair_checks:
            candidate_id = pair.get("candidate_event_id")
            s.add(Decision(
                entity_type="event", entity_id=str(event_id), stage="ingest_dedup_pair",
                decision=f"pair_{pair.get('decision', 'unknown')}",
                reason=(str(pair.get("reason"))[:300] if pair.get("reason") else None),
                details={**pair, "enforced": enforce}, score=pair.get("confidence"),
                ref_id=candidate_id, model=verdict.model, charter_version=charter_version,
            ))
        if verdict.action == MERGE_DUPLICATE and verdict.canonical_event_id is not None:
            keep, drop = verdict.canonical_event_id, event_id
            tier_keep, tier_drop = choose_canonical(
                event_id, event_tier_rank(s, event_id),
                verdict.canonical_event_id, event_tier_rank(s, verdict.canonical_event_id))
            details["keep_by_tier"] = tier_keep
            details["tier_flip"] = tier_keep != keep
            if canonical_by_tier:
                keep, drop = tier_keep, tier_drop
        s.add(Decision(
            entity_type="event", entity_id=str(event_id), stage="ingest_dedup",
            decision=f"ingest_{verdict.action}", reason=verdict.reason or None,
            details=details, model=verdict.model, charter_version=charter_version,
            score=verdict.confidence, ref_id=verdict.canonical_event_id,
        ))
        if enforce and keep is not None:
            moved = merge_events(s, keep=keep, drop=drop)
        s.commit()
    return DedupOneResult(
        event_id=(keep if enforce and keep is not None else event_id),
        verdict=verdict,
        merged=bool(enforce and keep is not None),
        items_moved=moved,
    )


def dedup_new_events(session_factory, dedup: "IngestDedup", *, enforce: bool = False,
                     canonical_by_tier: bool = False,
                     limit: int = 40, charter_version: str = "0.3",
                     max_unresolved_attempts: int = DEFAULT_MAX_UNRESOLVED_ATTEMPTS) -> dict[str, int]:
    """One ingest-dedup tick: check recent events for an earlier twin and (when enforcing)
    merge the later one into the earlier. Observe mode only journals the verdict (stage
    'ingest_dedup'), changing no state.

    An event is settled once it gets a RESOLVED verdict (duplicate/separate) and is not
    re-checked. An `unresolved` verdict (LLM outage) does NOT settle it — it is retried on
    later ticks until `max_unresolved_attempts`, after which it is given up (the downstream
    grace then lets it proceed). This is what stops an LLM outage from silently marking a
    whole batch as 'separate' forever."""
    from sqlalchemy import Integer, cast, func, select

    from newsroom.models import Decision, Event

    cutoff = _utc_now() - dt.timedelta(hours=dedup.config.window_hours)
    with session_factory() as s:
        resolved = (
            select(cast(Decision.entity_id, Integer))
            .where(Decision.entity_type == "event", Decision.stage == "ingest_dedup",
                   Decision.decision.in_(_RESOLVED_DECISIONS))
        )
        # events that already exhausted their unresolved retries — give up, don't spin
        unresolved_counts = (
            select(cast(Decision.entity_id, Integer).label("eid"), func.count().label("n"))
            .where(Decision.entity_type == "event", Decision.stage == "ingest_dedup",
                   Decision.decision == _DECISION_UNRESOLVED)
            .group_by(cast(Decision.entity_id, Integer))
        ).subquery()
        maxed_out = select(unresolved_counts.c.eid).where(
            unresolved_counts.c.n >= max_unresolved_attempts)

        ids = list(s.execute(
            select(Event.id).where(
                Event.status.in_(_MERGEABLE_STATUSES),
                Event.duplicate_of.is_(None),
                Event.first_seen_at >= cutoff,
                Event.id.not_in(resolved),
                Event.id.not_in(maxed_out),
            ).order_by(Event.id).limit(limit)
        ).scalars().all())

    stats = {"checked": 0, "pairs": 0, "merged": 0, "items_moved": 0, "unresolved": 0}
    for event_id in ids:
        result = dedup_one_event(
            session_factory, dedup, event_id, enforce=enforce,
            canonical_by_tier=canonical_by_tier, charter_version=charter_version)
        verdict = result.verdict
        stats["checked"] += 1
        stats["pairs"] += len(verdict.pair_checks)
        if verdict.action == MERGE_UNRESOLVED:
            stats["unresolved"] += 1
        if result.merged:
            stats["merged"] += 1
            stats["items_moved"] += result.items_moved
    return stats


def _facts(fact_base, *, limit: int = 5) -> list[str]:
    if not isinstance(fact_base, dict):
        return []
    rows = [f for f in (fact_base.get("facts") or []) if isinstance(f, dict) and f.get("text")]
    rows.sort(key=lambda f: int(f.get("confirmed_by") or 0), reverse=True)
    return [str(f["text"]).strip() for f in rows[:limit]]


def _event_facts(event, *, limit: int = 5) -> list[str]:
    if event is None:
        return []
    return (_facts(event.fact_base, limit=limit)
            or _facts({"facts": list(event.compact_facts or [])}, limit=limit))
