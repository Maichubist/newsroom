"""Publish-time deduplication — the last-second "did we just post this?" backstop.

Even a perfect ingest-time dedup can't stop a twin that arrived ten minutes later,
so right before a draft is sent we compare its event against everything drafted or
published in the last 72h and decide identity:

  * cheap signals first (free, no LLM): an exact content_hash → an unconditional
    duplicate; Telegram forwarded_from (origin channel name only, not a message id),
    a matching URL, close SimHash or high centroid cosine only make it a *candidate*;
  * grey zone (a candidate but not exact) → one pairwise LLM call, incoming event vs
    the nearest candidate, answering ONLY the identity question
    (duplicate | update | separate) — NOT the update_type (that stays StoryUpdater's job).

The decision is one of:
  * separate     → publish normally;
  * duplicate    → don't publish; the caller supersedes the draft + sets duplicate_of;
  * update       → don't publish as a primary post; the caller links the event to the
                   canonical story and hands it back to StoryUpdater;
  * hold_review  → LLM unavailable/invalid → hold for a human (conservative, §3.5).

Split for testability: `classify_signals` and the judge parser are pure/offline;
`find_publish_candidates` is the DB search (pg-tested); `LLMTwinJudge` is pluggable
(a fake in tests). This module NEVER mutates state — it returns a verdict; the
publisher applies it (and only when enforcing, never in observe mode).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

from newsroom.analyze.clustering import cosine
from newsroom.collectors.base import hamming_distance
from newsroom.promptutil import fill_prompt

log = logging.getLogger("newsroom.publishers.predup")

# decision vocabulary --------------------------------------------------------
ACTION_SEPARATE = "separate"
ACTION_DUPLICATE = "duplicate"
ACTION_UPDATE = "update"
ACTION_HOLD_REVIEW = "hold_review"

# deterministic classification of a candidate's signals
CLASS_AUTO_DUPLICATE = "auto_duplicate"   # exact content_hash / forwarded_from
CLASS_GREY = "grey"                       # a candidate, needs the LLM arbiter
CLASS_SEPARATE = "separate"               # not close enough to be a candidate

_CANDIDATE_STATUSES = ("published", "draft")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

class DedupConfigError(ValueError):
    """Raised when dedup.yaml is malformed."""


@dataclass(frozen=True)
class PredupConfig:
    window_hours: int = 72               # how far back to look for a twin
    vector_candidate: float = 0.55       # centroid cosine that makes an event a candidate (LOW — a net, not a verdict)
    simhash_max: int = 6                 # Hamming distance under which two texts are near-duplicate
    top_candidates: int = 3              # at most this many candidates go to the LLM arbiter


def load_predup_config(path: str | Path) -> PredupConfig:
    path = Path(path)
    if not path.exists():
        raise DedupConfigError(f"dedup config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = data.get("predup", data) if isinstance(data, dict) else {}
    try:
        config = PredupConfig(
            window_hours=int(section.get("window_hours", 72)),
            vector_candidate=float(section.get("vector_candidate", 0.55)),
            simhash_max=int(section.get("simhash_max", 6)),
            top_candidates=int(section.get("top_candidates", 3)),
        )
        _validate_config(config)
        return config
    except (TypeError, ValueError) as exc:
        raise DedupConfigError(f"bad predup value: {exc}") from exc


def _validate_config(config: PredupConfig) -> None:
    if config.window_hours < 1:
        raise DedupConfigError("window_hours must be >= 1")
    if not -1.0 <= config.vector_candidate <= 1.0:
        raise DedupConfigError("vector_candidate must be between -1 and 1")
    if not 0 <= config.simhash_max <= 64:
        raise DedupConfigError("simhash_max must be between 0 and 64")
    if config.top_candidates < 1:
        raise DedupConfigError("top_candidates must be >= 1")


# --------------------------------------------------------------------------- #
# signals (pure)
# --------------------------------------------------------------------------- #

# --- fact-fingerprint: typed distinctive numbers, a high-precision dedup booster ----
# Measured: two events sharing >=2 typed number-facts are almost always the SAME event
# even when their embeddings sit below the story-link threshold (cross-source
# fragmentation — cosine 0.29-0.59). Used ONLY to widen the candidate net; the LLM still
# decides, so numbers stay a booster, never a sole auto-merge.
_NUM_MONEY = re.compile(r"(?:€\s*)?(\d+(?:[.,]\d+)?)\s*(?:млрд|мільярд\w*)", re.I)
_NUM_AREA = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:км²|км2|кв\.?\s*км|квадратн\w*\s+кілометр\w*)", re.I)
_NUM_TROOPS = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(тис\w*\s+)?(?:солдат\w*|військовослужбовц\w*|осіб|загарбник\w*)", re.I)
_NUM_DRONES = re.compile(r"(\d+(?:[.,]\d+)?)\s*(тис\w*\s+)?(?:дрон\w*|безпілотник\w*|бпла|шахед\w*)", re.I)


def _num_token(raw: str, thousands: bool = False) -> str:
    try:
        v = float(raw.replace(" ", "").replace(",", "."))
    except ValueError:
        return raw
    return f"{v * 1000 if thousands else v:g}"


def extract_number_facts(text: str | None) -> frozenset[str]:
    """Typed distinctive numbers in the text (money-bn / area / troops / drones), normalized
    so '1,5 тисячі солдатів' and '1500 солдатів' collapse to one token. Pure/offline."""
    t = text or ""
    fp: set[str] = set()
    for m in _NUM_MONEY.finditer(t):
        fp.add("M" + _num_token(m.group(1)))
    for m in _NUM_AREA.finditer(t):
        fp.add("A" + _num_token(m.group(1)))
    for m in _NUM_TROOPS.finditer(t):
        fp.add("T" + _num_token(m.group(1), bool(m.group(2))))
    for m in _NUM_DRONES.finditer(t):
        fp.add("D" + _num_token(m.group(1), bool(m.group(2))))
    return frozenset(fp)


@dataclass(frozen=True)
class ItemSig:
    content_hash: str | None = None
    simhash: int | None = None
    forwarded_from: str | None = None
    url: str | None = None
    # Every blank caption/title has the same SHA-256 and zero SimHash, so neither
    # is an identity signal for media-only posts.
    text_eligible: bool = True


@dataclass(frozen=True)
class EventSig:
    event_id: int
    centroid: list[float] | None = None
    story_id: int | None = None
    items: tuple[ItemSig, ...] = ()
    number_facts: frozenset[str] = frozenset()   # typed distinctive numbers, for the fingerprint net


@dataclass(frozen=True)
class CandidateSignals:
    cosine: float | None = None
    simhash_distance: int | None = None
    exact_content_hash: bool = False
    forwarded_from_match: bool = False
    url_match: bool = False
    shared_number_facts: int = 0

    def as_details(self) -> dict:
        return {
            "cosine": round(self.cosine, 4) if self.cosine is not None else None,
            "simhash_distance": self.simhash_distance,
            "exact_content_hash": self.exact_content_hash,
            "forwarded_from_match": self.forwarded_from_match,
            "url_match": self.url_match,
            "shared_number_facts": self.shared_number_facts,
        }


def _norm(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    return v or None


def candidate_signals(incoming: EventSig, candidate: EventSig) -> CandidateSignals:
    """Compute the identity signals between an incoming event and a candidate. Pure."""
    cos = None
    if incoming.centroid is not None and candidate.centroid is not None:
        cos = cosine(incoming.centroid, candidate.centroid)

    inc_hashes = {i.content_hash for i in incoming.items if i.text_eligible and i.content_hash}
    cand_hashes = {j.content_hash for j in candidate.items if j.text_eligible and j.content_hash}
    exact_hash = bool(inc_hashes & cand_hashes)

    inc_fwd = {_norm(i.forwarded_from) for i in incoming.items if _norm(i.forwarded_from)}
    cand_fwd = {_norm(j.forwarded_from) for j in candidate.items if _norm(j.forwarded_from)}
    fwd_match = bool(inc_fwd & cand_fwd)

    inc_urls = {_norm(i.url) for i in incoming.items if _norm(i.url)}
    cand_urls = {_norm(j.url) for j in candidate.items if _norm(j.url)}
    url_match = bool(inc_urls & cand_urls)

    dist = None
    for i in incoming.items:
        if not i.text_eligible or i.simhash is None:
            continue
        for j in candidate.items:
            if not j.text_eligible or j.simhash is None:
                continue
            d = hamming_distance(i.simhash, j.simhash)
            if dist is None or d < dist:
                dist = d

    shared_numbers = len(incoming.number_facts & candidate.number_facts)

    return CandidateSignals(cosine=cos, simhash_distance=dist, exact_content_hash=exact_hash,
                            forwarded_from_match=fwd_match, url_match=url_match,
                            shared_number_facts=shared_numbers)


def in_candidate_net(sig: CandidateSignals, config: PredupConfig) -> bool:
    """Is this close enough to be considered at all? A LOW bar — the net that catches
    potential twins, not the duplicate verdict."""
    if sig.exact_content_hash or sig.forwarded_from_match or sig.url_match:
        return True
    if sig.cosine is not None and sig.cosine >= config.vector_candidate:
        return True
    if sig.simhash_distance is not None and sig.simhash_distance <= config.simhash_max:
        return True
    # fact-fingerprint: >=2 shared typed number-facts catches cross-source twins whose
    # embeddings fell below the vector bar (the fragmentation the measurement found).
    if sig.shared_number_facts >= 2:
        return True
    return False


def classify_signals(sig: CandidateSignals, config: PredupConfig) -> str:
    """Deterministic class of a candidate's signals. ONLY an exact content_hash match is a
    no-LLM auto-duplicate (byte-identical normalized text = genuinely the same post).

    forwarded_from is NOT auto: `items.forwarded_from` is only the ORIGIN CHANNEL NAME
    (e.g. "@kanal"), not the original message id, so two DIFFERENT posts that two channels
    both forwarded from the same source share it — auto-merging on it would swallow distinct
    news. It stays a candidate signal (→ the LLM arbiter, which reads the actual text and
    tells duplicate from separate). A matching URL / vector / SimHash is likewise a grey
    candidate. (A composite original id — peer + message id — would let forwarded_from be a
    reliable auto-duplicate; that is a later collector change.)"""
    if sig.exact_content_hash:
        return CLASS_AUTO_DUPLICATE
    if in_candidate_net(sig, config):
        return CLASS_GREY
    return CLASS_SEPARATE


# --------------------------------------------------------------------------- #
# LLM arbiter (pluggable)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TwinPair:
    """The structured context handed to the arbiter: incoming event vs one candidate."""
    incoming_title: str = ""
    incoming_texts: tuple[str, ...] = ()
    incoming_facts: tuple[str, ...] = ()
    incoming_sources: int = 0
    candidate_event_id: int = 0
    candidate_title: str = ""
    candidate_summary: str = ""
    candidate_facts: tuple[str, ...] = ()
    candidate_published: bool = False


@dataclass(frozen=True)
class TwinJudgment:
    decision: str = ACTION_HOLD_REVIEW   # duplicate | update | separate | hold_review
    confidence: float = 0.0
    reason: str = ""


_VALID_JUDGMENTS = frozenset({ACTION_DUPLICATE, ACTION_UPDATE, ACTION_SEPARATE})


def parse_twin_judgment(raw: str | None) -> TwinJudgment | None:
    """Parse the arbiter's JSON. None if unparsable (caller retries); an unknown
    decision is treated as unparsable so it is not silently mistaken for 'separate'."""
    if not raw:
        return None
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    decision = str(obj.get("decision") or "").strip().lower()
    if decision not in _VALID_JUDGMENTS:
        return None
    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return TwinJudgment(decision=decision, confidence=confidence,
                        reason=str(obj.get("reason") or "").strip()[:300])


class TwinJudge(Protocol):
    model: str

    def judge(self, pair: TwinPair) -> TwinJudgment: ...


DEFAULT_TWIN_PROMPT = """Ти редактор стрічки новин. Є НОВА подія і КАНДИДАТ — подія, про
яку ми вже маємо пост (чернетку або опубліковану). Визнач ЛИШЕ, як вони пов'язані:

- "duplicate" — це та сама подія, той самий факт; нового суттєвого немає (інше
  формулювання/джерело того самого);
- "update" — та сама історія, але нова подія додає СУТТЄВО новий факт/розвиток
  (нові цифри, наслідок, спростування), вартий окремого оновлення сюжету;
- "separate" — це РІЗНІ події (інше місто, інший випадок, інша заява), хай і схожі темою.

НЕ визначай тип оновлення (new_fact/consequence тощо) — лише зв'язок ідентичності.

Поверни лише JSON: {"decision": "duplicate|update|separate", "confidence": 0.0,
"reason": "коротко"}

НОВА ПОДІЯ:
{incoming}

КАНДИДАТ (event_id={candidate_id}, {candidate_state}):
{candidate}"""


def _render_pair(pair: TwinPair) -> tuple[str, str]:
    inc_lines = [pair.incoming_title.strip()]
    inc_lines += [f"- {t.strip()}" for t in pair.incoming_facts[:5] if t and t.strip()]
    for t in pair.incoming_texts[:2]:
        if t and t.strip():
            inc_lines.append(t.strip()[:600])
    inc_lines.append(f"(незалежних джерел: {pair.incoming_sources})")

    cand_lines = [pair.candidate_title.strip()]
    if pair.candidate_summary.strip():
        cand_lines.append(f"стан сюжету: {pair.candidate_summary.strip()[:600]}")
    cand_lines += [f"- {t.strip()}" for t in pair.candidate_facts[:5] if t and t.strip()]
    return "\n".join(filter(None, inc_lines)), "\n".join(filter(None, cand_lines))


class LLMTwinJudge:  # pragma: no cover - network
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini",
                 prompt: str = DEFAULT_TWIN_PROMPT):
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

    def _call(self, pair: TwinPair) -> str | None:
        incoming, candidate = _render_pair(pair)
        content = fill_prompt(
            self.prompt, incoming=incoming, candidate=candidate,
            candidate_id=str(pair.candidate_event_id),
            candidate_state="опублікована" if pair.candidate_published else "чернетка")
        try:
            from newsroom.llmutil import chat_json

            return chat_json(self._ensure_client(), model=self.model,
                             messages=[{"role": "user", "content": content}],
                             op="twin", max_tokens=256)
        except Exception as exc:  # noqa: BLE001
            log.warning("twin judge call failed", extra={"error": str(exc)})
            return None

    def judge(self, pair: TwinPair) -> TwinJudgment:
        for attempt in (1, 2):
            parsed = parse_twin_judgment(self._call(pair))
            if parsed is not None:
                return parsed
            log.warning("twin judgment unparsable", extra={"attempt": attempt})
        # conservative: hold for a human rather than publish a possible duplicate (§3.5)
        return TwinJudgment(decision=ACTION_HOLD_REVIEW, confidence=0.0, reason="llm_unavailable")


# --------------------------------------------------------------------------- #
# verdict + orchestrator
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TwinVerdict:
    action: str = ACTION_SEPARATE                # separate | duplicate | update | hold_review
    mode: str = "none"                           # none | auto | llm
    canonical_event_id: int | None = None        # the event this one duplicates / updates
    confidence: float | None = None
    reason: str = ""
    model: str | None = None
    signals: dict = field(default_factory=dict)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def load_event_sig(session, event_id: int) -> EventSig | None:
    from sqlalchemy import select

    from newsroom.models import Event, EventItem, Item

    ev = session.get(Event, event_id)
    if ev is None:
        return None
    rows = session.execute(
        select(Item.content_hash, Item.simhash, Item.forwarded_from, Item.url, Item.title, Item.text)
        .join(EventItem, EventItem.item_id == Item.id)
        .where(EventItem.event_id == event_id)
    ).all()
    from newsroom.collectors.base import normalize_text

    items = tuple(ItemSig(
        content_hash=ch, simhash=sh, forwarded_from=ff, url=u,
        text_eligible=bool(normalize_text(title) or normalize_text(text)),
    ) for ch, sh, ff, u, title, text in rows)
    blob = "\n".join([ev.title or ""] + [f"{t or ''}\n{x or ''}" for _c, _s, _f, _u, t, x in rows])
    return EventSig(event_id=event_id, centroid=ev.centroid, story_id=ev.story_id, items=items,
                    number_facts=extract_number_facts(blob))


def load_event_sigs(session, event_ids) -> dict[int, EventSig]:
    """Batch version of load_event_sig — TWO queries for many events (centroids, then all
    their items) instead of 1+N. Used for candidate lists so dedup does not fan out a query
    per candidate. Events with no row are omitted."""
    from sqlalchemy import select

    from newsroom.collectors.base import normalize_text
    from newsroom.models import Event, EventItem, Item

    ids = list(dict.fromkeys(event_ids))
    if not ids:
        return {}
    meta = {eid: (centroid, story_id, title) for eid, centroid, story_id, title in session.execute(
        select(Event.id, Event.centroid, Event.story_id, Event.title).where(Event.id.in_(ids))
    ).all()}
    items_by_event: dict[int, list[ItemSig]] = {eid: [] for eid in meta}
    text_by_event: dict[int, list[str]] = {eid: [meta[eid][2] or ""] for eid in meta}
    rows = session.execute(
        select(EventItem.event_id, Item.content_hash, Item.simhash, Item.forwarded_from,
               Item.url, Item.title, Item.text)
        .join(Item, Item.id == EventItem.item_id)
        .where(EventItem.event_id.in_(ids))
    ).all()
    for eid, ch, sh, ff, u, title, text in rows:
        if eid in items_by_event:
            items_by_event[eid].append(ItemSig(
                content_hash=ch, simhash=sh, forwarded_from=ff, url=u,
                text_eligible=bool(normalize_text(title) or normalize_text(text))))
            text_by_event[eid].append(f"{title or ''}\n{text or ''}")
    return {eid: EventSig(event_id=eid, centroid=c, story_id=sid,
                          items=tuple(items_by_event.get(eid, ())),
                          number_facts=extract_number_facts("\n".join(text_by_event.get(eid, ()))))
            for eid, (c, sid, _title) in meta.items()}


def find_publish_candidates(session, incoming_event_id: int, *, config: PredupConfig):
    """Return (incoming_sig, [(candidate_sig, signals), ...]) — events with a draft or
    published Telegram publication in the window whose signals put them in the candidate
    net, nearest first (exact matches first, then higher cosine, then lower SimHash)."""
    from sqlalchemy import select

    from newsroom.models import Event, Publication

    incoming = load_event_sig(session, incoming_event_id)
    if incoming is None:
        return None, []

    cutoff = _utc_now() - dt.timedelta(hours=config.window_hours)
    cand_ids = list(session.execute(
        select(Publication.event_id)
        .join(Event, Event.id == Publication.event_id)
        .where(
            Publication.channel == "telegram",
            Publication.status.in_(_CANDIDATE_STATUSES),
            Publication.event_id.is_not(None),
            Publication.event_id != incoming_event_id,
            Event.first_seen_at >= cutoff,
        ).distinct()
    ).scalars().all())

    published_ids: set[int] = set()
    if cand_ids:
        published_ids = set(session.execute(
            select(Publication.event_id).where(
                Publication.event_id.in_(cand_ids),
                Publication.status == "published",
            )
        ).scalars().all())

    scored: list[tuple[EventSig, CandidateSignals, bool]] = []
    sigs = load_event_sigs(session, cand_ids)          # 2 queries for all candidates, not 1 per
    for cid in cand_ids:
        csig = sigs.get(cid)
        if csig is None:
            continue
        sig = candidate_signals(incoming, csig)
        if in_candidate_net(sig, config):
            scored.append((csig, sig, cid in published_ids))

    scored.sort(key=lambda t: (
        0 if (t[1].exact_content_hash or t[1].forwarded_from_match) else 1,
        0 if t[2] else 1,                                    # prefer an already-published twin
        -(t[1].cosine if t[1].cosine is not None else 0.0),
        t[1].simhash_distance if t[1].simhash_distance is not None else 999,
    ))
    return incoming, scored[:config.top_candidates]


class PrepublishDedup:
    """Publish-time twin check. `check(event_id)` returns a TwinVerdict; it NEVER
    mutates state (the publisher applies the verdict, and only when enforcing)."""

    def __init__(self, session_factory, *, judge: "TwinJudge | None" = None,
                 config: PredupConfig | None = None):
        self.sf = session_factory
        self.judge = judge
        self.config = config or PredupConfig()

    def check(self, event_id: int | None) -> TwinVerdict:
        if event_id is None:
            return TwinVerdict()
        with self.sf() as s:
            incoming, candidates = find_publish_candidates(s, event_id, config=self.config)
            if incoming is None or not candidates:
                return TwinVerdict()

            # deterministic auto-duplicate on an exact signal — no LLM
            for csig, sig, _pub in candidates:
                if classify_signals(sig, self.config) == CLASS_AUTO_DUPLICATE:
                    return TwinVerdict(action=ACTION_DUPLICATE, mode="auto",
                                       canonical_event_id=csig.event_id,
                                       reason="exact content_hash",
                                       signals=sig.as_details())

            if self.judge is None:                    # no arbiter: stay conservative, don't guess
                return TwinVerdict()

            # grey zone: ask the arbiter, nearest candidate first, stop on a duplicate/update
            for csig, sig, is_pub in candidates:
                pair = self._build_pair(s, incoming, csig, is_pub)
                judgment = self.judge.judge(pair)
                if judgment.decision == ACTION_SEPARATE:
                    continue
                return TwinVerdict(
                    action=judgment.decision, mode="llm",
                    canonical_event_id=csig.event_id,
                    confidence=judgment.confidence, reason=judgment.reason,
                    model=getattr(self.judge, "model", None), signals=sig.as_details())
        return TwinVerdict()

    def _build_pair(self, s, incoming: EventSig, candidate: EventSig, is_published: bool) -> TwinPair:
        from sqlalchemy import select

        from newsroom.models import Event, EventItem, Item, Story

        inc_ev = s.get(Event, incoming.event_id)
        cand_ev = s.get(Event, candidate.event_id)
        cand_story = s.get(Story, cand_ev.story_id) if cand_ev and cand_ev.story_id else None
        inc_texts = list(s.execute(
            select(Item.text).join(EventItem, EventItem.item_id == Item.id)
            .where(EventItem.event_id == incoming.event_id).limit(2)
        ).scalars().all())
        return TwinPair(
            incoming_title=(inc_ev.title if inc_ev else "") or "",
            incoming_texts=tuple(t for t in inc_texts if t),
            incoming_facts=tuple(_facts(inc_ev.fact_base if inc_ev else None)),
            incoming_sources=int(inc_ev.independent_source_count or 0) if inc_ev else 0,
            candidate_event_id=candidate.event_id,
            candidate_title=(cand_ev.title if cand_ev else "") or "",
            candidate_summary=(cand_story.current_summary if cand_story else "") or "",
            candidate_facts=tuple(_facts(cand_ev.fact_base if cand_ev else None)),
            candidate_published=is_published,
        )


def _facts(fact_base, *, limit: int = 5) -> list[str]:
    if not isinstance(fact_base, dict):
        return []
    rows = [f for f in (fact_base.get("facts") or []) if isinstance(f, dict) and f.get("text")]
    rows.sort(key=lambda f: int(f.get("confirmed_by") or 0), reverse=True)
    return [str(f["text"]).strip() for f in rows[:limit]]
