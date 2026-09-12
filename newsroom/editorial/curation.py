"""Editorial curation (§: publish what's worth it, not a fixed rate).

Instead of throttling output to N posts/hour, we decide *which* events are worth
publishing and let the count follow the news — a quiet window yields few posts, a
big-news window yields more. Two paths:

  * must_publish — deterministic: a critical event with an official source, or a
    refutation (a correction must go out). These are marked publish without the LLM,
    so breaking news never waits on the ranker.
  * the rest — a **comparative** LLM pass over the recent window of candidates marks
    each publish/hold ("would a serious Ukrainian news+analysis channel run this?").
    Comparative-in-a-batch, not an absolute score (CLAUDE.md).

Curation gates *drafting*: only publish-marked events are drafted, so we don't spend
generation tokens on posts we would not publish. The significance gate still runs
first as a cheap pre-filter, so the ranker never sees obvious niche.

The vocabulary, must_publish and parsing are pure and offline-tested; the ranker is
pluggable (LLM in prod, fake in tests).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

from newsroom.promptutil import fill_prompt

log = logging.getLogger("newsroom.editorial.curation")

CURATE_PUBLISH = "publish"
CURATE_HOLD = "hold"
POSTABLE_STATUSES = ("reported", "confirmed", "rumor")


def must_publish(*, risk_level: str | None, has_official_source: bool,
                 update_type: str | None = None) -> bool:
    """Deterministic must-publish: a critical event confirmed by an official source
    (breaking safety/war) or a refutation (a correction always goes out). These skip
    the ranker so they are never held or delayed by it."""
    if (risk_level or "").lower() == "critical" and has_official_source:
        return True
    if (update_type or "").lower() == "refutation":
        return True
    return False


@dataclass(frozen=True)
class Candidate:
    event_id: int
    title: str
    rubric: str | None = None
    risk_level: str | None = None
    significance: float | None = None
    facts: list[str] = field(default_factory=list)


def parse_ranking(raw: str | None, valid_ids: set[int]) -> dict[int, str]:
    """Parse the ranker's JSON into {event_id: publish|hold}. Unknown ids are
    ignored; any valid candidate the model omits defaults to hold (be selective —
    silence means 'not worth it'). Unparsable -> everything holds."""
    out: dict[int, str] = {eid: CURATE_HOLD for eid in valid_ids}
    if not raw:
        return out
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError):
        return out
    rows = obj.get("decisions") if isinstance(obj, dict) else obj
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            eid = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        decision = str(row.get("decision") or "").strip().lower()
        if eid in valid_ids and decision in (CURATE_PUBLISH, CURATE_HOLD):
            out[eid] = decision
    return out


class EditorialRanker(Protocol):
    model: str

    def rank(self, candidates: list[Candidate]) -> dict[int, str]: ...


DEFAULT_RANK_PROMPT = """Ти — випусковий редактор серйозного українського новинно-
аналітичного каналу. Нижче список подій-кандидатів за останні години. Виріши, ЩО
справді варте публікації просто зараз, а що — ні. Будь вибагливим: краще менше, але
вагоме. Це не стрічка всього підряд.

ПУБЛІКУВати (publish) — подія має суспільну вагу для українського читача: політика,
безпека/фронт, економіка, важливі рішення, помітні міжнародні події, що нас
стосуються, резонансні суспільні історії.
ПРИТРИМАти (hold) — дрібне, вузьконішеве, прохідне, дубль уже відомого, суто
розважальне без ширшого значення.

Порівнюй кандидатів між собою: якщо подія слабша за решту в списку — hold.

Поверни лише JSON: {"decisions": [{"id": <число>, "decision": "publish|hold"}, ...]}
для КОЖНОГО кандидата.

КАНДИДАТИ:
{candidates}"""


def _render_candidates(candidates: list[Candidate]) -> str:
    lines: list[str] = []
    for c in candidates:
        head = f"[id={c.event_id}] ({c.rubric or '?'}/{c.risk_level or '?'}) {c.title.strip()}"
        lines.append(head)
        for f in c.facts[:3]:
            if f and f.strip():
                lines.append(f"   - {f.strip()}")
    return "\n".join(lines)


class LLMEditorialRanker:  # pragma: no cover - network
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini",
                 prompt: str = DEFAULT_RANK_PROMPT):
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

    def rank(self, candidates: list[Candidate]) -> dict[int, str]:
        valid = {c.event_id for c in candidates}
        if not valid:
            return {}
        content = fill_prompt(self.prompt, candidates=_render_candidates(candidates))
        try:
            resp = self._ensure_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return parse_ranking(resp.choices[0].message.content, valid)
        except Exception as exc:  # noqa: BLE001
            log.warning("curation rank failed", extra={"error": str(exc)})
            # conservative: hold all rather than publishing unreviewed on an error
            return {eid: CURATE_HOLD for eid in valid}


def _facts_brief(fact_base, *, limit: int = 3) -> list[str]:
    if not isinstance(fact_base, dict):
        return []
    rows = [f for f in (fact_base.get("facts") or []) if isinstance(f, dict) and f.get("text")]
    rows.sort(key=lambda f: int(f.get("confirmed_by") or 0), reverse=True)
    return [str(f["text"]).strip() for f in rows[:limit]]


def curate_pending(session_factory, ranker: "EditorialRanker", *, significance_threshold: float | None = None,
                   window_hours: int = 6, limit: int = 40) -> dict[str, int]:
    """One curation tick: mark recent significant, not-yet-curated events publish/hold.
    must-publish events are marked deterministically; the rest are ranked comparatively
    by the LLM. Only publish-marked events are later drafted."""
    import datetime as dt

    from sqlalchemy import func, select

    from newsroom.models import Decision, Event, EventItem, Item, Source

    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=window_hours)

    conditions = [
        Event.status.in_(POSTABLE_STATUSES),
        Event.curated.is_(None),
        Event.first_seen_at >= cutoff,
    ]
    if significance_threshold is not None:
        conditions.append(Event.significance >= significance_threshold)

    with session_factory() as s:
        rows = s.execute(
            select(Event.id, Event.title, Event.rubric, Event.risk_level,
                   Event.significance, Event.fact_base, Event.update_type)
            .where(*conditions).order_by(Event.significance.desc().nullslast(), Event.id).limit(limit)
        ).all()
        if not rows:
            return {"curated": 0, "publish": 0, "hold": 0, "must": 0}

        # which events have an official source (for must_publish)
        event_ids = [r[0] for r in rows]
        official_ids = set(s.execute(
            select(EventItem.event_id).join(Item, Item.id == EventItem.item_id)
            .join(Source, Source.id == Item.source_id)
            .where(EventItem.event_id.in_(event_ids), Source.is_official.is_(True))
            .distinct()
        ).scalars().all())

    decisions: dict[int, str] = {}
    to_rank: list[Candidate] = []
    for eid, title, rubric, risk, sig, fact_base, update_type in rows:
        if must_publish(risk_level=risk, has_official_source=eid in official_ids, update_type=update_type):
            decisions[eid] = CURATE_PUBLISH
        else:
            to_rank.append(Candidate(event_id=eid, title=title or "", rubric=rubric, risk_level=risk,
                                     significance=sig, facts=_facts_brief(fact_base)))
    must_count = len(decisions)

    if to_rank:
        decisions.update(ranker.rank(to_rank))

    stats = {"curated": 0, "publish": 0, "hold": 0, "must": must_count}
    with session_factory() as s:
        for eid, decision in decisions.items():
            event = s.get(Event, eid)
            if event is None or event.curated is not None:
                continue
            event.curated = decision
            s.add(Decision(
                entity_type="event", entity_id=str(eid), stage="edit",
                decision=f"curate_{decision}", reason=None,
                details={"must": eid in official_ids and decision == CURATE_PUBLISH},
                model=getattr(ranker, "model", None),
            ))
            stats["curated"] += 1
            stats[decision] += 1
        s.commit()
    return stats
