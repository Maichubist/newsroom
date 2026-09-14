"""Editorial curation (§: publish what's worth it, not a fixed rate).

Instead of throttling output to N posts/hour, we decide *which* events are worth
publishing and let the count follow the news — a quiet window yields few posts, a
big-news window yields more. Two paths:

  * must_publish — deterministic: ONLY a refutation (a correction must go out). It is
    marked publish without the LLM so a correction is never held or delayed.
  * everything else — including breaking critical news — a **comparative** LLM pass over
    the recent window of candidates marks
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


def must_publish(*, risk_level: str | None = None, has_official_source: bool = False,
                 update_type: str | None = None) -> bool:
    """Deterministic must-publish: ONLY a refutation (a correction must always go out,
    charter). Everything else — including breaking critical+official news — now goes
    through the editorial ranker, so content is judged and content-free war alerts no
    longer auto-publish. (risk_level/has_official_source are kept for signature
    stability but no longer force a publish.)"""
    return (update_type or "").lower() == "refutation"


@dataclass(frozen=True)
class Candidate:
    event_id: int
    title: str
    rubric: str | None = None
    risk_level: str | None = None
    significance: float | None = None
    facts: list[str] = field(default_factory=list)
    demand: float | None = None        # learned audience demand for the rubric (0..1), or None
    heat: float | None = None          # data-driven hot-topic heat for this event (0..1), or None


def _demand_label(demand: float | None) -> str:
    if demand is None:
        return "невідомо"
    if demand >= 0.66:
        return "високий"
    if demand >= 0.33:
        return "середній"
    return "низький"


def _heat_label(heat: float | None) -> str:
    if not heat:
        return "—"
    if heat >= 0.66:
        return "гаряча"
    if heat >= 0.33:
        return "тепла"
    return "прохолодна"


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

Зважуй чинники:
- ВАЖЛИВІСТЬ: суспільна вага для українського читача (політика, безпека/фронт,
  економіка, важливі рішення, помітні міжнародні події, що нас стосуються).
- ПОПИТ АУДИТОРІЇ: наскільки рубрика цікавить масового читача («попит» біля кандидата —
  вивчений із реакцій на схожі новини в інших каналах). Важливе, але буденне (щоденні
  зведення роками) читач гортає повз; свіже й резонансне — читає.
- ГАРЯЧІСТЬ ТЕМИ: «тема» біля кандидата — наскільки саме цей сюжет зараз активно
  обговорюють у стрічці конкурентів (гаряча = багато постів + залученість зараз).
  Гаряча тема — сильний сигнал на користь публікації, поки вона в тренді.

ПУБЛІКУВАТИ обов'язково — справді значущі безпекові/фронтові події: великі удари з
наслідками (жертви, руйнування, влучання по інфраструктурі), помітна ескалація,
важливі офіційні рішення й заяви щодо оборони. Рутинні алерти повітряної обстановки
(проліт/рух БпЛА без наслідків) сюди НЕ потрапляють — їх уже відсіяно або зведено в
окремий дайджест обстрілів, тож не публікуй поштучні «БпЛА над містом».

ПРИТРИМАти (hold) — дрібне, вузьконішеве, прохідне, дубль уже відомого, суто
розважальне без ширшого значення, або важливе-але-рутинне з низьким попитом.
Окремо: повідомлення про загибель КОНКРЕТНОЇ людини, невідомої широкому загалу
(некролог, прощання з бійцем, «загинув військовий N») — hold. Це гідно шани, але не
масова новина, і канал не має ставати стрічкою некрологів. ВИНЯТОК (publish): масові
втрати внаслідок удару/події або загибель відомої публічної особи.

Порівнюй кандидатів між собою: якщо подія слабша за решту в списку і за важливістю, і
за попитом — hold.

Поверни лише JSON: {"decisions": [{"id": <число>, "decision": "publish|hold"}, ...]}
для КОЖНОГО кандидата.

КАНДИДАТИ:
{candidates}"""


def _render_candidates(candidates: list[Candidate]) -> str:
    lines: list[str] = []
    for c in candidates:
        head = (f"[id={c.event_id}] ({c.rubric or '?'}/{c.risk_level or '?'}, "
                f"попит: {_demand_label(c.demand)}, тема: {_heat_label(c.heat)}) {c.title.strip()}")
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

    from sqlalchemy import select

    from newsroom.models import Decision, Event

    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=window_hours)

    conditions = [
        Event.status.in_(POSTABLE_STATUSES),
        Event.curated.is_(None),
        Event.duplicate_of.is_(None),        # skip events marked duplicate (LLM batch dedup)
        Event.first_seen_at >= cutoff,
    ]
    if significance_threshold is not None:
        conditions.append(Event.significance >= significance_threshold)

    from newsroom.analyze.demand import load_demand
    from newsroom.analyze.topics import load_hot_topics, topic_heat
    from newsroom.models import Publication

    with session_factory() as s:
        have_pub = select(Publication.event_id).where(Publication.event_id.is_not(None))
        rows = s.execute(
            select(Event.id, Event.title, Event.rubric, Event.risk_level,
                   Event.significance, Event.fact_base, Event.update_type, Event.keywords)
            .where(*conditions, Event.id.not_in(have_pub))   # don't re-curate already-published events
            .order_by(Event.significance.desc().nullslast(), Event.id).limit(limit)
        ).all()
        if not rows:
            return {"curated": 0, "publish": 0, "hold": 0, "must": 0}

        demand_by_rubric = load_demand(s)      # learned audience demand per rubric ({} until data)
        hot = load_hot_topics(s)               # data-driven hot-topic heat ({} until data)

    decisions: dict[int, str] = {}
    must_ids: set[int] = set()
    to_rank: list[Candidate] = []
    for eid, title, rubric, risk, sig, fact_base, update_type, keywords in rows:
        # only a refutation skips the editor (a correction must go out); everything else,
        # breaking critical news included, is judged comparatively by the ranker
        if must_publish(update_type=update_type):
            decisions[eid] = CURATE_PUBLISH
            must_ids.add(eid)
        else:
            to_rank.append(Candidate(event_id=eid, title=title or "", rubric=rubric, risk_level=risk,
                                     significance=sig, facts=_facts_brief(fact_base),
                                     demand=demand_by_rubric.get(rubric) if rubric else None,
                                     heat=topic_heat(keywords, hot) or None))
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
                details={"must": eid in must_ids},
                model=getattr(ranker, "model", None),
            ))
            stats["curated"] += 1
            stats[decision] += 1
        s.commit()
    return stats
