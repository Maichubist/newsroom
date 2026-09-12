"""Significance gate (T1): is an event worth posting at all?

The pipeline had no "interesting/significant for our reader" step — anything that
reached confirmed/reported got a post, so niche and minor news flooded the channel.
This adds a **deterministic** significance score from stable signals (rubric weight,
UA-relevance, source corroboration, story momentum) — not an absolute LLM score,
which CLAUDE.md forbids as unstable. An event below the config threshold is not
drafted (journalled low_significance). "Raise the threshold" = one config line.

The scoring and UA-relevance detection are pure and offline-tested; the DB step
(score_pending) writes events.significance and journals the decision.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger("newsroom.analyze.significance")

# statuses that could become a post — only these are scored
POSTABLE_STATUSES = ("reported", "confirmed", "rumor")


class SignificanceConfigError(ValueError):
    """Raised when significance.yaml is malformed."""


@dataclass(frozen=True)
class RubricWeight:
    weight: float
    locality_sensitive: bool


@dataclass(frozen=True)
class SignificanceConfig:
    threshold: float = 0.55
    ua_bonus: float = 0.25
    foreign_penalty: float = 0.30
    corroboration_step: float = 0.08
    corroboration_cap: float = 0.16
    story_bonus: float = 0.12
    story_min_events: int = 3
    critical_floor: float = 0.95
    default_weight: float = 0.50
    default_locality_sensitive: bool = True
    rubrics: dict[str, RubricWeight] = field(default_factory=dict)
    ua_markers: tuple[re.Pattern, ...] = ()

    def rubric_weight(self, rubric: str | None) -> RubricWeight:
        if rubric and rubric.lower() in self.rubrics:
            return self.rubrics[rubric.lower()]
        return RubricWeight(self.default_weight, self.default_locality_sensitive)


def load_significance_config(path: str | Path) -> SignificanceConfig:
    path = Path(path)
    if not path.exists():
        raise SignificanceConfigError(f"significance config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        rubrics = {
            str(name).lower(): RubricWeight(
                weight=float(spec["weight"]),
                locality_sensitive=bool(spec.get("locality_sensitive", True)),
            )
            for name, spec in (data.get("rubrics") or {}).items()
        }
        markers = tuple(
            re.compile(str(p), re.IGNORECASE) for p in (data.get("ua_markers") or []) if str(p).strip()
        )
        return SignificanceConfig(
            threshold=float(data.get("threshold", 0.55)),
            ua_bonus=float(data.get("ua_bonus", 0.25)),
            foreign_penalty=float(data.get("foreign_penalty", 0.30)),
            corroboration_step=float(data.get("corroboration_step", 0.08)),
            corroboration_cap=float(data.get("corroboration_cap", 0.16)),
            story_bonus=float(data.get("story_bonus", 0.12)),
            story_min_events=int(data.get("story_min_events", 3)),
            critical_floor=float(data.get("critical_floor", 0.95)),
            default_weight=float(data.get("default_weight", 0.50)),
            default_locality_sensitive=bool(data.get("default_locality_sensitive", True)),
            rubrics=rubrics,
            ua_markers=markers,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise SignificanceConfigError(f"bad significance value: {exc}") from exc


def is_ua_relevant(text: str | None, markers: tuple[re.Pattern, ...]) -> bool:
    """True if the text carries any Ukrainian marker. A positive signal (bonus),
    not a deny-filter — a miss only lowers the score, so a false negative can't
    block legitimate news the way an over-eager deny rule would."""
    blob = (text or "")
    return any(p.search(blob) for p in markers)


@dataclass(frozen=True)
class SignificanceInputs:
    rubric: str | None = None
    risk_level: str | None = None
    text: str = ""                       # title + excerpt, for UA-marker detection
    independent_source_count: int = 0
    story_event_count: int = 0


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passes: bool
    reasons: list[str] = field(default_factory=list)


def significance_score(inp: SignificanceInputs, config: SignificanceConfig) -> ScoreResult:
    """Deterministic significance in [0, 1]. Foreign penalty applies only to
    locality-sensitive rubrics (sport / society / lifestyle …), so a major foreign
    story (geopolitics, tech) is not treated as niche. Critical risk is floored so
    safety news is never dropped for 'significance' (official sourcing is a separate
    gate)."""
    reasons: list[str] = []
    rw = config.rubric_weight(inp.rubric)
    score = rw.weight

    if is_ua_relevant(inp.text, config.ua_markers):
        score += config.ua_bonus
        reasons.append("ua_relevant")
    elif rw.locality_sensitive:
        score -= config.foreign_penalty
        reasons.append("foreign_locality_sensitive")

    extra_sources = max(inp.independent_source_count - 1, 0)
    if extra_sources:
        score += min(extra_sources * config.corroboration_step, config.corroboration_cap)
        reasons.append("corroborated")

    if inp.story_event_count >= config.story_min_events:
        score += config.story_bonus
        reasons.append("developing_story")

    score = max(0.0, min(1.0, score))

    if (inp.risk_level or "").lower() == "critical":
        score = max(score, config.critical_floor)
        reasons.append("critical_floor")

    return ScoreResult(score=score, passes=score >= config.threshold, reasons=reasons)


def score_pending(session_factory, config: SignificanceConfig, *, limit: int = 100) -> dict[str, int]:
    """One scoring tick: score every postable event that has no significance yet.
    Writes events.significance and journals the verdict, so editorial can skip the
    low-significance ones. Deterministic — no network."""
    from sqlalchemy import func, select

    from newsroom.models import Decision, Event, EventItem, Item

    with session_factory() as s:
        ids = list(s.execute(
            select(Event.id)
            .where(Event.significance.is_(None), Event.status.in_(POSTABLE_STATUSES))
            .order_by(Event.id)
            .limit(limit)
        ).scalars().all())

    stats = {"scored": 0, "significant": 0, "low": 0}
    for event_id in ids:
        with session_factory() as s:
            event = s.get(Event, event_id)
            if event is None:
                continue
            rows = s.execute(
                select(Item.title, Item.text)
                .join(EventItem, EventItem.item_id == Item.id)
                .where(EventItem.event_id == event_id)
                .limit(6)
            ).all()
            text = (event.title or "") + "\n" + "\n".join(f"{t or ''} {x or ''}" for t, x in rows)
            story_count = 0
            if event.story_id is not None:
                story_count = int(s.scalar(
                    select(func.count()).select_from(Event).where(Event.story_id == event.story_id)
                ) or 0)

            result = significance_score(SignificanceInputs(
                rubric=event.rubric, risk_level=event.risk_level, text=text[:4000],
                independent_source_count=event.independent_source_count or 0,
                story_event_count=story_count,
            ), config)

            event.significance = result.score
            s.add(Decision(
                entity_type="event", entity_id=str(event_id), stage="signal",
                decision="significant" if result.passes else "low_significance",
                reason=",".join(result.reasons) or None,
                details={"score": round(result.score, 3), "threshold": config.threshold,
                         "reasons": result.reasons},
            ))
            s.commit()

        stats["scored"] += 1
        stats["significant" if result.passes else "low"] += 1
    return stats
