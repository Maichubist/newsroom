"""Taxonomy spine (charter v0.3) — the single fixed controlled rubric vocabulary.

Replaces the two out-of-sync systems (flat `events.rubric` + the learned pyramid
roots) with ONE stable spine. Beneath it the pyramid stays dynamic; the spine is
POLICY (risk floor, human oversight), not news structure, so it changes rarely and
is human-owned. Loaded from config/taxonomy_spine.yaml.

Pure/offline (like risk.py): parse + validate + resolve, no DB, no
network. `resolve()` maps any old flat rubric (English) OR old pyramid root
(Ukrainian) OR a slug to its canonical spine slug via the alias index.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

FLOORS = ("low", "high", "critical")


class SpineConfigError(ValueError):
    """Raised when taxonomy_spine.yaml is malformed."""


@dataclass(frozen=True)
class SpineRubric:
    slug: str
    display: str
    floor: str
    oversight: bool
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Spine:
    rubrics: dict[str, SpineRubric]           # slug -> rubric (insertion order = config order)
    alias_index: dict[str, str]               # normalized alias/slug -> canonical slug
    default_floor: str = "high"

    @staticmethod
    def _norm(name: str | None) -> str:
        return (name or "").strip().lower()

    def resolve(self, name: str | None) -> str | None:
        """Canonical slug for a slug / old flat rubric / old pyramid root / alias.
        None when unknown (caller decides the fallback)."""
        return self.alias_index.get(self._norm(name))

    def floor_for(self, name: str | None) -> str:
        """Risk floor for a rubric name; default_floor (conservative) when unknown."""
        slug = self.resolve(name)
        return self.rubrics[slug].floor if slug else self.default_floor

    def floor_for_any(self, names) -> str:
        """Highest floor among several rubric names (charter: при неоднозначності —
        вищий); default_floor when none resolve. Mirrors risk.RiskMatrix.level_for so
        the spine can replace it as the floor source."""
        rank = {"low": 0, "high": 1, "critical": 2}
        chosen: str | None = None
        for name in names or []:
            slug = self.resolve(name)
            if slug:
                lvl = self.rubrics[slug].floor
                if chosen is None or rank[lvl] > rank[chosen]:
                    chosen = lvl
        return chosen or self.default_floor

    def needs_oversight(self, name: str | None) -> bool:
        """True if a critical-topic post of this rubric must notify the supervisor.
        Unknown rubrics do NOT trigger oversight (they get the conservative floor,
        but a notice needs a known, deliberately-flagged rubric)."""
        slug = self.resolve(name)
        return bool(slug and self.rubrics[slug].oversight)

    def oversight_slugs(self) -> tuple[str, ...]:
        return tuple(r.slug for r in self.rubrics.values() if r.oversight)


def load_spine(path: str | Path) -> Spine:
    path = Path(path)
    if not path.exists():
        raise SpineConfigError(f"spine config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    raw = data.get("rubrics")
    if not isinstance(raw, list) or not raw:
        raise SpineConfigError("taxonomy_spine.yaml must have a non-empty 'rubrics' list")

    default_floor = str(data.get("default_floor", "high")).strip().lower()
    if default_floor not in FLOORS:
        raise SpineConfigError(f"default_floor must be one of {FLOORS}, got {default_floor!r}")

    rubrics: dict[str, SpineRubric] = {}
    alias_index: dict[str, str] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise SpineConfigError(f"each rubric must be a mapping, got {entry!r}")
        slug = str(entry.get("slug", "")).strip().lower()
        if not slug:
            raise SpineConfigError(f"rubric missing slug: {entry!r}")
        if slug in rubrics:
            raise SpineConfigError(f"duplicate slug {slug!r}")
        floor = str(entry.get("floor", "")).strip().lower()
        if floor not in FLOORS:
            raise SpineConfigError(f"rubric {slug!r}: floor must be one of {FLOORS}, got {floor!r}")
        display = str(entry.get("display", "")).strip() or slug
        oversight = bool(entry.get("oversight", False))

        aliases = entry.get("aliases") or []
        if not isinstance(aliases, list):
            raise SpineConfigError(f"rubric {slug!r}: aliases must be a list")
        norm_aliases: list[str] = []
        # the slug always resolves to itself
        for alias in [slug, *aliases]:
            key = str(alias).strip().lower()
            if not key:
                continue
            existing = alias_index.get(key)
            if existing is not None and existing != slug:
                raise SpineConfigError(
                    f"alias {key!r} maps to both {existing!r} and {slug!r}")
            alias_index[key] = slug
            if key != slug:
                norm_aliases.append(key)

        rubrics[slug] = SpineRubric(slug=slug, display=display, floor=floor,
                                    oversight=oversight, aliases=tuple(norm_aliases))

    return Spine(rubrics=rubrics, alias_index=alias_index, default_floor=default_floor)


# --- semi-automatic evolution: propose spine changes, a human ratifies --------
# The spine is stable POLICY; it does not auto-recluster. But the pyramid beneath it is
# dynamic, so the system WATCHES for two signals that the spine should change and surfaces
# them (admin /spine) for a human to approve — it never edits the spine itself.

@dataclass(frozen=True)
class SpineProposal:
    kind: str          # "add" (a topic maps to nothing) | "cold" (a rubric has no events)
    topic: str         # the unmapped rubric, or the cold spine slug
    count: int
    detail: str


def spine_proposals(rubric_counts: dict[str, int], spine: Spine, *, min_events: int = 20,
                    cold_min_total: int = 200) -> list[SpineProposal]:
    """Pure: from {rubric -> event count} propose spine changes. (1) A rubric with volume
    that maps to NO spine rubric -> propose adding a rubric or alias. (2) Only once there is
    enough data overall, a spine rubric that drew ZERO events -> propose review/merge. The
    human decides; this never mutates the spine."""
    proposals: list[SpineProposal] = []
    for rubric, n in sorted(rubric_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if n >= min_events and spine.resolve(rubric) is None:
            proposals.append(SpineProposal("add", rubric, int(n), "не мапиться на жодну рубрику"))

    total = sum(rubric_counts.values())
    if total >= cold_min_total:
        per_slug: dict[str, int] = {slug: 0 for slug in spine.rubrics}
        for rubric, n in rubric_counts.items():
            slug = spine.resolve(rubric)
            if slug:
                per_slug[slug] += int(n)
        for slug in spine.rubrics:            # config order, stable output
            if per_slug[slug] == 0:
                proposals.append(SpineProposal("cold", slug, 0, "нема подій — розглянь злиття/перегляд"))
    return proposals


def propose_spine_changes(session, spine: Spine, *, window_days: int = 14,
                          min_events: int = 20) -> list[SpineProposal]:
    """Count recent events per rubric and hand them to spine_proposals. Read-only."""
    import datetime as dt

    from sqlalchemy import func, select

    from newsroom.models import Event

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)
    rows = session.execute(
        select(Event.rubric, func.count(Event.id))
        .where(Event.first_seen_at >= cutoff, Event.rubric.is_not(None), Event.duplicate_of.is_(None))
        .group_by(Event.rubric)
    ).all()
    counts = {str(r): int(n) for r, n in rows if r}
    return spine_proposals(counts, spine, min_events=min_events)
