"""Risk matrix + publication gate (charter §3.1, §3.2).

Two pieces, both pure/offline:
  * RiskMatrix — rubric -> risk level from config/risk.yaml; multiple rubrics
    resolve to the HIGHEST level ("при неоднозначності застосовується вищий").
  * decide(...) — given the level and the source evidence, returns the charter
    status and whether the item may be published autonomously.

The gate is deliberately conservative (asymmetry of errors, architecture §3.5):
blocking a legitimate item is cheap, publishing forbidden/unverified is not.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

LEVELS = ("low", "high", "critical")
_RANK = {"low": 0, "high": 1, "critical": 2}

# charter §3.2 statuses (refuted is produced by the correction flow, not the gate)
STATUS_SIGNAL = "signal"      # not enough — wait for confirmation, do not publish
STATUS_RUMOR = "rumor"        # published under rule 3.7 with a label
STATUS_REPORTED = "reported"  # matrix satisfied but no first source ("Повідомляють")
STATUS_CONFIRMED = "confirmed"  # official / first-hand / documented


class RiskConfigError(ValueError):
    """Raised when risk.yaml is malformed."""


@dataclass(frozen=True)
class RiskMatrix:
    rubric_level: dict[str, str]
    default_level: str = "high"

    def level_for(self, rubrics: Iterable[str]) -> str:
        """Highest configured level among the rubrics; default when none known."""
        chosen: str | None = None
        for rubric in rubrics or []:
            lvl = self.rubric_level.get(str(rubric).strip().lower())
            if lvl and (chosen is None or _RANK[lvl] > _RANK[chosen]):
                chosen = lvl
        return chosen or self.default_level


def load_risk_matrix(path: str | Path) -> RiskMatrix:
    path = Path(path)
    if not path.exists():
        raise RiskConfigError(f"risk config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    levels = data.get("levels")
    if not isinstance(levels, dict) or not levels:
        raise RiskConfigError("risk.yaml must have a non-empty 'levels' mapping")

    rubric_level: dict[str, str] = {}
    for level, rubrics in levels.items():
        if level not in LEVELS:
            raise RiskConfigError(f"unknown level {level!r}; allowed: {LEVELS}")
        if not isinstance(rubrics, list):
            raise RiskConfigError(f"level {level!r} must map to a list of rubrics")
        for rubric in rubrics:
            key = str(rubric).strip().lower()
            if not key:
                continue
            if key in rubric_level and rubric_level[key] != level:
                # a rubric under two levels: keep the higher (conservative)
                if _RANK[level] <= _RANK[rubric_level[key]]:
                    continue
            rubric_level[key] = level

    default_level = str(data.get("default_level", "high")).strip().lower()
    if default_level not in LEVELS:
        raise RiskConfigError(f"default_level must be one of {LEVELS}, got {default_level!r}")
    return RiskMatrix(rubric_level=rubric_level, default_level=default_level)


@dataclass(frozen=True)
class GateDecision:
    level: str
    status: str
    publishable: bool
    reason: str


def decide(
    level: str,
    *,
    independent_sources: int = 0,
    has_official: bool = False,
    has_first_source: bool = False,
    high_reputation: bool = False,
    is_rumor: bool = False,
) -> GateDecision:
    """Apply the matrix condition for `level` to the available evidence."""
    if level not in LEVELS:
        raise ValueError(f"unknown risk level {level!r}")

    # Rumors (charter §3.7): allowed only below the critical level, always labelled.
    if is_rumor:
        if level == "critical":
            return GateDecision(level, STATUS_SIGNAL, False,
                                "rumor not allowed for critical topics (charter 3.7.2)")
        return GateDecision(level, STATUS_RUMOR, True, "published as rumor (charter 3.7)")

    if level == "critical":
        if has_official:
            return GateDecision(level, STATUS_CONFIRMED, True, "official source present")
        return GateDecision(level, STATUS_SIGNAL, False,
                            "critical topic requires an official source (charter 3.1)")

    if level == "high":
        if has_official or has_first_source:
            return GateDecision(level, STATUS_CONFIRMED, True, "first-hand or official source")
        if independent_sources >= 2:
            return GateDecision(level, STATUS_REPORTED, True, "2+ independent sources, no first source")
        return GateDecision(level, STATUS_SIGNAL, False,
                            "high-risk needs 2+ independent sources or a first source")

    # low
    if has_official or has_first_source:
        return GateDecision(level, STATUS_CONFIRMED, True, "official or first-hand source")
    if high_reputation:
        return GateDecision(level, STATUS_REPORTED, True, "single high-reputation source")
    return GateDecision(level, STATUS_SIGNAL, False, "low-risk needs a high-reputation source")
