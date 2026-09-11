"""Shadow-mode exit criteria (architecture §2).

Before publishing for real, the system runs 1-2 weeks into a closed test channel
(SHADOW_MODE routes there). Exit criteria are set before it starts and checked
here from what was actually published:

  * enough volume over enough days (a small sample proves nothing);
  * zero stop-list violations among published posts (re-checked on the final body);
  * duplicate share below threshold — are we posting the same story twice?

The remaining criterion — "no fake slipped through on the labelled sample" — is a
human judgment on a labelled set, so the report only flags it for a person; it is
never auto-marked met. The dup math is pure and offline-tested; the report reads
the shadow publications.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from newsroom.collectors.base import compute_simhash, hamming_distance


class ShadowConfigError(ValueError):
    """Raised when shadow.yaml is malformed."""


@dataclass(frozen=True)
class ShadowCriteria:
    min_days: int = 7
    min_publications: int = 30
    max_dup_rate: float = 0.15
    dup_max_hamming: int = 6


def load_shadow_criteria(path: str | Path) -> ShadowCriteria:
    path = Path(path)
    if not path.exists():
        raise ShadowConfigError(f"shadow config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return ShadowCriteria(
            min_days=int(data.get("min_days", 7)),
            min_publications=int(data.get("min_publications", 30)),
            max_dup_rate=float(data.get("max_dup_rate", 0.15)),
            dup_max_hamming=int(data.get("dup_max_hamming", 6)),
        )
    except (TypeError, ValueError) as exc:
        raise ShadowConfigError(f"bad shadow value: {exc}") from exc


def dup_rate_from_simhashes(simhashes: list[int], *, max_hamming: int) -> float:
    """Share of items that near-duplicate an earlier item (by simhash). Order is
    the publication order; the first occurrence of a story is not a duplicate, its
    repeats are. Pure — the tested core of the dup metric."""
    if not simhashes:
        return 0.0
    seen: list[int] = []
    dups = 0
    for sh in simhashes:
        if any(hamming_distance(sh, prev) <= max_hamming for prev in seen):
            dups += 1
        else:
            seen.append(sh)
    return dups / len(simhashes)


@dataclass(frozen=True)
class ShadowReport:
    publications: int = 0
    days_running: float | None = None
    stoplist_violations: int = 0
    dup_rate: float = 0.0
    auto_criteria_met: bool = False
    notes: list[str] = field(default_factory=list)


def shadow_report(session_factory, *, criteria: ShadowCriteria, stoplist_rules) -> ShadowReport:
    """Assess the automatable exit criteria over posts published in shadow mode."""
    from sqlalchemy import select

    from newsroom.analyze.stoplist import check as stoplist_check
    from newsroom.analyze.stoplist import is_blocked
    from newsroom.models import Publication

    with session_factory() as s:
        rows = list(s.execute(
            select(Publication.headline, Publication.body, Publication.published_at)
            .where(
                Publication.status == "published",
                Publication.features["shadow"].as_boolean().is_(True),
            )
            .order_by(Publication.published_at)
        ).all())

    count = len(rows)
    if count == 0:
        return ShadowReport(notes=["no shadow publications yet",
                                   "manual: confirm no fake slipped through on the labelled sample"])

    times = [r[2] for r in rows if r[2] is not None]
    days_running = None
    if times:
        days_running = (dt.datetime.now(dt.timezone.utc) - min(times)).total_seconds() / 86400.0

    stoplist_violations = sum(
        1 for headline, body, _ in rows if is_blocked(stoplist_check(headline, body, stoplist_rules))
    )
    simhashes = [compute_simhash(headline, body) for headline, body, _ in rows]
    dup_rate = dup_rate_from_simhashes(simhashes, max_hamming=criteria.dup_max_hamming)

    auto_met = (
        count >= criteria.min_publications
        and days_running is not None and days_running >= criteria.min_days
        and stoplist_violations == 0
        and dup_rate <= criteria.max_dup_rate
    )
    notes: list[str] = []
    if count < criteria.min_publications:
        notes.append(f"need {criteria.min_publications} publications, have {count}")
    if days_running is not None and days_running < criteria.min_days:
        notes.append(f"need {criteria.min_days} days, have {days_running:.1f}")
    if stoplist_violations:
        notes.append(f"{stoplist_violations} stop-list violation(s) among published posts")
    if dup_rate > criteria.max_dup_rate:
        notes.append(f"dup rate {dup_rate:.2f} exceeds {criteria.max_dup_rate:.2f}")
    notes.append("manual: confirm no fake slipped through on the labelled sample")

    return ShadowReport(
        publications=count,
        days_running=days_running,
        stoplist_violations=stoplist_violations,
        dup_rate=dup_rate,
        auto_criteria_met=auto_met,
        notes=notes,
    )
