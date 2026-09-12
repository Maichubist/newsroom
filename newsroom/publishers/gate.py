"""Publish gate — the last line before a post goes out (architecture §10).

Everything that could stop a draft from being published lives here, so the
decision is auditable and testable in one place:

  * the stop button — a full halt stored in system_state (§10);
  * charter hard rules (CLAUDE.md) — the stop-list must pass, the critic must
    have passed, war/defense topics need an official source, rumors must be
    labelled and never run in a critical topic;
  * rate limits — urgent posts per hour, rumors per day (§10);
  * surge detection — a spike of same-rubric posts in a short window reads as a
    possible information attack and is held (§10).

evaluate_gate is pure and collects *all* failing reasons (the supervisor should
see everything, not just the first). When in doubt we do not publish (§3.5).
The stop-button helpers read/write system_state; counts come from the DB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

STOP_KEY = "publish_stopped"


class LimitsConfigError(ValueError):
    """Raised when limits.yaml is malformed."""


@dataclass(frozen=True)
class Limits:
    urgent_per_hour: int = 6
    rumors_per_day: int = 8
    surge_window_minutes: int = 30
    surge_max_same_rubric: int = 5
    regular_per_hour: int = 8          # normal posts/hour, paced evenly (0 = no pacing)


def load_limits(path: str | Path) -> Limits:
    path = Path(path)
    if not path.exists():
        raise LimitsConfigError(f"limits config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return Limits(
            urgent_per_hour=int(data.get("urgent_per_hour", 6)),
            rumors_per_day=int(data.get("rumors_per_day", 8)),
            surge_window_minutes=int(data.get("surge_window_minutes", 30)),
            surge_max_same_rubric=int(data.get("surge_max_same_rubric", 5)),
            regular_per_hour=int(data.get("regular_per_hour", 8)),
        )
    except (TypeError, ValueError) as exc:
        raise LimitsConfigError(f"bad limits value: {exc}") from exc


@dataclass(frozen=True)
class GateInputs:
    stopped: bool = False
    critic_ok: bool = True
    stoplist_blocked: bool = False
    has_official_source: bool = False
    is_rumor: bool = False
    rumor_labeled: bool = False
    risk_level: str | None = None          # critical | high | low
    urgent_last_hour: int = 0
    rumors_last_day: int = 0
    surge_same_rubric: int = 0
    seconds_since_last_regular: float | None = None  # since the last normal post (None = none yet)


@dataclass(frozen=True)
class GateDecision:
    allow: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_gate(inputs: GateInputs, limits: Limits) -> GateDecision:
    """Pure decision. Returns allow=False with every reason it failed, or
    allow=True with no reasons."""
    reasons: list[str] = []
    critical = inputs.risk_level == "critical"

    # --- full halt / charter hard rules (block unconditionally) ---
    if inputs.stopped:
        reasons.append("stop_button")
    if not inputs.critic_ok:
        reasons.append("critic_failed")
    if inputs.stoplist_blocked:
        reasons.append("stoplist")
    if critical and not inputs.has_official_source:
        reasons.append("critical_no_official")     # war/defense: official source required
    if inputs.is_rumor:
        if critical:
            reasons.append("rumor_in_critical_topic")   # rumors never run in critical topics
        elif not inputs.rumor_labeled:
            reasons.append("rumor_unlabeled")           # charter 3.7: label required

    # --- rate limits (hold) ---
    if critical and inputs.urgent_last_hour >= limits.urgent_per_hour:
        reasons.append("urgent_rate_limit")
    if inputs.is_rumor and inputs.rumors_last_day >= limits.rumors_per_day:
        reasons.append("rumor_rate_limit")
    # general firehose control: PACE normal posts evenly across the hour instead of
    # publishing the first N then stopping. One normal post per (3600 / regular_per_hour)
    # seconds; because publishing is ordered by significance, each freed slot takes the
    # most-significant pending draft, so important news arriving later still gets the
    # next slot rather than being locked out. Urgent/critical and rumors are never
    # paced here (they have their own limits), so breaking news is never held by it.
    regular = not critical and not inputs.is_rumor
    if (regular and limits.regular_per_hour > 0 and inputs.seconds_since_last_regular is not None
            and inputs.seconds_since_last_regular < 3600.0 / limits.regular_per_hour):
        reasons.append("post_pacing")

    # --- surge / possible attack (hold) ---
    if inputs.surge_same_rubric >= limits.surge_max_same_rubric:
        reasons.append("surge")

    return GateDecision(allow=not reasons, reasons=sorted(set(reasons)))


# --- stop button (system_state) ------------------------------------------------

def is_publishing_stopped(session) -> bool:
    from newsroom.models import SystemState

    row = session.get(SystemState, STOP_KEY)
    return bool(row and isinstance(row.value, dict) and row.value.get("stopped"))


def set_publishing_stopped(session, stopped: bool, *, reason: str | None = None) -> None:
    """Set/clear the stop button. Flushes; caller commits."""
    from newsroom.models import SystemState

    row = session.get(SystemState, STOP_KEY)
    value = {"stopped": bool(stopped), "reason": reason}
    if row is None:
        session.add(SystemState(key=STOP_KEY, value=value))
    else:
        row.value = value
    session.flush()
