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
    # at most one post per story within this window — a fast-breaking story with many
    # sources/updates otherwise spams near-duplicate posts (refutations are exempt).
    story_cooldown_minutes: int = 60
    # hold a new event this many minutes before it may publish, so cross-source twins
    # arrive and get merged (dedup + story-merge) before the first one goes out. This is
    # enforced in the publish query, not evaluate_gate (refutations are exempt).
    publish_debounce_minutes: int = 6


def load_limits(path: str | Path) -> Limits:
    path = Path(path)
    if not path.exists():
        raise LimitsConfigError(f"limits config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return Limits(
            story_cooldown_minutes=int(data.get("story_cooldown_minutes", 60)),
            publish_debounce_minutes=int(data.get("publish_debounce_minutes", 6)),
        )
    except (TypeError, ValueError) as exc:
        raise LimitsConfigError(f"bad limits value: {exc}") from exc


@dataclass(frozen=True)
class GateInputs:
    stopped: bool = False
    critic_ok: bool = True
    stoplist_blocked: bool = False
    story_recent_posts: int = 0             # posts already out for this event's story in the cooldown window
    is_refutation: bool = False             # a correction always goes out (exempt from the cooldown)


@dataclass(frozen=True)
class GateDecision:
    allow: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_gate(inputs: GateInputs, limits: Limits) -> GateDecision:
    """Pure decision (charter v0.3). Returns allow=False with every reason it failed, or
    allow=True with no reasons.

    What to publish — and how much — is decided by popularity/editorial curation (charter
    §3), NOT by rate quotas: the old urgent-per-hour and same-rubric surge caps were
    removed because on a war-dominant channel they throttled legitimate coverage and read
    as "the bot stopped posting". What remains: the floor (manual halt, the critic which
    enforces OPSEC/ethics/AI-slop, and the OPSEC stop-list) and the dedup cooldown (one
    post per story; refutations exempt). Volume is bounded upstream by how selectively the
    ranker marks events publish, plus the debounce and dedup."""
    reasons: list[str] = []

    # --- floor (block unconditionally) ---
    if inputs.stopped:
        reasons.append("stop_button")
    if not inputs.critic_ok:
        reasons.append("critic_failed")
    if inputs.stoplist_blocked:
        reasons.append("stoplist")        # OPSEC/legal stop-list (charter §4)

    # --- one post per story per cooldown (hold) — a breaking story otherwise spams
    #     near-duplicate posts; refutations/corrections are exempt ---
    if inputs.story_recent_posts >= 1 and not inputs.is_refutation:
        reasons.append("story_cooldown")

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
