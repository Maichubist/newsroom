from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from newsroom.analyze.risk import (
    STATUS_CONFIRMED,
    STATUS_REPORTED,
    STATUS_RUMOR,
    STATUS_SIGNAL,
    RiskConfigError,
    decide,
    load_risk_matrix,
)

CONFIG = Path(__file__).resolve().parents[1] / "config" / "risk.yaml"


# --- config / rubric -> level --------------------------------------------------

def test_real_risk_config_loads():
    m = load_risk_matrix(CONFIG)
    assert m.level_for(["war"]) == "critical"
    assert m.level_for(["politics"]) == "high"
    assert m.level_for(["economy"]) == "low"


def test_unknown_rubric_uses_conservative_default():
    m = load_risk_matrix(CONFIG)
    assert m.level_for(["astrology"]) == "high"   # default_level
    assert m.level_for([]) == "high"


def test_multiple_rubrics_resolve_to_highest():
    m = load_risk_matrix(CONFIG)
    assert m.level_for(["economy", "war"]) == "critical"
    assert m.level_for(["sport", "politics"]) == "high"


def test_invalid_config_rejected(tmp_path):
    p = tmp_path / "risk.yaml"
    p.write_text(textwrap.dedent("""
        levels:
          extreme: [war]
    """), encoding="utf-8")
    with pytest.raises(RiskConfigError):
        load_risk_matrix(p)


# --- gate decision -------------------------------------------------------------

def test_critical_requires_official():
    assert decide("critical", independent_sources=5).publishable is False
    assert decide("critical", independent_sources=5).status == STATUS_SIGNAL
    ok = decide("critical", has_official=True)
    assert ok.publishable and ok.status == STATUS_CONFIRMED


def test_high_needs_two_independent_or_first_source():
    assert decide("high", independent_sources=1).publishable is False
    two = decide("high", independent_sources=2)
    assert two.publishable and two.status == STATUS_REPORTED
    first = decide("high", independent_sources=1, has_first_source=True)
    assert first.publishable and first.status == STATUS_CONFIRMED


def test_low_needs_high_reputation():
    assert decide("low", independent_sources=1).publishable is False
    rep = decide("low", high_reputation=True)
    assert rep.publishable and rep.status == STATUS_REPORTED


def test_rumor_blocked_at_critical_allowed_below():
    blocked = decide("critical", is_rumor=True)
    assert blocked.publishable is False and blocked.status == STATUS_SIGNAL
    allowed = decide("high", is_rumor=True)
    assert allowed.publishable and allowed.status == STATUS_RUMOR


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        decide("apocalyptic")
