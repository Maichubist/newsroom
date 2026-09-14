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

def test_critical_reads_like_high_no_official_requirement():
    # charter v0.3: no official-source gate. Critical corroborates like high — the label
    # follows the evidence, and a single weak source is 'signal' (waits for a second).
    assert decide("critical", independent_sources=1).publishable is False
    assert decide("critical", independent_sources=1).status == STATUS_SIGNAL
    two = decide("critical", independent_sources=2)
    assert two.publishable and two.status == STATUS_REPORTED
    off = decide("critical", has_official=True)
    assert off.publishable and off.status == STATUS_CONFIRMED
    first = decide("critical", has_first_source=True)
    assert first.publishable and first.status == STATUS_CONFIRMED


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


def test_rumor_is_a_label_not_a_block():
    # charter v0.3: rumor no longer blocks anywhere (incl. critical); it only sets the label
    at_critical = decide("critical", is_rumor=True)
    assert at_critical.publishable and at_critical.status == STATUS_RUMOR
    below = decide("high", is_rumor=True)
    assert below.publishable and below.status == STATUS_RUMOR


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        decide("apocalyptic")
