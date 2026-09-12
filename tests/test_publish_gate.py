from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from newsroom.publishers.gate import (
    GateInputs,
    Limits,
    evaluate_gate,
    is_publishing_stopped,
    load_limits,
    set_publishing_stopped,
)

CONFIG = Path(__file__).resolve().parents[1] / "config"
LIMITS = Limits(urgent_per_hour=6, rumors_per_day=8, surge_window_minutes=30, surge_max_same_rubric=5)


# --- load_limits (offline) ----------------------------------------------------

def test_load_limits_from_config():
    lim = load_limits(CONFIG / "limits.yaml")
    assert lim.urgent_per_hour >= 1 and lim.rumors_per_day >= 1 and lim.surge_max_same_rubric >= 1


# --- evaluate_gate: happy path ------------------------------------------------

def test_gate_allows_clean_low_risk_post():
    d = evaluate_gate(GateInputs(critic_ok=True, risk_level="low"), LIMITS)
    assert d.allow is True and d.reasons == []


def test_gate_allows_confirmed_critical_with_official_source():
    d = evaluate_gate(GateInputs(critic_ok=True, risk_level="critical", has_official_source=True), LIMITS)
    assert d.allow is True


# --- evaluate_gate: hard blocks -----------------------------------------------

def test_gate_stop_button_blocks():
    assert "stop_button" in evaluate_gate(GateInputs(stopped=True, risk_level="low"), LIMITS).reasons


def test_gate_critic_and_stoplist_block():
    d = evaluate_gate(GateInputs(critic_ok=False, stoplist_blocked=True, risk_level="low"), LIMITS)
    assert not d.allow and {"critic_failed", "stoplist"} <= set(d.reasons)


def test_gate_critical_requires_official_source():
    d = evaluate_gate(GateInputs(risk_level="critical", has_official_source=False), LIMITS)
    assert "critical_no_official" in d.reasons


def test_gate_rumor_must_be_labeled():
    assert "rumor_unlabeled" in evaluate_gate(
        GateInputs(risk_level="low", is_rumor=True, rumor_labeled=False), LIMITS).reasons
    # labeled rumor in a non-critical topic is fine
    assert evaluate_gate(
        GateInputs(risk_level="low", is_rumor=True, rumor_labeled=True), LIMITS).allow is True


def test_gate_rumor_never_in_critical_topic():
    d = evaluate_gate(GateInputs(risk_level="critical", has_official_source=True,
                                 is_rumor=True, rumor_labeled=True), LIMITS)
    assert "rumor_in_critical_topic" in d.reasons and not d.allow


# --- evaluate_gate: limits + surge --------------------------------------------

def test_gate_urgent_rate_limit():
    ok = evaluate_gate(GateInputs(risk_level="critical", has_official_source=True, urgent_last_hour=5), LIMITS)
    hit = evaluate_gate(GateInputs(risk_level="critical", has_official_source=True, urgent_last_hour=6), LIMITS)
    assert ok.allow is True and "urgent_rate_limit" in hit.reasons


def test_gate_rumor_rate_limit():
    hit = evaluate_gate(GateInputs(risk_level="low", is_rumor=True, rumor_labeled=True,
                                   rumors_last_day=8), LIMITS)
    assert "rumor_rate_limit" in hit.reasons


def test_gate_surge_detection():
    hit = evaluate_gate(GateInputs(risk_level="high", surge_same_rubric=5), LIMITS)
    assert "surge" in hit.reasons and not hit.allow


def test_gate_has_no_rate_pacing():
    # output volume is decided by editorial curation, not a per-post rate: a clean
    # low-risk post always passes the gate regardless of how recently we posted
    for _ in range(20):
        assert evaluate_gate(GateInputs(critic_ok=True, risk_level="low"), LIMITS).allow is True


def test_gate_reasons_are_deduped_and_sorted():
    d = evaluate_gate(GateInputs(stopped=True, critic_ok=False, stoplist_blocked=True, risk_level="low"), LIMITS)
    assert d.reasons == sorted(d.reasons) and len(d.reasons) == len(set(d.reasons))


# --- stop button (pg) ---------------------------------------------------------

@pytest.mark.pg
def test_stop_button_roundtrip(pg_engine):
    with Session(pg_engine) as s:
        assert is_publishing_stopped(s) is False          # unset -> not stopped
        set_publishing_stopped(s, True, reason="manual halt")
        s.commit()
    with Session(pg_engine) as s:
        assert is_publishing_stopped(s) is True
        set_publishing_stopped(s, False)
        s.commit()
    with Session(pg_engine) as s:
        assert is_publishing_stopped(s) is False
