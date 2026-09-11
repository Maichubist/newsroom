from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.orm import Session

from newsroom.factcheck.verdict import (
    VERDICT_FALSE,
    VERDICT_TRUE,
    VERDICT_UNVERIFIABLE,
    VerdictResult,
    apply_verdict,
    parse_verdict,
)


# --- parse_verdict (offline) --------------------------------------------------

def test_parse_verdict_full():
    r = parse_verdict('{"verdict": "true", "confidence": 0.9, "explanation": "два джерела",'
                      ' "stances": ["supports", "neutral"]}', evidence_count=2)
    assert r.verdict == VERDICT_TRUE and r.confidence == 0.9
    assert r.explanation == "два джерела" and r.stances == ["supports", "neutral"]


def test_parse_verdict_unknown_verdict_and_stance_collapse():
    r = parse_verdict('{"verdict": "definitely-fake", "confidence": 2, "stances": ["yes", "refutes"]}',
                      evidence_count=2)
    assert r.verdict == VERDICT_UNVERIFIABLE     # unknown -> conservative
    assert r.confidence == 1.0                    # clamped to [0,1]
    assert r.stances == ["neutral", "refutes"]    # unknown stance -> neutral


def test_parse_verdict_aligns_stances_to_evidence_count():
    # fewer stances than evidence -> padded neutral; more -> truncated
    assert parse_verdict('{"verdict": "false", "stances": ["refutes"]}', evidence_count=3).stances == \
        ["refutes", "neutral", "neutral"]
    assert parse_verdict('{"verdict": "false", "stances": ["refutes", "refutes", "refutes"]}',
                         evidence_count=1).stances == ["refutes"]


def test_parse_verdict_invalid_is_none():
    assert parse_verdict("not json") is None
    assert parse_verdict(None) is None
    assert parse_verdict("[1, 2, 3]") is None     # not an object


# --- apply_verdict (pg) -------------------------------------------------------

@pytest.mark.pg
def test_apply_verdict_writes_claim_and_stances(pg_engine):
    from newsroom.models import Claim, ClaimEvidence, Event

    with Session(pg_engine) as s:
        ev = Event(status="reported", title="Подія", first_seen_at=dt.datetime.now(dt.timezone.utc))
        s.add(ev)
        s.flush()
        claim = Claim(event_id=ev.id, text="твердження")
        s.add(claim)
        s.flush()
        e1 = ClaimEvidence(claim_id=claim.id, evidence_kind="corpus", stance="neutral")
        e2 = ClaimEvidence(claim_id=claim.id, evidence_kind="factcheck_db", stance="neutral",
                           external_ref="https://voxcheck.org/x")
        s.add_all([e1, e2])
        s.flush()

        apply_verdict(s, claim.id, [e1.id, e2.id],
                      VerdictResult(verdict=VERDICT_FALSE, confidence=0.8, explanation="спростовано",
                                    stances=["neutral", "refutes"]))
        s.commit()

        got = s.get(Claim, claim.id)
        assert got.verdict == VERDICT_FALSE and got.confidence == 0.8
        assert s.get(ClaimEvidence, e1.id).stance == "neutral"
        assert s.get(ClaimEvidence, e2.id).stance == "refutes"
