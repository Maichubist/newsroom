from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.factcheck.claims import ClaimDraft, parse_claims, store_claims


# --- parse_claims (offline) ----------------------------------------------------

def test_parse_claims_object_form():
    claims = parse_claims('{"claims": [{"text": "НБУ знизив ставку", "claim_type": "What"}, '
                          '{"text": "ставка 13%", "claim_type": "number"}]}')
    assert claims is not None and len(claims) == 2
    assert claims[0].text == "НБУ знизив ставку" and claims[0].claim_type == "what"


def test_parse_claims_bare_list_and_strings():
    claims = parse_claims('["перше твердження", {"text": "друге"}]')
    assert [c.text for c in claims] == ["перше твердження", "друге"]


def test_parse_claims_invalid_is_none():
    assert parse_claims("not json") is None
    assert parse_claims(None) is None
    assert parse_claims('{"claims": "нема списку"}') is None


def test_parse_claims_empty_list():
    assert parse_claims('{"claims": []}') == []


# --- store_claims (pg) ---------------------------------------------------------

@pytest.mark.pg
def test_store_claims_persists(pg_engine):
    import datetime as dt

    from newsroom.models import Claim, Event

    with Session(pg_engine) as s:
        ev = Event(status="confirmed", title="Подія", first_seen_at=dt.datetime.now(dt.timezone.utc))
        s.add(ev)
        s.flush()
        ids = store_claims(s, ev.id, [ClaimDraft("твердження 1", "what"), ClaimDraft("твердження 2")])
        s.commit()

        assert len(ids) == 2
        rows = s.execute(select(Claim).where(Claim.event_id == ev.id).order_by(Claim.id)).scalars().all()
        assert [r.text for r in rows] == ["твердження 1", "твердження 2"]
        assert rows[0].claim_type == "what" and rows[1].claim_type is None
