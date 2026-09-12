from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsroom.factcheck import FactChecker, check_pending
from newsroom.factcheck.claims import ClaimDraft
from newsroom.factcheck.evidence import EvidenceRef
from newsroom.factcheck.verdict import VerdictResult

pytestmark = pytest.mark.pg


class FakeExtractor:
    model = "fake-extract"

    def extract(self, title, text):
        return [ClaimDraft("НБУ знизив ставку", "what"), ClaimDraft("ставка 13%", "number")]


class FakeSearcher:
    """Returns one corpus ref (a real item id) plus one external ref."""

    def __init__(self, item_id: int):
        self.item_id = item_id

    def search(self, claim_text, *, exclude_item_ids=()):
        assert self.item_id not in set(exclude_item_ids)  # own items excluded
        return [
            EvidenceRef(stance="neutral", evidence_kind="corpus", item_id=self.item_id, score=0.9),
            EvidenceRef(stance="neutral", evidence_kind="factcheck_db",
                        external_ref="https://voxcheck.org/x"),
        ]


class FakeJudge:
    model = "fake-judge"

    def judge(self, claim_text, evidence_snippets):
        # stance per evidence: first supports, rest neutral
        stances = ["supports"] + ["neutral"] * (len(evidence_snippets) - 1)
        return VerdictResult(verdict="true", confidence=0.7, explanation="ок", stances=stances)


def _seed(pg_engine):
    from newsroom.models import Event, EventItem, Item, Source

    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://fc.example/feed", name="FC",
                     origin="ua", tier="media")
        s.add(src)
        s.flush()
        own = Item(source_id=src.id, external_id="own", content_hash="o1", title="Подія",
                   text="НБУ знизив облікову ставку до 13%.")
        evi = Item(source_id=src.id, external_id="evi", content_hash="e1", title="Довідка",
                   text="Рішення НБУ підтверджує пресреліз.")
        s.add_all([own, evi])
        s.flush()
        ev = Event(status="confirmed", title="НБУ знизив ставку",
                   first_seen_at=dt.datetime.now(dt.timezone.utc))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=own.id, role="origin", similarity=1.0))
        s.commit()
        return ev.id, own.id, evi.id


def test_check_event_extracts_evidence_and_verdict(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Claim, ClaimEvidence

    sf = make_session_factory(pg_engine)
    event_id, own_id, evi_id = _seed(pg_engine)

    checker = FactChecker(sf, extractor=FakeExtractor(), searcher=FakeSearcher(evi_id), judge=FakeJudge())
    result = checker.check_event(event_id)
    assert result.extracted == 2 and result.checked == 2 and not result.skipped

    with Session(pg_engine) as s:
        claims = s.execute(select(Claim).where(Claim.event_id == event_id).order_by(Claim.id)).scalars().all()
        assert [c.text for c in claims] == ["НБУ знизив ставку", "ставка 13%"]
        assert all(c.verdict == "true" and c.confidence == 0.7 for c in claims)
        # each claim has 2 evidence rows; the corpus one got stance 'supports'
        for c in claims:
            evs = s.execute(select(ClaimEvidence).where(ClaimEvidence.claim_id == c.id)
                            .order_by(ClaimEvidence.id)).scalars().all()
            assert len(evs) == 2
            assert evs[0].evidence_kind == "corpus" and evs[0].stance == "supports"
            assert evs[1].evidence_kind == "factcheck_db"

    # idempotent: event already has claims -> skipped, no duplicates
    again = checker.check_event(event_id)
    assert again.skipped
    with Session(pg_engine) as s:
        assert s.scalar(select(func.count()).select_from(Claim).where(Claim.event_id == event_id)) == 2


def test_check_pending_selects_publishable_without_claims(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event

    sf = make_session_factory(pg_engine)
    event_id, own_id, evi_id = _seed(pg_engine)
    # a signal event should NOT be picked up
    with Session(pg_engine) as s:
        sig = Event(status="signal", title="Сигнал", first_seen_at=dt.datetime.now(dt.timezone.utc))
        s.add(sig)
        s.commit()
        sig_id = sig.id

    checker = FactChecker(sf, extractor=FakeExtractor(), searcher=FakeSearcher(evi_id), judge=FakeJudge())
    stats = check_pending(sf, checker, limit=50)
    assert stats["events"] == 1 and stats["claims"] == 2 and stats["checked"] == 2

    from newsroom.models import Claim
    with Session(pg_engine) as s:
        assert s.scalar(select(func.count()).select_from(Claim).where(Claim.event_id == sig_id)) == 0


def test_check_pending_high_only_skips_low_risk(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Claim, Event

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        from newsroom.models import Item, Source
        src = Source(kind="rss", handle_or_url="https://r.example/feed", name="R", origin="ua", tier="media")
        s.add(src)
        s.flush()
        evi = Item(source_id=src.id, external_id="rev", content_hash="r1", title="Довідка", text="Пруф.")
        s.add(evi)
        s.flush()
        evi_id = evi.id
        low = Event(status="confirmed", risk_level="low", title="Економіка", first_seen_at=now)
        high = Event(status="confirmed", risk_level="high", title="Політика", first_seen_at=now)
        unknown = Event(status="confirmed", risk_level=None, title="Невідомо", first_seen_at=now)
        s.add_all([low, high, unknown])
        s.flush()
        low_id, high_id, unknown_id = low.id, high.id, unknown.id
        s.commit()

    checker = FactChecker(sf, extractor=FakeExtractor(), searcher=FakeSearcher(evi_id), judge=FakeJudge())
    check_pending(sf, checker, limit=50, risk_levels=("high", "critical"))

    with Session(pg_engine) as s:
        def n(eid):
            return s.scalar(select(func.count()).select_from(Claim).where(Claim.event_id == eid))
        assert n(high_id) > 0            # high-risk checked
        assert n(unknown_id) > 0         # unknown-risk checked (conservative)
        assert n(low_id) == 0            # low-risk skipped
