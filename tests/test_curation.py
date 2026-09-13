from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.orm import Session

from newsroom.editorial.curation import (
    Candidate,
    _is_memorial,
    curate_pending,
    must_publish,
    parse_ranking,
)


def test_is_memorial_detects_individual_loss():
    assert _is_memorial("На Сумщині загинув військовий Володимир Гринь") is True
    assert _is_memorial("Прощання з Героєм у Львові") is True
    assert _is_memorial("на щиті повернувся боєць") is True
    # mass-casualty / general war news is NOT an individual memorial
    assert _is_memorial("Унаслідок удару по Краматорську загинули двоє людей") is False
    assert _is_memorial("Зеленський підписав указ") is False


# --- must_publish (offline) ---------------------------------------------------

def test_must_publish_critical_with_official():
    assert must_publish(risk_level="critical", has_official_source=True) is True
    assert must_publish(risk_level="critical", has_official_source=False) is False  # needs official


def test_must_publish_refutation_always():
    assert must_publish(risk_level="low", has_official_source=False, update_type="refutation") is True


def test_must_publish_ordinary_is_not():
    assert must_publish(risk_level="high", has_official_source=False) is False
    assert must_publish(risk_level="low", has_official_source=True) is False


# --- parse_ranking (offline) --------------------------------------------------

def test_parse_ranking_maps_decisions():
    raw = '{"decisions": [{"id": 1, "decision": "publish"}, {"id": 2, "decision": "hold"}]}'
    assert parse_ranking(raw, {1, 2}) == {1: "publish", 2: "hold"}


def test_parse_ranking_missing_candidate_defaults_hold():
    raw = '{"decisions": [{"id": 1, "decision": "publish"}]}'
    assert parse_ranking(raw, {1, 2, 3}) == {1: "publish", 2: "hold", 3: "hold"}


def test_parse_ranking_ignores_unknown_ids_and_bad_json():
    assert parse_ranking('{"decisions": [{"id": 99, "decision": "publish"}]}', {1}) == {1: "hold"}
    assert parse_ranking("garbage", {1, 2}) == {1: "hold", 2: "hold"}


# --- curate_pending (pg) ------------------------------------------------------

class FakeRanker:
    model = "fake-rank"

    def __init__(self, decisions):
        self.decisions = decisions
        self.seen: list[int] = []
        self.candidates: list = []

    def rank(self, candidates):
        self.seen = [c.event_id for c in candidates]
        self.candidates = list(candidates)
        return {c.event_id: self.decisions.get(c.event_id, "hold") for c in candidates}


@pytest.mark.pg
def test_memorial_routed_to_ranker_not_must_publish(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, EventItem, Item, Source

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        official = Source(kind="telegram", handle_or_url="@mil", name="Військові", origin="ua",
                          tier="official", is_official=True)
        s.add(official)
        s.flush()
        mem = Event(status="confirmed", risk_level="critical", rubric="war",
                    title="На Сумщині загинув військовий Володимир Гринь", significance=0.9, first_seen_at=now)
        breaking = Event(status="confirmed", risk_level="critical", rubric="war",
                         title="Масований ракетний удар по Києву, працює ППО", significance=0.9, first_seen_at=now)
        s.add_all([mem, breaking])
        s.flush()
        for ev in (mem, breaking):
            it = Item(source_id=official.id, external_id=f"o{ev.id}", content_hash=str(ev.id).ljust(64, "0"), title="t")
            s.add(it)
            s.flush()
            s.add(EventItem(event_id=ev.id, item_id=it.id, role="official"))
        mem_id, break_id = mem.id, breaking.id
        s.commit()

    ranker = FakeRanker({mem_id: "hold"})     # editor holds the individual memorial
    curate_pending(sf, ranker, significance_threshold=0.55)

    assert mem_id in ranker.seen              # memorial went through the editor...
    assert break_id not in ranker.seen        # ...breaking critical+official is must-publish
    with Session(pg_engine) as s:
        assert s.get(Event, mem_id).curated == "hold"
        assert s.get(Event, break_id).curated == "publish"


@pytest.mark.pg
def test_curate_pending_passes_learned_demand_to_ranker(pg_engine):
    from newsroom.analyze.demand import store_demand
    from newsroom.db import make_session_factory
    from newsroom.models import Event

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        ev = Event(status="confirmed", risk_level="low", rubric="politics", title="Подія",
                   significance=0.8, first_seen_at=now)
        s.add(ev)
        s.flush()
        eid = ev.id
        store_demand(s, {"politics": 0.9, "sport": 0.1})   # learned demand index
        s.commit()

    ranker = FakeRanker({eid: "publish"})
    curate_pending(sf, ranker, significance_threshold=0.55, window_hours=6)
    assert ranker.candidates and ranker.candidates[0].demand == pytest.approx(0.9)   # demand attached


@pytest.mark.pg
def test_curate_pending_must_publish_bypasses_ranker_and_ranks_the_rest(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, EventItem, Item, Source

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        official = Source(kind="telegram", handle_or_url="@gs", name="Генштаб", origin="ua",
                          tier="official", is_official=True)
        s.add(official)
        s.flush()
        # critical + official -> must-publish (no ranker)
        e_must = Event(status="confirmed", risk_level="critical", rubric="war", title="Удар",
                       significance=0.9, first_seen_at=now)
        # significant, low-risk -> goes to the ranker
        e_pub = Event(status="confirmed", risk_level="low", rubric="economy", title="Курс",
                      significance=0.8, first_seen_at=now)
        e_hold = Event(status="confirmed", risk_level="low", rubric="sport", title="Матч",
                       significance=0.6, first_seen_at=now)
        # below the significance bar -> not even considered
        e_low = Event(status="confirmed", risk_level="low", rubric="culture", title="Дрібниця",
                      significance=0.2, first_seen_at=now)
        s.add_all([e_must, e_pub, e_hold, e_low])
        s.flush()
        it = Item(source_id=official.id, external_id="o1", content_hash="o1".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        s.add(EventItem(event_id=e_must.id, item_id=it.id, role="official"))
        ids = {"must": e_must.id, "pub": e_pub.id, "hold": e_hold.id, "low": e_low.id}
        s.commit()

    ranker = FakeRanker({ids["pub"]: "publish", ids["hold"]: "hold"})
    stats = curate_pending(sf, ranker, significance_threshold=0.55, window_hours=6)

    assert stats["must"] == 1 and stats["publish"] == 2 and stats["hold"] == 1
    assert ids["must"] not in ranker.seen          # must-publish never reached the LLM
    assert set(ranker.seen) == {ids["pub"], ids["hold"]}
    with Session(pg_engine) as s:
        from newsroom.models import Event
        curated = {eid: s.get(Event, eid).curated for eid in ids.values()}
        assert curated == {ids["must"]: "publish", ids["pub"]: "publish",
                           ids["hold"]: "hold", ids["low"]: None}

    # idempotent: already curated -> nothing to do
    assert curate_pending(sf, FakeRanker({}), significance_threshold=0.55)["curated"] == 0
