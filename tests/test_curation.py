from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.orm import Session

from newsroom.editorial.curation import (
    Candidate,
    curate_pending,
    must_publish,
    parse_ranking,
)


# --- must_publish (offline) ---------------------------------------------------

def test_must_publish_only_refutation():
    # refutation (a correction) always goes out without the ranker...
    assert must_publish(update_type="refutation") is True


def test_must_publish_critical_now_goes_to_ranker():
    # the critical+official auto-publish was removed: breaking critical news is now judged
    # by the editorial ranker, not force-published
    assert must_publish(risk_level="critical", has_official_source=True) is False
    assert must_publish(risk_level="critical", has_official_source=False) is False


def test_must_publish_ordinary_is_not():
    assert must_publish(risk_level="high", has_official_source=False) is False
    assert must_publish(risk_level="low", has_official_source=True) is False
    assert must_publish(update_type="new_fact") is False


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
def test_critical_events_go_through_the_ranker(pg_engine):
    # critical no longer auto-publishes: BOTH a content-free individual memorial and a
    # genuine breaking strike are judged by the editor, which holds the memorial and
    # publishes the breaking one.
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

    ranker = FakeRanker({mem_id: "hold", break_id: "publish"})
    curate_pending(sf, ranker, significance_threshold=0.55)

    assert mem_id in ranker.seen and break_id in ranker.seen   # both went through the editor
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
def test_attack_reserved_to_digest_beats_must_publish(pg_engine):
    # reserve_attacks runs before curate in the tick -> a critical+official attack is
    # reserved to the digest, NOT must-published individually.
    from pathlib import Path

    from newsroom.db import make_session_factory
    from newsroom.editorial import load_digest_config, reserve_attacks
    from newsroom.models import Event, EventItem, Item, Source

    cfg = load_digest_config(Path(__file__).resolve().parents[1] / "config" / "digest.yaml")
    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        official = Source(kind="telegram", handle_or_url="@ps", name="ПС", origin="ua",
                          tier="official", is_official=True)
        s.add(official)
        s.flush()
        ev = Event(status="confirmed", risk_level="critical", rubric="war",
                   title="Атака дронів на Київщині: пошкоджено склади", significance=0.9, first_seen_at=now)
        s.add(ev)
        s.flush()
        it = Item(source_id=official.id, external_id="atk1", content_hash="atk1".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id, role="official"))
        eid = ev.id
        s.commit()

    reserve_attacks(sf, cfg)                       # tick step 1
    ranker = FakeRanker({})
    curate_pending(sf, ranker, significance_threshold=0.55)   # tick step 2
    with Session(pg_engine) as s:
        assert s.get(Event, eid).curated == "digest"   # reserved, not must-published
    assert eid not in ranker.seen


@pytest.mark.pg
def test_curate_pending_attaches_topic_heat_to_ranker(pg_engine):
    from newsroom.analyze.topics import store_hot_topics
    from newsroom.db import make_session_factory
    from newsroom.models import Event

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        ev = Event(status="confirmed", risk_level="low", rubric="economy", title="Подія",
                   significance=0.8, keywords=["дрон", "покровськ"], first_seen_at=now)
        s.add(ev)
        s.flush()
        eid = ev.id
        store_hot_topics(s, {"heat": {"дрон": 0.9}, "topics": [], "at": now.isoformat()})
        s.commit()

    ranker = FakeRanker({eid: "publish"})
    curate_pending(sf, ranker, significance_threshold=0.55, window_hours=6)
    assert ranker.candidates and ranker.candidates[0].heat == pytest.approx(0.9)   # hottest keyword


@pytest.mark.pg
def test_curate_pending_refutation_bypasses_ranker_and_ranks_the_rest(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, EventItem, Item, Source

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        official = Source(kind="telegram", handle_or_url="@gs", name="Генштаб", origin="ua",
                          tier="official", is_official=True)
        s.add(official)
        s.flush()
        # a refutation (correction) -> must-publish, never reaches the ranker
        e_must = Event(status="confirmed", risk_level="high", rubric="politics", title="Спростування",
                       update_type="refutation", significance=0.8, first_seen_at=now)
        # a critical event is NO LONGER auto-published: it goes to the ranker like the rest
        e_crit = Event(status="confirmed", risk_level="critical", rubric="war", title="Удар",
                       significance=0.9, first_seen_at=now)
        e_pub = Event(status="confirmed", risk_level="low", rubric="economy", title="Курс",
                      significance=0.8, first_seen_at=now)
        e_hold = Event(status="confirmed", risk_level="low", rubric="sport", title="Матч",
                       significance=0.6, first_seen_at=now)
        # below the significance bar -> not even considered
        e_low = Event(status="confirmed", risk_level="low", rubric="culture", title="Дрібниця",
                      significance=0.2, first_seen_at=now)
        s.add_all([e_must, e_crit, e_pub, e_hold, e_low])
        s.flush()
        it = Item(source_id=official.id, external_id="o1", content_hash="o1".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        s.add(EventItem(event_id=e_crit.id, item_id=it.id, role="official"))
        ids = {"must": e_must.id, "crit": e_crit.id, "pub": e_pub.id, "hold": e_hold.id, "low": e_low.id}
        s.commit()

    ranker = FakeRanker({ids["crit"]: "publish", ids["pub"]: "publish", ids["hold"]: "hold"})
    stats = curate_pending(sf, ranker, significance_threshold=0.55, window_hours=6)

    assert stats["must"] == 1 and stats["publish"] == 3 and stats["hold"] == 1
    assert ids["must"] not in ranker.seen          # the refutation never reached the LLM
    assert set(ranker.seen) == {ids["crit"], ids["pub"], ids["hold"]}   # critical is ranked now
    with Session(pg_engine) as s:
        from newsroom.models import Event
        curated = {eid: s.get(Event, eid).curated for eid in ids.values()}
        assert curated == {ids["must"]: "publish", ids["crit"]: "publish", ids["pub"]: "publish",
                           ids["hold"]: "hold", ids["low"]: None}

    # idempotent: already curated -> nothing to do
    assert curate_pending(sf, FakeRanker({}), significance_threshold=0.55)["curated"] == 0
