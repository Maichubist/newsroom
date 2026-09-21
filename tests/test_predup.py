from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.publishers.predup import (
    ACTION_DUPLICATE,
    ACTION_HOLD_REVIEW,
    ACTION_SEPARATE,
    ACTION_UPDATE,
    CLASS_AUTO_DUPLICATE,
    CLASS_GREY,
    CLASS_SEPARATE,
    CandidateSignals,
    EventSig,
    ItemSig,
    PredupConfig,
    PrepublishDedup,
    TwinJudgment,
    candidate_signals,
    classify_signals,
    extract_number_facts,
    find_publish_candidates,
    in_candidate_net,
    load_predup_config,
    parse_twin_judgment,
)

UTC = dt.timezone.utc
CONFIG = Path(__file__).resolve().parents[1] / "config"


# --- config (offline) ---------------------------------------------------------

def test_load_predup_config_real_file():
    cfg = load_predup_config(CONFIG / "dedup.yaml")
    assert cfg.window_hours == 72 and cfg.simhash_max == 6
    assert 0.0 < cfg.vector_candidate < 1.0 and cfg.top_candidates >= 1


def test_load_predup_config_defaults(tmp_path):
    p = tmp_path / "dedup.yaml"
    p.write_text("version: 0.3\npredup:\n  window_hours: 24\n", encoding="utf-8")
    cfg = load_predup_config(p)
    assert cfg.window_hours == 24 and cfg.simhash_max == 6   # missing keys fall back


def test_load_predup_config_missing_file(tmp_path):
    from newsroom.publishers.predup import DedupConfigError

    with pytest.raises(DedupConfigError):
        load_predup_config(tmp_path / "nope.yaml")


# --- candidate_signals (offline) ----------------------------------------------

def _ev(eid, *, centroid=None, items=(), number_facts=()):
    return EventSig(event_id=eid, centroid=centroid, items=tuple(items),
                    number_facts=frozenset(number_facts))


def test_candidate_signals_exact_content_hash():
    a = _ev(1, items=[ItemSig(content_hash="ABC")])
    b = _ev(2, items=[ItemSig(content_hash="ABC")])
    sig = candidate_signals(a, b)
    assert sig.exact_content_hash is True


def test_candidate_signals_forwarded_and_url_and_simhash():
    a = _ev(1, items=[ItemSig(forwarded_from="@Origin", url="https://t.me/x/1", simhash=0)])
    b = _ev(2, items=[ItemSig(forwarded_from="@origin", url="https://t.me/x/1", simhash=1)])
    sig = candidate_signals(a, b)
    assert sig.forwarded_from_match is True          # case-insensitive
    assert sig.url_match is True
    assert sig.simhash_distance == 1


def test_candidate_signals_cosine():
    a = _ev(1, centroid=[1.0, 0.0, 0.0])
    b = _ev(2, centroid=[1.0, 0.0, 0.0])
    c = _ev(3, centroid=[0.0, 1.0, 0.0])
    assert candidate_signals(a, b).cosine == pytest.approx(1.0)
    assert candidate_signals(a, c).cosine == pytest.approx(0.0)


def test_candidate_signals_no_overlap():
    a = _ev(1, items=[ItemSig(content_hash="A", simhash=0, forwarded_from="@x", url="u1")])
    b = _ev(2, items=[ItemSig(content_hash="B", simhash=None, forwarded_from="@y", url="u2")])
    sig = candidate_signals(a, b)
    assert not sig.exact_content_hash and not sig.forwarded_from_match and not sig.url_match
    assert sig.simhash_distance is None              # one side has no simhash


# --- classify_signals (offline) -----------------------------------------------

CFG = PredupConfig(vector_candidate=0.55, simhash_max=6)


def test_classify_only_exact_hash_is_auto_duplicate():
    assert classify_signals(CandidateSignals(exact_content_hash=True), CFG) == CLASS_AUTO_DUPLICATE


# --- fact-fingerprint (offline) -----------------------------------------------

def test_extract_number_facts_typed_and_normalized():
    fp = extract_number_facts(
        "Україна отримала 3,3 млрд євро; зачистили 85 км²; втратили 1,5 тисячі солдатів; 12 тисяч дронів")
    assert {"M3.3", "A85", "T1500", "D12000"} <= fp
    # '1500 солдатів' collapses to the same token as '1,5 тисячі солдатів'
    assert extract_number_facts("1500 солдатів") == frozenset({"T1500"})
    assert extract_number_facts("нема чисел") == frozenset()


def test_number_facts_widen_the_candidate_net():
    a = _ev(1, number_facts={"T1500", "A75"})
    b = _ev(2, number_facts={"T1500", "A75", "M3.3"})
    sig = candidate_signals(a, b)
    assert sig.shared_number_facts == 2
    assert in_candidate_net(sig, CFG) is True            # >=2 shared -> a candidate (LLM decides)
    # one shared number is NOT enough on its own (a lone "1500 втрат" is common)
    c = _ev(3, number_facts={"T1500"})
    assert candidate_signals(a, c).shared_number_facts == 1
    assert in_candidate_net(candidate_signals(a, c), CFG) is False


def test_number_facts_are_grey_not_auto():
    # a fingerprint match is a candidate for the LLM arbiter, never a no-LLM auto-merge
    sig = candidate_signals(_ev(1, number_facts={"T1500", "A75"}), _ev(2, number_facts={"T1500", "A75"}))
    assert classify_signals(sig, CFG) == CLASS_GREY


def test_classify_forwarded_from_is_grey_not_auto():
    # forwarded_from is only the ORIGIN CHANNEL NAME (not a message id): two different posts
    # forwarded from the same source share it, so it must never auto-merge -> arbiter decides
    assert classify_signals(CandidateSignals(forwarded_from_match=True), CFG) == CLASS_GREY


def test_classify_url_only_is_grey_not_auto():
    # a shared URL can be a substantially updated article -> arbiter decides, not auto
    assert classify_signals(CandidateSignals(url_match=True), CFG) == CLASS_GREY


def test_classify_similarity_is_grey():
    assert classify_signals(CandidateSignals(cosine=0.70), CFG) == CLASS_GREY
    assert classify_signals(CandidateSignals(simhash_distance=3), CFG) == CLASS_GREY


def test_classify_below_thresholds_is_separate():
    assert classify_signals(CandidateSignals(cosine=0.40, simhash_distance=20), CFG) == CLASS_SEPARATE
    assert classify_signals(CandidateSignals(), CFG) == CLASS_SEPARATE


def test_in_candidate_net():
    assert in_candidate_net(CandidateSignals(cosine=0.55), CFG)
    assert in_candidate_net(CandidateSignals(simhash_distance=6), CFG)
    assert not in_candidate_net(CandidateSignals(cosine=0.54, simhash_distance=7), CFG)


# --- parse_twin_judgment (offline) --------------------------------------------

def test_parse_twin_judgment_valid():
    j = parse_twin_judgment('{"decision": "duplicate", "confidence": 0.8, "reason": "той самий удар"}')
    assert j.decision == ACTION_DUPLICATE and j.confidence == 0.8 and "удар" in j.reason


def test_parse_twin_judgment_update_and_separate():
    assert parse_twin_judgment('{"decision":"update"}').decision == ACTION_UPDATE
    assert parse_twin_judgment('{"decision":"separate"}').decision == ACTION_SEPARATE


def test_parse_twin_judgment_unknown_or_garbage_is_none():
    assert parse_twin_judgment('{"decision":"maybe"}') is None    # unknown -> None (not silently 'separate')
    assert parse_twin_judgment("not json") is None
    assert parse_twin_judgment(None) is None
    assert parse_twin_judgment("[1,2]") is None


# --- PrepublishDedup / find_publish_candidates (pg) ---------------------------

class FakeJudge:
    model = "fake"

    def __init__(self, decision: str):
        self.decision = decision
        self.calls = 0

    def judge(self, pair):
        self.calls += 1
        return TwinJudgment(decision=self.decision, confidence=0.9, reason="fake")


def _seed_published(session, *, title, content_hash, simhash=None, story_id=None,
                    first_seen=None, handle="https://m/f"):
    from newsroom.models import Event, EventItem, Item, Publication, Source

    src = session.execute(
        select(Source).where(Source.handle_or_url == handle)
    ).scalar_one_or_none()
    if src is None:
        src = Source(kind="rss", handle_or_url=handle, name="M", origin="ua", tier="media")
        session.add(src)
        session.flush()
    ext = f"{content_hash}-{title}"
    it = Item(source_id=src.id, external_id=ext, content_hash=content_hash, simhash=simhash, title=title)
    session.add(it)
    session.flush()
    ev = Event(status="confirmed", title=title, story_id=story_id,
               first_seen_at=first_seen or dt.datetime.now(UTC))
    session.add(ev)
    session.flush()
    session.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
    session.add(Publication(event_id=ev.id, channel="telegram", kind="post", status="published",
                            headline=title, body="b", published_at=dt.datetime.now(UTC),
                            features={"critic_ok": True}))
    session.flush()
    return ev.id


def _seed_incoming(session, *, title, content_hash, simhash=None, handle="https://in/f"):
    from newsroom.models import Event, EventItem, Item, Source

    src = session.execute(
        select(Source).where(Source.handle_or_url == handle)
    ).scalar_one_or_none()
    if src is None:
        src = Source(kind="rss", handle_or_url=handle, name="IN", origin="ua", tier="media")
        session.add(src)
        session.flush()
    it = Item(source_id=src.id, external_id=f"{content_hash}-{title}",
              content_hash=content_hash, simhash=simhash, title=title)
    session.add(it)
    session.flush()
    ev = Event(status="confirmed", title=title, first_seen_at=dt.datetime.now(UTC))
    session.add(ev)
    session.flush()
    session.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
    session.flush()
    return ev.id


@pytest.mark.pg
def test_find_publish_candidates_matches_exact_hash_and_excludes_self(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    cfg = PredupConfig()
    with Session(pg_engine) as s:
        canon = _seed_published(s, title="Наступ ЗСУ", content_hash="H".ljust(64, "0"))
        incoming = _seed_incoming(s, title="Сили оборони почали наступ", content_hash="H".ljust(64, "0"))
        s.commit()

    with Session(pg_engine) as s:
        inc_sig, candidates = find_publish_candidates(s, incoming, config=cfg)
        assert inc_sig is not None
        ids = [c.event_id for c, _sig, _pub in candidates]
        assert canon in ids                                   # exact content_hash twin found
        assert incoming not in ids                            # self excluded
        sig = [sig for c, sig, _ in candidates if c.event_id == canon][0]
        assert sig.exact_content_hash is True


@pytest.mark.pg
def test_find_publish_candidates_respects_window(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    cfg = PredupConfig(window_hours=72)
    old = dt.datetime.now(UTC) - dt.timedelta(hours=100)
    with Session(pg_engine) as s:
        _seed_published(s, title="Старе", content_hash="OLD".ljust(64, "0"), first_seen=old)
        incoming = _seed_incoming(s, title="Нове", content_hash="OLD".ljust(64, "0"))
        s.commit()

    with Session(pg_engine) as s:
        _inc, candidates = find_publish_candidates(s, incoming, config=cfg)
        assert candidates == []                               # the twin is outside the 72h window


@pytest.mark.pg
def test_check_auto_duplicate_without_judge(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        canon = _seed_published(s, title="Подія", content_hash="Z".ljust(64, "0"))
        incoming = _seed_incoming(s, title="Та сама подія", content_hash="Z".ljust(64, "0"))
        s.commit()

    verdict = PrepublishDedup(sf, judge=None).check(incoming)     # no LLM needed for an exact match
    assert verdict.action == ACTION_DUPLICATE and verdict.mode == "auto"
    assert verdict.canonical_event_id == canon


@pytest.mark.pg
def test_check_grey_zone_calls_judge(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        canon = _seed_published(s, title="Наступ", content_hash="A".ljust(64, "0"), simhash=0)
        # different content_hash but a near-identical SimHash -> grey (not auto) -> judge decides
        incoming = _seed_incoming(s, title="Наступ триває", content_hash="B".ljust(64, "0"), simhash=1)
        s.commit()

    judge = FakeJudge(ACTION_UPDATE)
    verdict = PrepublishDedup(sf, judge=judge).check(incoming)
    assert judge.calls == 1
    assert verdict.action == ACTION_UPDATE and verdict.mode == "llm"
    assert verdict.canonical_event_id == canon


@pytest.mark.pg
def test_check_no_candidates_is_separate(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        _seed_published(s, title="Щось", content_hash="P".ljust(64, "0"), simhash=0)
        # SimHash far from 0 (0xFFFF = 16 set bits -> Hamming 16 > simhash_max), different
        # content_hash, no URL/forward, no centroid -> nothing puts it in the candidate net
        incoming = _seed_incoming(s, title="Інше зовсім", content_hash="Q".ljust(64, "0"), simhash=0xFFFF)
        s.commit()

    judge = FakeJudge(ACTION_DUPLICATE)
    verdict = PrepublishDedup(sf, judge=judge).check(incoming)
    assert verdict.action == ACTION_SEPARATE and judge.calls == 0    # nothing in the net
