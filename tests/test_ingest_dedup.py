from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.analyze.ingest_dedup import (
    MERGE_DUPLICATE,
    MERGE_SEPARATE,
    IngestDedup,
    MergeConfig,
    dedup_new_events,
    find_merge_candidates,
    load_merge_config,
    merge_events,
)
from newsroom.publishers.predup import TwinJudgment

UTC = dt.timezone.utc
CONFIG = Path(__file__).resolve().parents[1] / "config"


# --- config (offline) ---------------------------------------------------------

def test_load_merge_config_real_file():
    cfg = load_merge_config(CONFIG / "dedup.yaml")
    assert cfg.window_hours == 48 and cfg.simhash_max == 6 and cfg.top_candidates >= 1


def test_load_merge_config_defaults(tmp_path):
    p = tmp_path / "dedup.yaml"
    p.write_text("merge:\n  window_hours: 12\n", encoding="utf-8")
    cfg = load_merge_config(p)
    assert cfg.window_hours == 12 and cfg.vector_candidate == 0.55   # missing keys fall back


def test_load_merge_config_missing_section(tmp_path):
    p = tmp_path / "dedup.yaml"
    p.write_text("predup:\n  window_hours: 72\n", encoding="utf-8")
    cfg = load_merge_config(p)                    # no merge: section -> all defaults
    assert cfg.window_hours == 48


# --- dedup_settled_clause (offline: disabled -> no gating) ---------------------

def test_dedup_settled_clause_disabled_is_none():
    from newsroom.analyze.ingest_dedup import dedup_settled_clause

    assert dedup_settled_clause(False, dt.datetime.now(UTC)) is None       # off -> no-op
    assert dedup_settled_clause(True, dt.datetime.now(UTC)) is not None      # on -> a real clause


# --- fakes / seeding ----------------------------------------------------------

class FakeJudge:
    model = "fake-judge"

    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def judge(self, pair):
        self.calls += 1
        return TwinJudgment(decision=self.decision, confidence=0.9, reason="fake")


_tag = {"n": 0}


def _event_with_item(session, *, title, content_hash, simhash=None, first_seen=None, status="confirmed"):
    from newsroom.models import Event, EventItem, Item, Source

    _tag["n"] += 1
    t = _tag["n"]
    src = session.execute(select(Source).where(Source.handle_or_url == f"https://s/{t}")).scalar_one_or_none()
    if src is None:
        src = Source(kind="rss", handle_or_url=f"https://s/{t}", name="S", origin="ua", tier="media")
        session.add(src)
        session.flush()
    it = Item(source_id=src.id, external_id=f"e{t}", content_hash=content_hash, simhash=simhash, title=title)
    session.add(it)
    session.flush()
    ev = Event(status=status, title=title, first_seen_at=first_seen or dt.datetime.now(UTC))
    session.add(ev)
    session.flush()
    session.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
    session.flush()
    return ev.id, it.id


# --- find_merge_candidates (pg) -----------------------------------------------

@pytest.mark.pg
def test_find_merge_candidates_only_earlier_events(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    cfg = MergeConfig()
    with Session(pg_engine) as s:
        e1, _ = _event_with_item(s, title="Наступ", content_hash="H".ljust(64, "0"))
        e2, _ = _event_with_item(s, title="Наступ триває", content_hash="H".ljust(64, "0"))
        s.commit()

    with Session(pg_engine) as s:
        # the later event sees the earlier as a candidate
        _inc, cands = find_merge_candidates(s, e2, config=cfg)
        assert [c.event_id for c, _sig in cands] == [e1]
        # the earlier event has nothing earlier to merge into
        _inc, cands2 = find_merge_candidates(s, e1, config=cfg)
        assert cands2 == []


@pytest.mark.pg
def test_find_merge_candidates_excludes_duplicate_and_window(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event

    sf = make_session_factory(pg_engine)
    cfg = MergeConfig(window_hours=48)
    old = dt.datetime.now(UTC) - dt.timedelta(hours=100)
    with Session(pg_engine) as s:
        e_old, _ = _event_with_item(s, title="Старе", content_hash="H".ljust(64, "0"), first_seen=old)
        e_dup, _ = _event_with_item(s, title="Дубль", content_hash="H".ljust(64, "0"))
        e_new, _ = _event_with_item(s, title="Нове", content_hash="H".ljust(64, "0"))
        s.get(Event, e_dup).duplicate_of = e_old      # already merged -> not a candidate
        s.commit()

    with Session(pg_engine) as s:
        _inc, cands = find_merge_candidates(s, e_new, config=cfg)
        ids = [c.event_id for c, _ in cands]
        assert e_old not in ids                        # outside the 48h window
        assert e_dup not in ids                        # already a duplicate


# --- merge_events (pg) --------------------------------------------------------

@pytest.mark.pg
def test_merge_events_moves_items_and_marks_duplicate(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, EventItem

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        keep, _ = _event_with_item(s, title="Канон", content_hash="A".ljust(64, "0"))
        drop, drop_item = _event_with_item(s, title="Дубль", content_hash="B".ljust(64, "0"))
        s.commit()

    with Session(pg_engine) as s:
        moved = merge_events(s, keep=keep, drop=drop)
        s.commit()
        assert moved == 1

    with Session(pg_engine) as s:
        assert s.get(Event, drop).duplicate_of == keep
        keep_items = s.execute(select(EventItem.item_id).where(EventItem.event_id == keep)).scalars().all()
        assert drop_item in keep_items                 # source moved to the canonical
        assert s.execute(select(EventItem).where(EventItem.event_id == drop)).scalars().all() == []


@pytest.mark.pg
def test_merge_events_idempotent_on_already_duplicate(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        keep, _ = _event_with_item(s, title="K", content_hash="A".ljust(64, "0"))
        drop, _ = _event_with_item(s, title="D", content_hash="B".ljust(64, "0"))
        s.get(Event, drop).duplicate_of = keep
        s.commit()

    with Session(pg_engine) as s:
        assert merge_events(s, keep=keep, drop=drop) == 0     # already a duplicate -> no-op


# --- IngestDedup.check (pg) ---------------------------------------------------

@pytest.mark.pg
def test_check_auto_duplicate_exact_hash(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        e1, _ = _event_with_item(s, title="Подія", content_hash="Z".ljust(64, "0"))
        e2, _ = _event_with_item(s, title="Та сама", content_hash="Z".ljust(64, "0"))
        s.commit()

    verdict = IngestDedup(sf, judge=None).check(e2)          # exact -> no LLM
    assert verdict.action == MERGE_DUPLICATE and verdict.mode == "auto"
    assert verdict.canonical_event_id == e1


@pytest.mark.pg
def test_check_grey_update_does_not_merge(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        e1, _ = _event_with_item(s, title="Наступ", content_hash="A".ljust(64, "0"), simhash=0)
        e2, _ = _event_with_item(s, title="Наступ, деталі", content_hash="B".ljust(64, "0"), simhash=1)
        s.commit()

    judge = FakeJudge("update")                              # an update is NOT merged here
    verdict = IngestDedup(sf, judge=judge).check(e2)
    assert judge.calls == 1 and verdict.action == MERGE_SEPARATE


@pytest.mark.pg
def test_check_grey_duplicate_merges(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        e1, _ = _event_with_item(s, title="Наступ", content_hash="A".ljust(64, "0"), simhash=0)
        e2, _ = _event_with_item(s, title="Наступ інакше", content_hash="B".ljust(64, "0"), simhash=1)
        s.commit()

    verdict = IngestDedup(sf, judge=FakeJudge("duplicate")).check(e2)
    assert verdict.action == MERGE_DUPLICATE and verdict.mode == "llm" and verdict.canonical_event_id == e1


# --- dedup_new_events (pg) ----------------------------------------------------

@pytest.mark.pg
def test_dedup_new_events_enforce_merges(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        e1, _ = _event_with_item(s, title="Подія", content_hash="H".ljust(64, "0"))
        e2, e2_item = _event_with_item(s, title="Та сама подія", content_hash="H".ljust(64, "0"))
        s.commit()

    stats = dedup_new_events(sf, IngestDedup(sf, judge=None), enforce=True)
    assert stats["checked"] == 2 and stats["merged"] == 1 and stats["items_moved"] == 1
    with Session(pg_engine) as s:
        assert s.get(Event, e2).duplicate_of == e1
        assert e2_item in s.execute(select(EventItem.item_id).where(EventItem.event_id == e1)).scalars().all()
        # both events were journalled (stage ingest_dedup) -> not re-checked next tick
        stages = s.execute(select(Decision.decision).where(Decision.stage == "ingest_dedup")).scalars().all()
        assert "ingest_duplicate" in stages and "ingest_separate" in stages

    assert dedup_new_events(sf, IngestDedup(sf, judge=None), enforce=True)["checked"] == 0  # idempotent


@pytest.mark.pg
def test_dedup_settled_clause_gates_only_unsettled_recent_events(pg_engine):
    # the clause lets through events that are ingest-dedup-settled OR older than the grace,
    # and holds back a fresh event still awaiting its verdict.
    from newsroom.analyze.ingest_dedup import dedup_settled_clause
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(UTC)
    with Session(pg_engine) as s:
        fresh, _ = _event_with_item(s, title="Свіжа", content_hash="F".ljust(64, "0"))          # recent, no verdict
        settled, _ = _event_with_item(s, title="Вирішена", content_hash="S".ljust(64, "0"))     # recent, has verdict
        old, _ = _event_with_item(s, title="Стара", content_hash="O".ljust(64, "0"),
                                  first_seen=now - dt.timedelta(seconds=600))                    # past the grace
        s.add(Decision(entity_type="event", entity_id=str(settled), stage="ingest_dedup",
                       decision="ingest_separate"))
        s.commit()

    with Session(pg_engine) as s:
        clause = dedup_settled_clause(True, now - dt.timedelta(seconds=300))
        ids = set(s.execute(select(Event.id).where(clause)).scalars().all())
        assert fresh not in ids                      # held: recent and no verdict yet
        assert settled in ids and old in ids         # verdict in, or grace elapsed


@pytest.mark.pg
def test_merge_events_reverifies_canonical(pg_engine):
    # after moving an independent source in, the canonical event's derived state must be
    # refreshed: source count up, fact_base/significance invalidated, status re-decided.
    from newsroom.db import make_session_factory
    from newsroom.models import Event

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        keep, _ = _event_with_item(s, title="Удар по інфраструктурі", content_hash="A".ljust(64, "0"))
        drop, _ = _event_with_item(s, title="Той самий удар (інше джерело)", content_hash="B".ljust(64, "0"))
        k = s.get(Event, keep)
        k.risk_level = "high"
        k.status = "signal"                 # high + 1 source -> was 'signal' (needs a 2nd)
        k.independent_source_count = 1
        k.fact_base = {"facts": [{"text": "стара база"}]}
        s.commit()

    with Session(pg_engine) as s:
        merge_events(s, keep=keep, drop=drop)
        s.commit()

    with Session(pg_engine) as s:
        k = s.get(Event, keep)
        assert k.independent_source_count == 2       # gained the drop's independent source
        assert k.fact_base is None                   # invalidated for rebuild
        assert k.status == "reported"                # re-decided: high + 2 independent sources


@pytest.mark.pg
def test_dedup_new_events_retries_unresolved_then_gives_up(pg_engine):
    # an LLM outage (hold_review -> unresolved) must NOT settle the event as 'separate':
    # it is retried until the attempt cap, then given up (not spun forever).
    from sqlalchemy import func
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        e1, _ = _event_with_item(s, title="Наступ", content_hash="A".ljust(64, "0"), simhash=0)
        e2, _ = _event_with_item(s, title="Наступ, деталі", content_hash="B".ljust(64, "0"), simhash=1)
        s.commit()

    dedup = IngestDedup(sf, judge=FakeJudge("hold_review"))
    st1 = dedup_new_events(sf, dedup, enforce=True, max_unresolved_attempts=2)
    assert st1["checked"] == 2 and st1["unresolved"] == 1 and st1["merged"] == 0
    with Session(pg_engine) as s:
        assert s.get(Event, e2).duplicate_of is None      # unresolved never merges

    st2 = dedup_new_events(sf, dedup, enforce=True, max_unresolved_attempts=2)
    assert st2["checked"] == 1 and st2["unresolved"] == 1  # e1 settled(separate); e2 retried

    st3 = dedup_new_events(sf, dedup, enforce=True, max_unresolved_attempts=2)
    assert st3["checked"] == 0                              # e2 hit the cap -> given up
    with Session(pg_engine) as s:
        n = s.scalar(select(func.count()).select_from(Decision).where(
            Decision.entity_id == str(e2), Decision.decision == "ingest_unresolved"))
        assert n == 2


@pytest.mark.pg
def test_dedup_new_events_observe_only_logs(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        _e1, _ = _event_with_item(s, title="Подія", content_hash="H".ljust(64, "0"))
        e2, _ = _event_with_item(s, title="Та сама", content_hash="H".ljust(64, "0"))
        s.commit()

    stats = dedup_new_events(sf, IngestDedup(sf, judge=None), enforce=False)
    assert stats["checked"] == 2 and stats["merged"] == 0
    with Session(pg_engine) as s:
        assert s.get(Event, e2).duplicate_of is None          # observe changes no state
        dec = s.execute(select(Decision).where(
            Decision.entity_id == str(e2), Decision.stage == "ingest_dedup")).scalars().one()
        assert dec.decision == "ingest_duplicate" and dec.details["enforced"] is False
