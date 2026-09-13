from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.orm import Session

from newsroom.editorial.digest import (
    compose_digest,
    due_windows,
    is_attack,
    load_digest_config,
    reserve_attacks,
    window_range,
)

CONFIG = Path(__file__).resolve().parents[1] / "config" / "digest.yaml"
CFG = load_digest_config(CONFIG)
KYIV = ZoneInfo("Europe/Kyiv")


# --- config + detection (offline) ---------------------------------------------

def test_config_loads():
    assert CFG.windows and CFG.rubrics and CFG.markers
    assert any(w.name.startswith("Обстріли за ніч") for w in CFG.windows)


def test_is_attack_needs_rubric_and_marker():
    assert is_attack("war", "Ударний БпЛА атакував Миколаїв", CFG) is True
    assert is_attack("war", "Зеленський підписав указ про бюджет", CFG) is False   # war but no attack marker
    assert is_attack("economy", "Обстріл спричинив зростання цін", CFG) is False   # marker but wrong rubric


# --- composition (offline) ----------------------------------------------------

def test_compose_digest_lists_events():
    headline, body = compose_digest("Обстріли за ніч", ["Атака на Миколаїв", "БпЛА на Черкаси"])
    assert headline == "Обстріли за ніч (2)"
    assert "• Атака на Миколаїв" in body and "• БпЛА на Черкаси" in body


# --- window/clock logic (offline) ---------------------------------------------

def test_night_window_range_crosses_midnight():
    night = next(w for w in CFG.windows if w.start_hour > w.end_hour)
    start, end = window_range(night, dt.date(2026, 9, 13), KYIV)
    assert start == dt.datetime(2026, 9, 12, night.start_hour, tzinfo=KYIV)
    assert end == dt.datetime(2026, 9, 13, night.end_hour, tzinfo=KYIV)


def test_due_windows_fire_after_publish_time_once():
    night = next(w for w in CFG.windows if w.start_hour > w.end_hour)
    at_publish = dt.datetime(2026, 9, 13, night.publish_hour, night.publish_minute, tzinfo=KYIV)
    due = due_windows(at_publish, {}, CFG)
    assert any(w.name == night.name for w, *_ in due)
    # already published today -> not due again
    due2 = due_windows(at_publish, {night.name: "2026-09-13"}, CFG)
    assert all(w.name != night.name for w, *_ in due2)
    # before publish time -> not due
    before = dt.datetime(2026, 9, 13, night.publish_hour - 1, tzinfo=KYIV)
    assert all(w.name != night.name for w, *_ in due_windows(before, {}, CFG))


# --- reservation (pg) ---------------------------------------------------------

@pytest.mark.pg
def test_reserve_attacks_marks_only_attacks(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        atk = Event(status="confirmed", rubric="war", title="Ударний БпЛА атакував Миколаїв", first_seen_at=now)
        pol = Event(status="confirmed", rubric="politics", title="Уряд ухвалив бюджет", first_seen_at=now)
        s.add_all([atk, pol])
        s.flush()
        atk_id, pol_id = atk.id, pol.id
        s.commit()

    assert reserve_attacks(sf, CFG)["reserved"] == 1
    with Session(pg_engine) as s:
        assert s.get(Event, atk_id).curated == "digest"    # attack reserved
        assert s.get(Event, pol_id).curated is None         # politics untouched
    # idempotent
    assert reserve_attacks(sf, CFG)["reserved"] == 0


@pytest.mark.pg
def test_reserve_skips_already_published_attack(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, Publication

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        ev = Event(status="confirmed", rubric="war", title="Ударний БпЛА атакував Черкаси",
                   first_seen_at=now)
        s.add(ev)
        s.flush()
        s.add(Publication(event_id=ev.id, channel="telegram", kind="post", status="published",
                          headline="h", body="b"))
        s.flush()
        eid = ev.id
        s.commit()

    assert reserve_attacks(sf, CFG)["reserved"] == 0        # already posted -> not folded into a digest
    with Session(pg_engine) as s:
        assert s.get(Event, eid).curated is None
