from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.orm import Session

from newsroom.editorial.digest import (
    _attack_blob,
    compose_digest,
    digest_category,
    due_windows,
    load_digest_config,
    reserve_digests,
    window_range,
)

CONFIG = Path(__file__).resolve().parents[1] / "config" / "digest.yaml"
CFG = load_digest_config(CONFIG)
KYIV = ZoneInfo("Europe/Kyiv")


# --- config + detection (offline) ---------------------------------------------

def test_config_loads():
    assert CFG.windows and CFG.categories
    assert any(c.name == "Обстріли" for c in CFG.categories)
    assert any(w.name == "night" for w in CFG.windows)


def test_digest_category_needs_rubric_and_marker():
    assert digest_category("war", "Ударний БпЛА атакував Миколаїв", CFG).name == "Обстріли"
    assert digest_category("war", "Зеленський підписав указ про бюджет", CFG) is None   # war but no marker
    assert digest_category("politics", "Уряд ухвалив бюджет", CFG) is None              # not a digest rubric


def test_digest_category_precedence_and_new_buckets():
    # attacks beat losses (config order = precedence), plus the new routine buckets
    assert digest_category("war", "Росіяни втратили понад 1400 військових за добу", CFG).name == "Втрати ворога"
    assert digest_category("law_crime", "Поліція затримала трьох підозрюваних у крадіжці", CFG).name == "Кримінальна хроніка"
    assert digest_category("society", "На Хрещатику ДТП: зіткнулися дві автівки", CFG).name == "Місцева хроніка"
    assert digest_category("economy", "Долар подорожчав на міжбанку", CFG).name == "Економічні брифи"


def test_attack_blob_reads_title_and_facts():
    # a headline-less monitoring alert carries the marker only in the fact base
    fb = {"facts": [{"text": "Ударний БпЛА над містом, курсом на центр"}, {"text": ""}]}
    blob = _attack_blob("⚠ Одеса", fb)
    assert "⚠ Одеса" in blob and "Ударний БпЛА" in blob
    assert digest_category("war", blob, CFG).name == "Обстріли"          # matched via the body...
    assert digest_category("war", _attack_blob("⚠ Одеса", None), CFG) is None   # ...title alone would miss
    assert _attack_blob("t", {"facts": "not-a-list"}) == "t"               # malformed fact_base is safe


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
def test_reserve_digests_marks_only_attacks(pg_engine):
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

    assert reserve_digests(sf, CFG)["reserved"] == 1
    with Session(pg_engine) as s:
        assert s.get(Event, atk_id).curated == "digest"    # attack reserved
        assert s.get(Event, pol_id).curated is None         # politics untouched
    # idempotent
    assert reserve_digests(sf, CFG)["reserved"] == 0


@pytest.mark.pg
def test_reserve_catches_headline_less_alert_via_facts(pg_engine):
    # the real-world miss: a monitoring alert titled "⚠ Одеса" with the attack words only
    # in the fact base must still be reserved (a title-only check would let it be posted).
    from newsroom.db import make_session_factory
    from newsroom.models import Event

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        ev = Event(status="confirmed", rubric="war", title="⚠ Одеса",
                   fact_base={"facts": [{"text": "Ударний БпЛА над містом, курсом на центр"}]},
                   first_seen_at=now)
        s.add(ev)
        s.flush()
        eid = ev.id
        s.commit()

    assert reserve_digests(sf, CFG)["reserved"] == 1
    with Session(pg_engine) as s:
        assert s.get(Event, eid).curated == "digest"


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

    assert reserve_digests(sf, CFG)["reserved"] == 0        # already posted -> not folded into a digest
    with Session(pg_engine) as s:
        assert s.get(Event, eid).curated is None
