from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from newsroom.analyze.significance import (
    SignificanceConfig,
    SignificanceInputs,
    is_ua_relevant,
    load_significance_config,
    score_pending,
    significance_score,
)

CONFIG = Path(__file__).resolve().parents[1] / "config" / "significance.yaml"
CFG = load_significance_config(CONFIG)


# --- config loading (offline) -------------------------------------------------

def test_config_loads_real_file():
    assert 0.0 < CFG.threshold < 1.0
    assert CFG.rubric_weight("sport").locality_sensitive is True
    assert CFG.rubric_weight("technology").locality_sensitive is False   # global topic
    assert CFG.ua_markers                                                # markers compiled


def test_missing_config_raises(tmp_path):
    from newsroom.analyze.significance import SignificanceConfigError

    with pytest.raises(SignificanceConfigError):
        load_significance_config(tmp_path / "nope.yaml")


# --- is_ua_relevant (offline; guard false positives) --------------------------

def test_ua_markers_detect_ukraine():
    assert is_ua_relevant("У Львові відкрили виставку", CFG.ua_markers) is True
    assert is_ua_relevant("Кабмін ухвалив рішення", CFG.ua_markers) is True
    assert is_ua_relevant("Шахтар зіграв унічию", CFG.ua_markers) is True


def test_ua_markers_do_not_false_positive_on_foreign_text():
    assert is_ua_relevant("Apple unveils a new iPhone in California", CFG.ua_markers) is False
    assert is_ua_relevant("PSV coach comments on the draw", CFG.ua_markers) is False


# --- significance_score calibration (offline) ---------------------------------

def _passes(rubric, text, *, risk=None, sources=1, story=0):
    return significance_score(SignificanceInputs(
        rubric=rubric, risk_level=risk, text=text,
        independent_source_count=sources, story_event_count=story), CFG).passes


def test_foreign_minor_is_dropped():
    assert _passes("society", "Road collapses near a Malibu beach house") is False
    assert _passes("sport", "PSV coach comments after the Champions League draw") is False
    assert _passes("lifestyle", "Celebrity spotted at a Paris restaurant") is False


def test_foreign_but_globally_significant_is_kept():
    # geopolitics and tech are not locality-sensitive -> no foreign penalty
    assert _passes("security", "US and Iran escalate military tensions") is True
    assert _passes("technology", "Apple unveils the new iPhone lineup") is True


def test_ukraine_relevant_is_kept_even_in_low_rubric():
    assert _passes("sport", "Шахтар зіграв унічию з ПСВ") is True         # UA club -> bonus
    assert _passes("society", "У Києві відкрили новий транспортний вузол") is True


def test_critical_is_never_dropped_for_significance():
    # even a locality-sensitive, foreign-looking critical item is floored above threshold
    assert _passes("crime", "Explosion reported abroad", risk="critical") is True


def test_corroboration_and_story_momentum_lift_score():
    base = significance_score(SignificanceInputs(rubric="crime", text="foreign incident"), CFG).score
    more = significance_score(SignificanceInputs(rubric="crime", text="foreign incident",
                                                 independent_source_count=3, story_event_count=4), CFG).score
    assert more > base


# --- score_pending (pg) -------------------------------------------------------

@pytest.mark.pg
def test_score_pending_writes_significance_and_journals(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Item, Source

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="src", name="S", origin="world", tier="media")
        s.add(src)
        s.flush()
        # a foreign niche sport event (should score low) and a UA politics event (high)
        niche = Event(status="confirmed", rubric="sport", title="PSV draw in the league",
                      first_seen_at=now)
        good = Event(status="confirmed", rubric="politics", title="Зеленський підписав указ",
                     first_seen_at=now)
        s.add_all([niche, good])
        s.flush()
        for ev, txt in ((niche, "PSV coach comments on the draw"), (good, "Президент України підписав указ")):
            it = Item(source_id=src.id, external_id=f"i{ev.id}", content_hash=str(ev.id).ljust(64, "0"),
                      title=ev.title, text=txt)
            s.add(it)
            s.flush()
            s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
        s.commit()
        niche_id, good_id = niche.id, good.id

    stats = score_pending(sf, CFG, limit=50)
    assert stats["scored"] == 2 and stats["significant"] == 1 and stats["low"] == 1

    with Session(pg_engine) as s:
        assert s.get(Event, niche_id).significance < CFG.threshold
        assert s.get(Event, good_id).significance >= CFG.threshold
        decisions = {d.entity_id: d.decision for d in s.query(Decision).all()}
        assert decisions[str(niche_id)] == "low_significance"
        assert decisions[str(good_id)] == "significant"

    # idempotent: already scored -> nothing to do
    assert score_pending(sf, CFG, limit=50)["scored"] == 0
