from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from newsroom.analyze.stoplist import (
    StopListConfigError,
    check,
    is_blocked,
    load_stoplist,
    worst_action,
)

RULES = load_stoplist(Path(__file__).resolve().parents[1] / "config" / "stoplist.yaml")


def _ids(violations):
    return {v.rule_id for v in violations}


# --- positives -----------------------------------------------------------------

def test_realtime_missile_tracking_blocked():
    v = check("", "Шахеди курсом на Київ, орієнтовний час — 10 хвилин.", RULES)
    assert "realtime-missile-tracking" in _ids(v) and is_blocked(v)


def test_airdefense_positions_blocked():
    v = check("", "Позиції ППО поблизу Одеси помітили місцеві.", RULES)
    assert "ua-airdefense-positions" in _ids(v) and worst_action(v) == "block"


def test_military_personal_data_blocked():
    v = check("", "Боєць з позивним «Сокіл», проживає за адресою в Києві.", RULES)
    assert "ua-military-personal-data" in _ids(v)


def test_suicide_method_is_review_not_block():
    v = check("", "Слідство встановлює спосіб самогубства.", RULES)
    assert "suicide-method" in _ids(v)
    assert worst_action(v) == "review" and is_blocked(v) is False


# --- false positives (must NOT trigger) ----------------------------------------

def test_missile_intercept_result_is_not_tracking():
    assert check("", "Сили ППО збили ракету над Києвом уночі.", RULES) == []


def test_person_travelling_is_not_missile_tracking():
    assert check("", "Президент прямує на саміт у Брюсселі.", RULES) == []


def test_airdefense_success_without_location_is_ok():
    assert check("", "ППО збила 10 дронів цієї ночі.", RULES) == []


def test_callsign_without_address_is_ok():
    assert check("", "Боєць з позивним «Сокіл» отримав державну нагороду.", RULES) == []


def test_suicide_rate_news_is_ok():
    assert check("", "Рівень самогубств у країні знизився за рік.", RULES) == []


# --- scope / side --------------------------------------------------------------

def test_ua_side_rule_skipped_for_russian_side():
    text = "Позиції ППО поблизу Москви."
    assert "ua-airdefense-positions" in _ids(check("", text, RULES, side="unknown"))
    assert check("", text, RULES, side="ru") == []   # RU air defence is publishable (§4.3)


# --- config validation ---------------------------------------------------------

@pytest.mark.parametrize("body, needle", [
    ("rules:\n  - {id: r1, scope: nowhere, action: block, any: ['x']}", "scope"),
    ("rules:\n  - {id: r1, scope: all, action: nuke, any: ['x']}", "action"),
    ("rules:\n  - {id: r1, scope: all, action: block, any: ['(oops']}", "bad regex"),
    ("rules:\n  - {id: r1, scope: all, action: block}", "any"),
])
def test_invalid_config_rejected(tmp_path, body, needle):
    p = tmp_path / "stoplist.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    with pytest.raises(StopListConfigError) as exc:
        load_stoplist(p)
    assert needle in str(exc.value)
