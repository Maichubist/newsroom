from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from newsroom.analyze.signal import (
    FilterConfigError,
    classify_noise,
    ipso_markers,
    load_filters,
)

FILTERS = load_filters(Path(__file__).resolve().parents[1] / "config" / "filters.yaml")


# --- noise: positives ----------------------------------------------------------

def test_advertising_is_noise():
    v = classify_noise("Огляд ноутбука", "На правах реклами. Купуйте зараз.", FILTERS)
    assert v.is_noise and "advertising" in v.reasons


def test_subscribe_to_win_is_noise():
    v = classify_noise("Розіграш!", "Підпишись на канал і вигравай приз щодня.", FILTERS)
    assert v.is_noise and "subscribe_to_win" in v.reasons


# --- noise: false positives (must NOT be dropped) ------------------------------

def test_normal_news_with_subscribe_footer_is_not_noise():
    v = classify_noise(
        "Курс гривні зміцнився",
        "НБУ повідомив про зміцнення гривні. Підписуйтесь на наш канал.",
        FILTERS,
    )
    assert v.is_noise is False and v.reasons == []


def test_news_about_advertising_market_is_not_noise():
    v = classify_noise("Ринок реклами виріс на 20%", "Аналітики зафіксували зростання ринку реклами.", FILTERS)
    assert v.is_noise is False


# --- ipso: positives -----------------------------------------------------------

def test_calls_to_share_and_panic_flagged():
    markers = ipso_markers("", "Терміново всім! Поширте це повідомлення, поки не видалили.", FILTERS)
    assert "calls_to_share" in markers and "panic_urgency" in markers


def test_anonymous_insider_flagged():
    assert "anonymous_insider" in ipso_markers("", "Інсайдер повідомляє про відставку.", FILTERS)


# --- ipso: false positives (imperative verb vs. noun) --------------------------

def test_spread_noun_is_not_a_call_to_share():
    # "поширення"/"розповсюдження" (nouns) must not trigger the imperative patterns
    assert ipso_markers("", "Медики попереджають про поширення вірусу та розповсюдження інфекції.", FILTERS) == []


def test_plain_news_has_no_ipso_markers():
    assert ipso_markers("Уряд ухвалив бюджет", "Кабмін затвердив держбюджет на наступний рік.", FILTERS) == []


# --- config --------------------------------------------------------------------

def test_bad_regex_rejected(tmp_path):
    p = tmp_path / "filters.yaml"
    p.write_text(textwrap.dedent("""
        noise:
          broken: ['(unclosed']
    """), encoding="utf-8")
    with pytest.raises(FilterConfigError):
        load_filters(p)
