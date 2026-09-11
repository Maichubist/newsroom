from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from newsroom.analyze.stoplist import load_stoplist
from newsroom.collectors.base import compute_simhash
from newsroom.publishers.shadow import (
    ShadowCriteria,
    dup_rate_from_simhashes,
    load_shadow_criteria,
    shadow_report,
)
from newsroom.publishers.telegram import TelegramPublisher

UTC = dt.timezone.utc
CONFIG = Path(__file__).resolve().parents[1] / "config"
STOP = load_stoplist(CONFIG / "stoplist.yaml")


# --- transport routing (offline) ----------------------------------------------

def test_shadow_routes_to_test_channel():
    tg = TelegramPublisher("token", -100500, enabled=True, shadow=True, shadow_chat_id=-100999,
                           poster=lambda m, p: {"ok": True, "result": {"message_id": 1}})
    assert tg.active_chat_id == -100999 and tg.is_enabled() is True


def test_shadow_without_test_channel_is_disabled():
    tg = TelegramPublisher("token", -100500, enabled=True, shadow=True, shadow_chat_id=None)
    assert tg.active_chat_id is None and tg.is_enabled() is False   # no test channel -> off


def test_non_shadow_uses_real_channel():
    tg = TelegramPublisher("token", -100500, enabled=True, shadow=False, shadow_chat_id=-100999)
    assert tg.active_chat_id == -100500


def test_send_text_targets_active_channel():
    calls = []
    tg = TelegramPublisher("token", -100500, enabled=True, shadow=True, shadow_chat_id=-100999,
                           poster=lambda m, p: calls.append(p) or {"ok": True, "result": {"message_id": 7}})
    tg.send_text("привіт")
    assert calls[0]["chat_id"] == -100999


def test_from_env_shadow(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHANNEL_CHAT_ID", "-100500")
    monkeypatch.setenv("PUBLISH_ENABLED", "true")
    monkeypatch.setenv("SHADOW_MODE", "true")
    monkeypatch.setenv("TELEGRAM_SHADOW_CHANNEL_CHAT_ID", "-100999")
    tg = TelegramPublisher.from_env(poster=lambda m, p: {"ok": True})
    assert tg.shadow is True and tg.active_chat_id == -100999


# --- dup_rate + criteria (offline) --------------------------------------------

def test_dup_rate_counts_repeats_only():
    a = compute_simhash("НБУ знизив ставку", "Нацбанк знизив облікову ставку до 13 відсотків.")
    a2 = compute_simhash("НБУ знизив ставку", "Нацбанк знизив облікову ставку до 13 відсотків!")  # near-dup
    b = compute_simhash("Погода", "Синоптики обіцяють дощ у Києві на вихідних.")
    rate = dup_rate_from_simhashes([a, a2, b], max_hamming=6)
    assert rate == pytest.approx(1 / 3)   # a2 duplicates a; a and b are firsts


def test_dup_rate_empty_is_zero():
    assert dup_rate_from_simhashes([], max_hamming=6) == 0.0


def test_load_shadow_criteria():
    c = load_shadow_criteria(CONFIG / "shadow.yaml")
    assert c.min_days >= 1 and c.min_publications >= 1 and 0 < c.max_dup_rate < 1


# --- shadow_report (pg) --------------------------------------------------------

CRIT = ShadowCriteria(min_days=7, min_publications=2, max_dup_rate=0.15, dup_max_hamming=6)


def _shadow_pub(pg_engine, *, headline, body, when, shadow=True, status="published"):
    from newsroom.models import Publication

    with Session(pg_engine) as s:
        pub = Publication(channel="telegram", kind="post", status=status, headline=headline, body=body,
                          channel_ref="1", published_at=when, features={"critic_ok": True, "shadow": shadow})
        s.add(pub)
        s.commit()


@pytest.mark.pg
def test_shadow_report_empty(pg_engine):
    from newsroom.db import make_session_factory

    rep = shadow_report(make_session_factory(pg_engine), criteria=CRIT, stoplist_rules=STOP)
    assert rep.publications == 0 and rep.auto_criteria_met is False
    assert any("no shadow publications" in n for n in rep.notes)


@pytest.mark.pg
def test_shadow_report_ready_when_criteria_met(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    old = dt.datetime.now(UTC) - dt.timedelta(days=10)
    _shadow_pub(pg_engine, headline="Подія А", body="Перший унікальний матеріал про економіку.", when=old)
    _shadow_pub(pg_engine, headline="Подія Б", body="Другий геть інший матеріал про культуру та спорт.",
                when=old + dt.timedelta(hours=1))
    # a non-shadow published post must not count
    _shadow_pub(pg_engine, headline="Реальний", body="Не тіньовий.", when=old, shadow=False)

    rep = shadow_report(sf, criteria=CRIT, stoplist_rules=STOP)
    assert rep.publications == 2 and rep.stoplist_violations == 0
    assert rep.dup_rate == 0.0 and rep.auto_criteria_met is True
    assert any("manual:" in n for n in rep.notes)   # human criterion always flagged


@pytest.mark.pg
def test_shadow_report_not_ready_when_too_few(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    _shadow_pub(pg_engine, headline="Подія", body="Єдиний матеріал.",
                when=dt.datetime.now(UTC) - dt.timedelta(days=10))
    rep = shadow_report(sf, criteria=CRIT, stoplist_rules=STOP)
    assert rep.publications == 1 and rep.auto_criteria_met is False
    assert any("need 2 publications" in n for n in rep.notes)
