from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.orm import Session

from newsroom.publishers.admin import (
    AdminConsole,
    is_readonly_sql,
    parse_command,
    run_readonly_sql,
)

UTC = dt.timezone.utc


# --- parse_command (offline) --------------------------------------------------

def test_parse_command():
    assert parse_command("/stats") == ("stats", "")
    assert parse_command("/pub 5") == ("pub", "5")
    assert parse_command("/sql SELECT 1 FROM t") == ("sql", "SELECT 1 FROM t")
    assert parse_command("/stats@my_bot") == ("stats", "")   # @botname stripped
    assert parse_command("hello") is None                     # not a command
    assert parse_command("/") is None                         # empty command
    assert parse_command("") is None


# --- is_readonly_sql (offline) ------------------------------------------------

def test_readonly_sql_allows_select_and_with():
    assert is_readonly_sql("SELECT 1")[0] is True
    assert is_readonly_sql("  select * from events ")[0] is True
    assert is_readonly_sql("WITH t AS (SELECT 1) SELECT * FROM t")[0] is True
    assert is_readonly_sql("SELECT 1;")[0] is True            # trailing ; is fine


def test_readonly_sql_rejects_writes_and_tricks():
    assert is_readonly_sql("DELETE FROM events")[0] is False
    assert is_readonly_sql("update sources set active=false")[0] is False
    assert is_readonly_sql("DROP TABLE items")[0] is False
    assert is_readonly_sql("SELECT 1; DROP TABLE x")[0] is False        # multiple statements
    assert is_readonly_sql("SELECT pg_read_file('/etc/passwd')")[0] is False   # blocklist
    assert is_readonly_sql("SELECT * FROM dblink('','')")[0] is False
    assert is_readonly_sql("")[0] is False


# --- AdminConsole dispatch (offline; no DB needed) ----------------------------

def test_console_ignores_non_commands_and_serves_help():
    c = AdminConsole(None)
    assert c.handle("just chatting") is None
    assert c.handle("/help").startswith("🛠")
    assert "Невідома команда" in c.handle("/nope")


# --- run_readonly_sql (pg) ----------------------------------------------------

@pytest.mark.pg
def test_run_readonly_sql_returns_rows(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    out = run_readonly_sql(sf, "SELECT 7 AS answer")
    assert "answer" in out and "7" in out


@pytest.mark.pg
def test_run_readonly_sql_rejects_non_select(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    assert "SELECT/WITH" in run_readonly_sql(sf, "DROP TABLE items")
    assert "pg_read_file" in run_readonly_sql(sf, "SELECT pg_read_file('x')")


@pytest.mark.pg
def test_run_readonly_sql_blocks_data_modifying_cte(pg_engine):
    # a DELETE hidden in a CTE passes the SELECT/WITH check but the READ ONLY
    # transaction must still block it (defence in depth).
    from newsroom.db import make_session_factory
    from newsroom.models import Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        s.add(Source(kind="telegram", handle_or_url="@keepme", name="keep", origin="ua", tier="media"))
        s.commit()

    out = run_readonly_sql(sf, "WITH x AS (DELETE FROM sources RETURNING id) SELECT count(*) FROM x")
    assert out.startswith("❌")                                   # read-only tx refused it
    with Session(pg_engine) as s:
        from sqlalchemy import func, select
        assert s.scalar(select(func.count()).select_from(Source)) >= 1   # nothing deleted


# --- canned reports (pg) ------------------------------------------------------

@pytest.mark.pg
def test_report_stats_and_event(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        ev = Event(status="confirmed", risk_level="low", rubric="economy", title="Тест",
                   significance=0.7, first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="confirmed"))
        eid = ev.id
        s.commit()

    console = AdminConsole(sf)
    assert "Стан пайплайна" in console.handle("/stats")
    ev_out = console.handle(f"/event {eid}")
    assert f"event {eid}" in ev_out and "confirmed" in ev_out
    assert console.handle("/event") == "Вкажи id події: /event 123"


# --- bot dispatch: admin-chat gating (offline) --------------------------------

class FakeTelegram:
    def __init__(self):
        self.sent = []

    def send_text(self, text, *, chat_id=None, disable_preview=True, **kw):
        self.sent.append((chat_id, text))
        from newsroom.publishers.telegram import PublishResult
        return PublishResult(True, message_id=1)


def _bot(admin_id):
    from newsroom.publishers.bot import SupervisionBot

    tg = FakeTelegram()
    bot = SupervisionBot(None, tg, admin_chat_id=admin_id, console=AdminConsole(None))
    return bot, tg


def test_bot_dispatches_command_only_from_admin_chat():
    bot, tg = _bot(404)
    # from the admin chat -> handled, reply sent to the admin
    assert bot.handle_update({"message": {"chat": {"id": 404}, "text": "/help"}}) is True
    assert tg.sent and tg.sent[0][0] == 404 and tg.sent[0][1].startswith("🛠")


def test_bot_ignores_non_admin_and_non_commands():
    bot, tg = _bot(404)
    # a different chat is ignored entirely (trust boundary)
    assert bot.handle_update({"message": {"chat": {"id": 999}, "text": "/help"}}) is False
    # the admin sending non-command text is not a command
    assert bot.handle_update({"message": {"chat": {"id": 404}, "text": "hi"}}) is False
    assert tg.sent == []


def test_bot_without_console_ignores_messages():
    from newsroom.publishers.bot import SupervisionBot

    tg = FakeTelegram()
    bot = SupervisionBot(None, tg)                     # no console/admin id
    assert bot.handle_update({"message": {"chat": {"id": 404}, "text": "/help"}}) is False


@pytest.mark.pg
def test_all_canned_commands_run_without_error(pg_engine):
    from newsroom.db import make_session_factory

    console = AdminConsole(make_session_factory(pg_engine))
    for cmd in ("/stats", "/pub", "/pub 3", "/demand", "/topics", "/topics 5", "/sources", "/top", "/top 5", "/media"):
        out = console.handle(cmd)
        assert isinstance(out, str) and out
        assert "помилка команди" not in out          # no dispatch crash


@pytest.mark.pg
def test_report_demand_empty_then_filled(pg_engine):
    from newsroom.analyze.demand import store_demand
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    assert "ще нема" in AdminConsole(sf).handle("/demand")
    with Session(pg_engine) as s:
        store_demand(s, {"politics": 0.9, "sport": 0.2})
        s.commit()
    out = AdminConsole(sf).handle("/demand")
    assert "politics" in out and "0.90" in out


@pytest.mark.pg
def test_report_topics_empty_then_filled(pg_engine):
    from newsroom.analyze.topics import store_hot_topics
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    assert "ще нема" in AdminConsole(sf).handle("/topics")
    with Session(pg_engine) as s:
        store_hot_topics(s, {"topics": [{"topic": "дрон", "events": 3, "posts": 5, "heat": 0.9}],
                             "heat": {"дрон": 0.9}, "at": "2026-09-13T00:00:00+00:00"})
        s.commit()
    out = AdminConsole(sf).handle("/topics")
    assert "дрон" in out
