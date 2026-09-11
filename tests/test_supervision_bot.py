from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.publishers.bot import CallbackAction, SupervisionBot, parse_callback
from newsroom.publishers.gate import is_publishing_stopped
from newsroom.publishers.telegram import TelegramPublisher

UTC = dt.timezone.utc


# --- parse_callback (offline) -------------------------------------------------

def test_parse_callback_variants():
    assert parse_callback("stop") == CallbackAction("stop")
    assert parse_callback("resume") == CallbackAction("resume")
    assert parse_callback("retract:42") == CallbackAction("retract", publication_id=42)
    assert parse_callback("retract:notint").kind == "unknown"
    assert parse_callback("garbage").kind == "unknown"
    assert parse_callback(None).kind == "unknown"


# --- bot actions (pg) ---------------------------------------------------------

class RecordingTelegram:
    """Fake adapter recording deleteMessage / answerCallbackQuery calls."""

    def __init__(self, *, active_chat_id=-100999, delete_ok=True):
        self.active_chat_id = active_chat_id
        self._delete_ok = delete_ok
        self.deleted = []
        self.answers = []

    def delete_message(self, chat_id, message_id):
        self.deleted.append((chat_id, message_id))
        return self._delete_ok

    def answer_callback_query(self, cq_id, text=None):
        self.answers.append((cq_id, text))
        return True


def _published_pub(pg_engine, *, channel_ref="555", status="published"):
    from newsroom.models import Publication

    with Session(pg_engine) as s:
        pub = Publication(channel="telegram", kind="post", status=status, headline="h", body="b",
                          channel_ref=channel_ref, published_at=dt.datetime.now(UTC))
        s.add(pub)
        s.flush()
        pid = pub.id
        s.commit()
        return pid


@pytest.mark.pg
def test_retract_deletes_message_and_marks_retracted(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Publication

    sf = make_session_factory(pg_engine)
    tg = RecordingTelegram(active_chat_id=-100999)
    pid = _published_pub(pg_engine, channel_ref="555")

    bot = SupervisionBot(sf, tg)
    result = bot.handle(parse_callback(f"retract:{pid}"))
    assert result == "Відкликано"
    assert tg.deleted == [(-100999, 555)]

    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "retracted"
        dec = s.execute(
            select(Decision).where(Decision.entity_id == str(pid), Decision.decision == "retracted")
        ).scalars().one()
        assert dec.details["channel_deleted"] is True


@pytest.mark.pg
def test_retract_missing_or_not_published(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    tg = RecordingTelegram()
    bot = SupervisionBot(sf, tg)
    assert bot.handle(parse_callback("retract:999999")) == "Публікацію не знайдено"

    draft = _published_pub(pg_engine, status="draft")
    assert "не активна" in bot.handle(parse_callback(f"retract:{draft}"))
    assert tg.deleted == []   # nothing deleted for a non-published post


@pytest.mark.pg
def test_stop_and_resume_flip_system_state(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    bot = SupervisionBot(sf, RecordingTelegram())

    bot.handle(parse_callback("stop"))
    with Session(pg_engine) as s:
        assert is_publishing_stopped(s) is True
    bot.handle(parse_callback("resume"))
    with Session(pg_engine) as s:
        assert is_publishing_stopped(s) is False


@pytest.mark.pg
def test_handle_update_answers_callback(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    tg = RecordingTelegram()
    bot = SupervisionBot(sf, tg)

    handled = bot.handle_update({"callback_query": {"id": "cq1", "data": "stop"}})
    assert handled is True and tg.answers == [("cq1", "Публікацію зупинено")]

    # a non-callback update is ignored
    assert bot.handle_update({"message": {"text": "hi"}}) is False
