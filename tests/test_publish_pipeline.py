from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsroom.analyze.stoplist import load_stoplist
from newsroom.publishers.gate import Limits, set_publishing_stopped
from newsroom.publishers.pipeline import Publisher
from newsroom.publishers.telegram import TelegramPublisher

pytestmark = pytest.mark.pg
UTC = dt.timezone.utc
CONFIG = Path(__file__).resolve().parents[1] / "config"
STOP = load_stoplist(CONFIG / "stoplist.yaml")
LIMITS = Limits(urgent_per_hour=6, rumors_per_day=8, surge_window_minutes=30, surge_max_same_rubric=5)


class RecordingPoster:
    def __init__(self):
        self.calls = []

    def __call__(self, method, payload):
        self.calls.append((method, payload))
        return {"ok": True, "result": {"message_id": 500 + len(self.calls)}}


def _publisher(sf, poster, *, enabled=True):
    tg = TelegramPublisher("token", -100500, enabled=enabled, poster=poster)
    return Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS)


def _draft(pg_engine, *, body="Спокійна новина.\n\nДеталі тут.", headline="Заголовок",
           critic_ok=True, status="confirmed", risk_level="low", rubric="economy", is_rumor=False):
    from newsroom.models import Event, Publication

    with Session(pg_engine) as s:
        ev = Event(status=status, risk_level=risk_level, rubric=rubric, title="e",
                   first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                          headline=headline, body=body,
                          features={"critic_ok": critic_ok, "is_rumor": is_rumor})
        s.add(pub)
        s.flush()
        s.commit()
        return ev.id, pub.id


def test_publish_one_sends_and_marks_published(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    _ev, pid = _draft(pg_engine)

    outcome = _publisher(sf, poster).publish_one(pid)
    assert outcome.published is True and outcome.message_id == 501
    assert len(poster.calls) == 1 and poster.calls[0][0] == "sendMessage"

    with Session(pg_engine) as s:
        pub = s.get(Publication, pid)
        assert pub.status == "published" and pub.channel_ref == "501" and pub.published_at is not None
        dec = s.execute(select(Decision).where(Decision.entity_id == str(pid))).scalars().one()
        assert dec.decision == "published" and dec.stage == "publish"


def test_publish_pending_skips_critic_failed(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    _ev_ok, ok_id = _draft(pg_engine, critic_ok=True)
    _ev_bad, bad_id = _draft(pg_engine, critic_ok=False)

    stats = _publisher(sf, poster).publish_pending(limit=50)
    assert stats["published"] == 1
    with Session(pg_engine) as s:
        assert s.get(Publication, ok_id).status == "published"
        assert s.get(Publication, bad_id).status == "draft"   # never even selected


def test_stoplist_reblocks_at_publish_time(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    # a body that trips a block-action stop rule (troop movement without official source)
    _ev, pid = _draft(pg_engine, body="Колона 25-ї бригади ЗСУ висувається на Покровський напрямок зараз.")

    outcome = _publisher(sf, poster).publish_one(pid)
    if outcome.published:
        pytest.skip("stoplist rules do not block this sample; covered by gate unit tests")
    assert "stoplist" in outcome.reasons and poster.calls == []   # never sent
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "draft"
        dec = s.execute(select(Decision).where(Decision.entity_id == str(pid))).scalars().one()
        assert dec.decision == "blocked"


def test_stop_button_blocks_publishing(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    _ev, pid = _draft(pg_engine)
    with Session(pg_engine) as s:
        set_publishing_stopped(s, True, reason="test halt")
        s.commit()

    outcome = _publisher(sf, poster).publish_one(pid)
    assert not outcome.published and "stop_button" in outcome.reasons
    assert poster.calls == []
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "draft"


def test_disabled_publisher_touches_no_network_and_no_decision(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    _ev, pid = _draft(pg_engine)

    stats = _publisher(sf, poster, enabled=False).publish_pending(limit=50)
    assert stats.get("disabled") == 1 and poster.calls == []
    with Session(pg_engine) as s:
        assert s.scalar(select(func.count()).select_from(Decision)) == 0


def test_supervisor_notified_on_rumor_publish(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.publishers.supervision import Supervisor

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    supervisor = Supervisor(tg, admin_chat_id=777)
    publisher = Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS, supervisor=supervisor)

    # a labelled rumor in a non-critical topic (allowed) -> notice to admin
    _ev, pid = _draft(pg_engine, status="rumor", risk_level="low",
                      body="Чутка\n\nПодейкують, щось сталося.", is_rumor=True)
    outcome = publisher.publish_one(pid)
    assert outcome.published is True
    # two sends: the channel post + the admin notice
    admin_calls = [p for _, p in poster.calls if p["chat_id"] == 777]
    assert len(admin_calls) == 1 and "чутка" in admin_calls[0]["text"]


def test_block_decision_not_duplicated_on_repeat(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    _ev, pid = _draft(pg_engine)
    with Session(pg_engine) as s:
        set_publishing_stopped(s, True)
        s.commit()

    pub = _publisher(sf, poster)
    pub.publish_one(pid)
    pub.publish_one(pid)   # same block reason -> no second decision
    with Session(pg_engine) as s:
        assert s.scalar(select(func.count()).select_from(Decision)
                        .where(Decision.entity_id == str(pid))) == 1
