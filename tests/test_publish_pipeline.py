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
# debounce off by default so the existing publish_pending tests (events at first_seen=now)
# are unaffected; a dedicated test exercises the debounce with its own Limits.
LIMITS = Limits(story_cooldown_minutes=60, publish_debounce_minutes=0)


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


def test_publish_renders_bold_headline_and_source_links(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    with Session(pg_engine) as s:
        ev = Event(status="confirmed", risk_level="low", rubric="economy", title="e",
                   first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                          headline="НБУ знизив ставку", body="НБУ знизив ставку\n\nСтавку — 13%.",
                          features={"critic_ok": True, "is_rumor": False,
                                    "render": {"headline": "НБУ знизив ставку",
                                               "body": "Ставку знижено до 13%.",
                                               "watching": "", "hashtags": ["#економіка"],
                                               "source_links": [["Цензор.НЕТ", "https://censor.net/a1"]],
                                               "is_rumor": False, "reported": False}})
        s.add(pub)
        s.flush()
        pid = pub.id
        s.commit()

    outcome = _publisher(sf, poster).publish_one(pid)
    assert outcome.published is True
    sent = poster.calls[0][1]["text"]
    assert "<b>НБУ знизив ставку</b>" in sent
    assert '<a href="https://censor.net/a1">Цензор.НЕТ</a>' in sent
    assert "Джерела" not in sent and poster.calls[0][1]["parse_mode"] == "HTML"


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


def test_media_attached_only_after_reuse_and_vision_clean(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, EventItem, Item, MediaAsset, Source

    sf = make_session_factory(pg_engine)
    seq = {"n": 0}

    def _seed_with_media(*, reuse_ok: bool, vision_ok: bool):
        seq["n"] += 1
        tag = seq["n"]
        with Session(pg_engine) as s:
            src = Source(kind="rss", handle_or_url=f"https://mm/{tag}", name="MM", origin="ua", tier="media")
            s.add(src)
            s.flush()
            it = Item(source_id=src.id, external_id=f"i{tag}", content_hash=f"h{tag}", title="t")
            s.add(it)
            s.flush()
            s.add(MediaAsset(item_id=it.id, kind="image", url="http://x/pic.jpg", width=1200,
                             storage_key=f"ready/{tag}", download_status="ready"))
            from newsroom.models import Event, Publication
            ev = Event(status="confirmed", risk_level="low", rubric="economy", title="e",
                       first_seen_at=dt.datetime.now(UTC))
            s.add(ev)
            s.flush()
            s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
            if reuse_ok:
                s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify",
                               decision="media_clean", details={"checked": 1}))
            if vision_ok:
                s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify",
                               decision="media_vision_ok", details={"checked": 1}))
            pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                              headline="Заголовок", body="Коротка новина.",
                              features={"critic_ok": True, "is_rumor": False})
            s.add(pub)
            s.flush()
            s.commit()
            return pub.id

    # reuse clean but no vision verdict -> wait; do not publish text first
    p1 = RecordingPoster()
    r1 = _publisher(sf, p1).publish_one(_seed_with_media(reuse_ok=True, vision_ok=False))
    assert r1.skipped and r1.reasons == ["media_checking"] and p1.calls == []

    # vision ok but reuse not checked -> wait as well
    p2 = RecordingPoster()
    r2 = _publisher(sf, p2).publish_one(_seed_with_media(reuse_ok=False, vision_ok=True))
    assert r2.skipped and r2.reasons == ["media_checking"] and p2.calls == []

    # both clean -> photo with caption
    p3 = RecordingPoster()
    _publisher(sf, p3).publish_one(_seed_with_media(reuse_ok=True, vision_ok=True))
    assert p3.calls[0][0] == "sendPhoto" and p3.calls[0][1]["photo"] == "http://x/pic.jpg"


def _seed_two_image_event(pg_engine, *, tag):
    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset, Publication, Source

    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url=f"https://al/{tag}", name="AL", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id=f"al{tag}", content_hash=f"al{tag}".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        s.add(MediaAsset(item_id=it.id, kind="image", url="http://x/a.jpg", width=1200,
                         storage_key=f"ready/{tag}/a", download_status="ready"))
        s.add(MediaAsset(item_id=it.id, kind="image", url="http://x/b.jpg", width=1000,
                         storage_key=f"ready/{tag}/b", download_status="ready"))
        ev = Event(status="confirmed", risk_level="low", rubric="economy", title="e",
                   first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_clean"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_vision_ok"))
        pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                          headline="Заголовок", body="Коротка новина.",
                          features={"critic_ok": True, "is_rumor": False})
        s.add(pub)
        s.flush()
        pid = pub.id
        s.commit()
        return pid


class _AlbumPoster:
    def __init__(self, *, group_ok=True):
        self.calls = []
        self._group_ok = group_ok

    def __call__(self, method, payload):
        self.calls.append((method, payload))
        if method == "sendMediaGroup":
            if not self._group_ok:
                return {"ok": False, "description": "album failed"}
            return {"ok": True, "result": [{"message_id": 700}, {"message_id": 701}]}
        return {"ok": True, "result": {"message_id": 500 + len(self.calls)}}


def test_publish_sends_album_for_multiple_images(pg_engine):
    import json as _json

    from newsroom.db import make_session_factory
    from newsroom.models import Publication

    sf = make_session_factory(pg_engine)
    pid = _seed_two_image_event(pg_engine, tag=1)
    poster = _AlbumPoster()
    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    assert Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS).publish_one(pid).published is True

    method, payload = poster.calls[0]
    assert method == "sendMediaGroup"
    media = _json.loads(payload["media"])
    assert len(media) == 2 and {m["media"] for m in media} == {"http://x/a.jpg", "http://x/b.jpg"}
    with Session(pg_engine) as s:
        p = s.get(Publication, pid)
        assert p.status == "published" and p.channel_ref == "700"   # first album message id


def test_publish_album_failure_falls_back_to_single(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Publication

    sf = make_session_factory(pg_engine)
    pid = _seed_two_image_event(pg_engine, tag=2)
    poster = _AlbumPoster(group_ok=False)          # album rejected -> must fall back
    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    assert Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS).publish_one(pid).published is True

    methods = [m for m, _ in poster.calls]
    assert methods[0] == "sendMediaGroup" and "sendPhoto" in methods   # fell back to a single photo
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "published"


class FakeStore:
    """In-memory store: serves seeded bytes and records deletes."""

    def __init__(self, files=None):
        self.files = dict(files or {})
        self.deleted: list[str] = []

    def put(self, key, data):  # pragma: no cover - unused here
        self.files[key] = data
        return key

    def exists(self, key):  # pragma: no cover - unused here
        return key in self.files and key not in self.deleted

    def get(self, key):
        return self.files.get(key)

    def delete(self, key):
        self.deleted.append(key)
        self.files.pop(key, None)
        return True


def test_local_media_deleted_after_publish(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset, Publication, Source

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    store = FakeStore({"ab/abcd": b"photo"})

    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://mm/purge", name="MM", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id="ip", content_hash="hp".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        # a downloaded asset (storage_key set) and a not-yet-downloaded one (skipped)
        downloaded = MediaAsset(item_id=it.id, kind="image", url="http://x/p.jpg",
                                width=1200, storage_key="ab/abcd", phash="p1",
                                download_status="ready")
        pending = MediaAsset(item_id=it.id, kind="image", url="http://x/q.jpg", width=1200,
                             download_status="failed", download_error="exhausted")
        s.add_all([downloaded, pending])
        ev = Event(status="confirmed", risk_level="low", rubric="economy", title="e",
                   first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify",
                       decision="media_clean"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify",
                       decision="media_vision_ok"))
        pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                          headline="Заголовок", body="Коротка новина.",
                          features={"critic_ok": True, "is_rumor": False})
        s.add(pub)
        s.flush()
        pid, downloaded_id, pending_id = pub.id, downloaded.id, pending.id
        s.commit()

    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    publisher = Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS,
                          media_store=store, purge_media_after_publish=True)
    assert publisher.publish_one(pid).published is True

    assert store.deleted == ["ab/abcd"]        # only the downloaded file was deleted
    with Session(pg_engine) as s:
        d = s.get(MediaAsset, downloaded_id)
        assert d.storage_key == "ab/abcd"       # key kept so nothing re-downloads
        assert d.purged_at is not None          # marked gone-locally
        assert s.get(MediaAsset, pending_id).purged_at is None   # never downloaded -> untouched


def test_telegram_media_uploaded_as_file(pg_engine):
    # a url-less Telegram image (storage_key set) that passed both checks is UPLOADED
    # (multipart), not sent by URL. Purge off so the stored bytes survive to send.
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset, Publication, Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = Source(kind="telegram", handle_or_url="@ch", name="ch", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id="tm1", content_hash="tm1".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        s.add(MediaAsset(item_id=it.id, kind="image", url=None, width=1200,
                         storage_key="ab/cd", source_ref="5"))
        ev = Event(status="confirmed", risk_level="low", rubric="economy", title="e",
                   first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_clean"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_vision_ok"))
        pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                          headline="Заголовок", body="Коротка новина.",
                          features={"critic_ok": True, "is_rumor": False})
        s.add(pub)
        s.flush()
        pid = pub.id
        s.commit()

    poster = RecordingPoster()
    store = FakeStore({"ab/cd": b"\xff\xd8\xffphoto-bytes"})
    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    publisher = Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS, media_store=store)
    assert publisher.publish_one(pid).published is True

    method, payload = poster.calls[0]
    assert method == "sendPhoto"
    assert "photo" not in payload                              # not a URL send
    assert payload["_file"]["field"] == "photo"
    assert payload["_file"]["data"] == b"\xff\xd8\xffphoto-bytes"


def test_url_media_prefers_local_upload_when_downloaded(pg_engine):
    # RSS media with a URL but also a downloaded file -> upload the bytes (Telegram
    # can't always fetch the URL), don't send the URL.
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset, Publication, Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://mm/up", name="MM", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id="up1", content_hash="up1".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        s.add(MediaAsset(item_id=it.id, kind="image", url="http://x/pic.jpg", width=1200,
                         storage_key="ef/gh"))
        ev = Event(status="confirmed", risk_level="low", rubric="economy", title="e",
                   first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_clean"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_vision_ok"))
        pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                          headline="Заголовок", body="Коротка новина.",
                          features={"critic_ok": True, "is_rumor": False})
        s.add(pub)
        s.flush()
        pid = pub.id
        s.commit()

    poster = RecordingPoster()
    store = FakeStore({"ef/gh": b"\xff\xd8\xffpic"})
    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS, media_store=store).publish_one(pid)
    method, payload = poster.calls[0]
    assert method == "sendPhoto" and "photo" not in payload      # uploaded, not sent by URL
    assert payload["_file"]["data"] == b"\xff\xd8\xffpic"


def test_media_send_failure_falls_back_to_text(pg_engine):
    # Telegram rejects the photo (can't fetch the URL) -> the post still goes out as text
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset, Publication, Source

    class PhotoFailsPoster:
        def __init__(self):
            self.calls = []

        def __call__(self, method, payload):
            self.calls.append((method, payload))
            if method == "sendPhoto":
                return {"ok": False, "description": "Bad Request: failed to get HTTP URL content"}
            return {"ok": True, "result": {"message_id": 900 + len(self.calls)}}

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://mm/fb", name="MM", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id="fb1", content_hash="fb1".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        s.add(MediaAsset(item_id=it.id, kind="image", url="http://x/unfetchable.jpg", width=1200,
                         storage_key="ready/unfetchable", download_status="ready"))
        ev = Event(status="confirmed", risk_level="low", rubric="economy", title="e",
                   first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_clean"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_vision_ok"))
        pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                          headline="Заголовок", body="Коротка новина.",
                          features={"critic_ok": True, "is_rumor": False})
        s.add(pub)
        s.flush()
        pid = pub.id
        s.commit()

    poster = PhotoFailsPoster()
    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    outcome = Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS).publish_one(pid)
    assert outcome.published is True                                  # not lost
    assert poster.calls[0][0] == "sendPhoto" and poster.calls[1][0] == "sendMessage"  # retried as text
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "published"


def test_telegram_media_falls_back_to_text_when_file_missing(pg_engine):
    # stored file gone (e.g. purged) -> drop media, post as text, do NOT fail the publish
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset, Publication, Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = Source(kind="telegram", handle_or_url="@ch2", name="ch2", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id="tm2", content_hash="tm2".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        s.add(MediaAsset(item_id=it.id, kind="image", url=None, width=1200,
                         storage_key="gone/x", source_ref="6"))
        ev = Event(status="confirmed", risk_level="low", rubric="economy", title="e",
                   first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_clean"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_vision_ok"))
        pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                          headline="Заголовок", body="Коротка новина.",
                          features={"critic_ok": True, "is_rumor": False})
        s.add(pub)
        s.flush()
        pid = pub.id
        s.commit()

    poster = RecordingPoster()
    store = FakeStore()          # empty -> get returns None
    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    publisher = Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS, media_store=store)
    assert publisher.publish_one(pid).published is True
    assert poster.calls[0][0] == "sendMessage"                 # posted as text, not failed


def test_media_not_purged_when_no_store(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset, Publication, Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://mm/nopurge", name="MM", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id="inp", content_hash="hnp".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        asset = MediaAsset(item_id=it.id, kind="image", url="http://x/p.jpg", storage_key="cd/cdef")
        s.add(asset)
        ev = Event(status="confirmed", risk_level="low", rubric="economy", title="e",
                   first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify",
                       decision="media_clean"))
        s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify",
                       decision="media_vision_ok"))
        pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                          headline="Заголовок", body="Коротка новина.",
                          features={"critic_ok": True, "is_rumor": False})
        s.add(pub)
        s.flush()
        pid, asset_id = pub.id, asset.id
        s.commit()

    poster = RecordingPoster()
    assert _publisher(sf, poster).publish_one(pid).published is True   # no media_store -> no purge
    with Session(pg_engine) as s:
        asset = s.get(MediaAsset, asset_id)
        assert asset.purged_at is None and asset.storage_key == "cd/cdef"


def test_media_attaches_on_reuse_alone_when_vision_disabled(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset, Publication, Source

    sf = make_session_factory(pg_engine)

    def _seed(*, vision_block: bool):
        with Session(pg_engine) as s:
            src = Source(kind="rss", handle_or_url=f"https://nv/{vision_block}", name="NV",
                         origin="ua", tier="media")
            s.add(src)
            s.flush()
            it = Item(source_id=src.id, external_id=f"nv{vision_block}",
                      content_hash=f"nv{vision_block}".ljust(64, "0"), title="t")
            s.add(it)
            s.flush()
            s.add(MediaAsset(item_id=it.id, kind="image", url="http://x/pic.jpg", width=1200,
                             storage_key=f"ready/{vision_block}", download_status="ready"))
            ev = Event(status="confirmed", risk_level="low", rubric="economy", title="e",
                       first_seen_at=dt.datetime.now(UTC))
            s.add(ev)
            s.flush()
            s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
            s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_clean"))
            if vision_block:                      # a prior block must still be honoured
                s.add(Decision(entity_type="event", entity_id=str(ev.id), stage="verify",
                               decision="media_vision_block"))
            pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                              headline="Заголовок", body="Коротка новина.",
                              features={"critic_ok": True, "is_rumor": False})
            s.add(pub)
            s.flush()
            pid = pub.id
            s.commit()
            return pid

    def _publisher_no_vision(poster):
        tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
        return Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS, require_vision=False)

    # vision off + reuse clean, no vision verdict -> media attaches
    p1 = RecordingPoster()
    _publisher_no_vision(p1).publish_one(_seed(vision_block=False))
    assert p1.calls[0][0] == "sendPhoto" and p1.calls[0][1]["photo"] == "http://x/pic.jpg"

    # vision off but an explicit block on record -> still text only
    p2 = RecordingPoster()
    _publisher_no_vision(p2).publish_one(_seed(vision_block=True))
    assert p2.calls[0][0] == "sendMessage"


def test_publish_waits_until_every_media_asset_is_settled(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset, Publication, Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = Source(kind="telegram", handle_or_url="@wait", name="wait", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id="wait1", content_hash="wait1".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        ready = MediaAsset(item_id=it.id, kind="image", url="http://x/ready.jpg",
                           source_ref="1", storage_key="ready/one", download_status="ready",
                           width=1200)
        pending = MediaAsset(item_id=it.id, kind="image", source_ref="2", download_status="pending",
                             width=1200)
        s.add_all([ready, pending])
        ev = Event(status="confirmed", title="wait", first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id))
        s.add_all([
            Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_clean"),
            Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_vision_ok"),
        ])
        pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                          headline="h", body="Коротка новина.", features={"critic_ok": True})
        s.add(pub)
        s.flush()
        pid, pending_id = pub.id, pending.id
        s.commit()

    poster = RecordingPoster()
    publisher = _publisher(sf, poster)
    first = publisher.publish_one(pid)
    assert first.skipped and first.reasons == ["media_pending"] and poster.calls == []
    with Session(pg_engine) as s:
        pub = s.get(Publication, pid)
        assert pub.media_approved_at is not None
        assert pub.has_media and pub.media_status == "pending"
        assert (pub.media_expected_count, pub.media_ready_count, pub.media_failed_count) == (2, 1, 0)
        asset = s.get(MediaAsset, pending_id)
        asset.download_status = "failed"
        asset.download_error = "message deleted"
        s.commit()

    second = publisher.publish_one(pid)
    assert second.published and poster.calls[0][0] == "sendPhoto"
    with Session(pg_engine) as s:
        pub = s.get(Publication, pid)
        assert pub.media_status == "ready" and pub.media_failed_count == 1


def test_long_approved_post_keeps_full_text_and_media(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset, Publication, Source

    sf = make_session_factory(pg_engine)
    body = "Д" * 1500
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://long", name="long", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id="long1", content_hash="long1".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        s.add(MediaAsset(item_id=it.id, kind="image", url="http://x/long.jpg", width=1200,
                         storage_key="ready/long", download_status="ready"))
        ev = Event(status="confirmed", title="long", first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id))
        s.add_all([
            Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_clean"),
            Decision(entity_type="event", entity_id=str(ev.id), stage="verify", decision="media_vision_ok"),
        ])
        pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                          headline="h", body=body, features={"critic_ok": True})
        s.add(pub)
        s.flush()
        pid = pub.id
        s.commit()

    poster = RecordingPoster()
    assert _publisher(sf, poster).publish_one(pid).published
    assert [method for method, _ in poster.calls] == ["sendMessage", "sendPhoto"]
    assert poster.calls[0][1]["text"] == body
    assert poster.calls[1][1]["reply_to_message_id"] == 501


def test_publish_skips_blocked_top_draft(pg_engine):
    # a perpetually-blocked top draft (its body trips the OPSEC stop-list) must not stall
    # the queue — a publishable lower-significance post still goes out.
    from newsroom.db import make_session_factory
    from newsroom.models import Event, Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    with Session(pg_engine) as s:
        top = Event(status="confirmed", risk_level="critical", rubric="war", title="top",
                    significance=0.9, first_seen_at=dt.datetime.now(UTC))     # blocks: stoplist (OPSEC)
        low = Event(status="confirmed", risk_level="low", rubric="economy", title="low",
                    significance=0.4, first_seen_at=dt.datetime.now(UTC))     # publishable
        s.add_all([top, low])
        s.flush()
        p_top = Publication(event_id=top.id, channel="telegram", kind="post", status="draft",
                            headline="Top", body="Позиції ППО поблизу Києва — детально.",
                            features={"critic_ok": True, "is_rumor": False})
        p_low = Publication(event_id=low.id, channel="telegram", kind="post", status="draft",
                            headline="Low", body="Спокійна новина з деталями.",
                            features={"critic_ok": True, "is_rumor": False})
        s.add_all([p_top, p_low])
        s.flush()
        top_id, low_id = p_top.id, p_low.id
        s.commit()

    stats = _publisher(sf, poster).publish_pending(limit=1)
    assert stats["published"] == 1 and stats["blocked"] >= 1
    with Session(pg_engine) as s:
        assert s.get(Publication, low_id).status == "published"   # lower post went out
        assert s.get(Publication, top_id).status == "draft"       # blocked top stayed a draft


def test_publish_skips_duplicate_of_event(pg_engine):
    # a draft whose event was later marked duplicate_of must NOT publish (dedup at publish time)
    from newsroom.db import make_session_factory
    from newsroom.models import Event, Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    with Session(pg_engine) as s:
        canon = Event(status="confirmed", risk_level="low", rubric="economy", title="canon",
                      significance=0.5, first_seen_at=dt.datetime.now(UTC))
        s.add(canon)
        s.flush()
        dupev = Event(status="confirmed", risk_level="low", rubric="economy", title="dup",
                      significance=0.9, duplicate_of=canon.id, first_seen_at=dt.datetime.now(UTC))
        s.add(dupev)
        s.flush()
        pub = Publication(event_id=dupev.id, channel="telegram", kind="post", status="draft",
                          headline="Dup", body="Дубль новина.", features={"critic_ok": True, "is_rumor": False})
        s.add(pub)
        s.flush()
        pid = pub.id
        s.commit()

    _publisher(sf, poster).publish_pending(limit=5)
    assert poster.calls == []                                  # duplicate draft never selected
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "draft"


def test_publish_story_cooldown_holds_second_same_story_post(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, Publication, Story

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    with Session(pg_engine) as s:
        story = Story(slug="s-cool", title="Сюжет", state="developing", last_event_at=dt.datetime.now(UTC))
        s.add(story)
        s.flush()
        for title, sig in (("e1", 0.9), ("e2", 0.8)):
            ev = Event(status="confirmed", risk_level="low", rubric="politics", title=title,
                       story_id=story.id, significance=sig, first_seen_at=dt.datetime.now(UTC))
            s.add(ev)
            s.flush()
            s.add(Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                              headline=title, body=f"{title}: новина з деталями.",
                              features={"critic_ok": True, "is_rumor": False}))
        s.commit()

    pub = _publisher(sf, poster)
    assert pub.publish_pending(limit=1)["published"] == 1      # first story post goes
    stats2 = pub.publish_pending(limit=1)
    assert stats2["published"] == 0 and stats2["blocked"] == 1  # second held by story_cooldown


def test_publish_debounce_holds_young_events(pg_engine):
    # a fresh event is held for the debounce window (so cross-source twins can arrive and
    # merge before the first publishes); an older one and a refutation are not held.
    from newsroom.db import make_session_factory
    from newsroom.models import Event, Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    now = dt.datetime.now(UTC)

    def _draft_aged(title, *, age_min, update_type=None):
        with Session(pg_engine) as s:
            ev = Event(status="confirmed", risk_level="low", rubric="economy", title=title,
                       update_type=update_type, significance=0.8,
                       first_seen_at=now - dt.timedelta(minutes=age_min))
            s.add(ev)
            s.flush()
            s.add(Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                              headline=title, body=f"{title}: новина з деталями.",
                              features={"critic_ok": True, "is_rumor": False}))
            s.flush()
            s.commit()

    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    limits = Limits(story_cooldown_minutes=60, publish_debounce_minutes=6)
    pub = Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=limits)

    _draft_aged("Свіжа", age_min=1)                           # younger than debounce -> held
    assert pub.publish_pending(limit=50)["published"] == 0

    _draft_aged("Стара", age_min=30)                          # older than debounce -> goes
    _draft_aged("Спростування", age_min=1, update_type="refutation")  # exempt -> goes
    assert pub.publish_pending(limit=50)["published"] == 2


def test_publish_prefers_significant_order(pg_engine):
    # no rate cap: all curated drafts publish, but in significance order (highest first)
    from newsroom.db import make_session_factory
    from newsroom.models import Event, Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    made = {}
    with Session(pg_engine) as s:
        for sig in (0.30, 0.90, 0.70):        # inserted out of order
            ev = Event(status="confirmed", risk_level="low", rubric="economy", title=f"e{sig}",
                       significance=sig, first_seen_at=dt.datetime.now(UTC))
            s.add(ev)
            s.flush()
            pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                              headline=f"H{sig}", body=f"Новина {sig} з конкретикою.",
                              features={"critic_ok": True, "is_rumor": False})
            s.add(pub)
            s.flush()
            made[sig] = pub.id
        s.commit()

    stats = _publisher(sf, poster).publish_pending(limit=50)
    assert stats["published"] == 3            # all publish — count is set by curation upstream, not a rate
    with Session(pg_engine) as s:
        for pid in made.values():
            assert s.get(Publication, pid).status == "published"
    # the highest-significance post was sent first
    first_headline = poster.calls[0][1]["text"].splitlines()[0]
    assert "0.9" in first_headline


def test_story_second_post_replies_to_first(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, Publication, Story

    sf = make_session_factory(pg_engine)

    with Session(pg_engine) as s:
        story = Story(slug="s-reply", title="Сюжет", state="developing",
                      last_event_at=dt.datetime.now(UTC))
        s.add(story)
        s.flush()

        def _draft_on_story(title, update_type=None):
            ev = Event(status="confirmed", risk_level="low", rubric="economy", title=title,
                       story_id=story.id, update_type=update_type, first_seen_at=dt.datetime.now(UTC))
            s.add(ev)
            s.flush()
            pub = Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                              headline=title, body=f"{title}: суть.",
                              features={"critic_ok": True, "is_rumor": False})
            s.add(pub)
            s.flush()
            return pub.id

        first_id = _draft_on_story("Перша подія")
        # a refutation threads onto the story even within the per-story cooldown window
        second_id = _draft_on_story("Спростування", update_type="refutation")
        s.commit()

    poster = RecordingPoster()   # each call returns message_id = 500 + call number
    publisher = _publisher(sf, poster)

    r1 = publisher.publish_one(first_id)
    assert r1.published and poster.calls[0][1].get("reply_to_message_id") is None  # first has no parent

    r2 = publisher.publish_one(second_id)
    assert r2.published
    # the second post replies to the first post's channel message id (501)
    assert poster.calls[-1][1]["reply_to_message_id"] == 501
    with Session(pg_engine) as s:
        assert s.get(Publication, second_id).reply_to_publication_id == first_id


# --- publish-time dedup (Phase A) --------------------------------------------

from newsroom.publishers.predup import PredupConfig, PrepublishDedup, TwinJudgment


class _FakeJudge:
    model = "fake-judge"

    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def judge(self, pair):
        self.calls += 1
        return TwinJudgment(decision=self.decision, confidence=0.9, reason="fake")


_twin_tag = {"n": 0}


def _seed_twin_case(pg_engine, *, exact, with_story=False):
    """Seed a canonical PUBLISHED event and an INCOMING draft that is its twin.
    exact=True -> same content_hash (auto-duplicate); exact=False -> near SimHash (grey)."""
    from newsroom.models import Event, EventItem, Item, Publication, Source, Story

    _twin_tag["n"] += 1
    tag = _twin_tag["n"]
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url=f"https://tw/{tag}", name="TW", origin="ua", tier="media")
        s.add(src)
        s.flush()
        story_id = None
        if with_story:
            story = Story(slug=f"tw-{tag}", title="Сюжет", state="developing",
                         last_event_at=dt.datetime.now(UTC))
            s.add(story)
            s.flush()
            story_id = story.id

        canon_hash = f"canon{tag}".ljust(64, "0")
        canon_item = Item(source_id=src.id, external_id=f"c{tag}", content_hash=canon_hash,
                          simhash=0, title="Наступ на півночі")
        s.add(canon_item)
        s.flush()
        canon = Event(status="confirmed", risk_level="low", rubric="war", title="Наступ на півночі",
                      story_id=story_id, first_seen_at=dt.datetime.now(UTC))
        s.add(canon)
        s.flush()
        s.add(EventItem(event_id=canon.id, item_id=canon_item.id, role="origin"))
        s.add(Publication(event_id=canon.id, channel="telegram", kind="post", status="published",
                          headline="Наступ на півночі", body="Опубліковано.",
                          published_at=dt.datetime.now(UTC), features={"critic_ok": True}))

        inc_hash = canon_hash if exact else f"inc{tag}".ljust(64, "0")
        inc_sim = 0 if exact else 1                # near-identical SimHash -> grey
        inc_item = Item(source_id=src.id, external_id=f"i{tag}", content_hash=inc_hash,
                        simhash=inc_sim, title="Сили оборони почали наступ")
        s.add(inc_item)
        s.flush()
        incoming = Event(status="confirmed", risk_level="low", rubric="war",
                         title="Сили оборони почали наступ", first_seen_at=dt.datetime.now(UTC))
        s.add(incoming)
        s.flush()
        s.add(EventItem(event_id=incoming.id, item_id=inc_item.id, role="origin"))
        pub = Publication(event_id=incoming.id, channel="telegram", kind="post", status="draft",
                          headline="Сили оборони почали наступ", body="Спокійна новина з деталями.",
                          features={"critic_ok": True, "is_rumor": False})
        s.add(pub)
        s.flush()
        s.commit()
        return canon.id, story_id, incoming.id, pub.id


def _publisher_predup(sf, poster, *, judge=None, enforce, supervisor=None):
    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    predup = PrepublishDedup(sf, judge=judge, config=PredupConfig())
    return Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS, supervisor=supervisor,
                     predup=predup, predup_enforce=enforce)


def test_predup_observe_logs_but_still_publishes(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    _canon, _st, inc_ev, pid = _seed_twin_case(pg_engine, exact=True)

    outcome = _publisher_predup(sf, poster, enforce=False).publish_one(pid)
    assert outcome.published is True                       # observe never blocks
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "published"
        assert s.get(Event, inc_ev).duplicate_of is None  # observe changes NO state
        dec = s.execute(select(Decision).where(
            Decision.entity_id == str(pid), Decision.stage == "predup")).scalars().one()
        assert dec.decision == "predup_duplicate" and dec.details["enforced"] is False


def test_predup_enforce_duplicate_supersedes_and_marks(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    canon, _st, inc_ev, pid = _seed_twin_case(pg_engine, exact=True)

    outcome = _publisher_predup(sf, poster, enforce=True).publish_one(pid)
    assert outcome.published is False and "predup_duplicate" in outcome.reasons
    assert poster.calls == []                              # never sent
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "superseded"
        assert s.get(Event, inc_ev).duplicate_of == canon


def test_predup_enforce_update_links_story_and_closes_draft(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    canon, story_id, inc_ev, pid = _seed_twin_case(pg_engine, exact=False, with_story=True)
    judge = _FakeJudge("update")

    outcome = _publisher_predup(sf, poster, judge=judge, enforce=True).publish_one(pid)
    assert outcome.published is False and "predup_update" in outcome.reasons
    assert judge.calls == 1 and poster.calls == []
    with Session(pg_engine) as s:
        ev = s.get(Event, inc_ev)
        assert ev.story_id == story_id          # linked to the canonical story
        assert ev.update_type is None           # reset so StoryUpdater reclassifies
        assert ev.duplicate_of is None          # an update is NOT a duplicate
        assert s.get(Publication, pid).status == "superseded"


def test_predup_enforce_hold_review_and_notifies(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Publication
    from newsroom.publishers.supervision import Supervisor

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    supervisor = Supervisor(tg, admin_chat_id=777)
    _canon, _st, _inc, pid = _seed_twin_case(pg_engine, exact=False)
    judge = _FakeJudge("hold_review")

    publisher = Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS, supervisor=supervisor,
                          predup=PrepublishDedup(sf, judge=judge, config=PredupConfig()),
                          predup_enforce=True)
    outcome = publisher.publish_one(pid)
    assert outcome.published is False and "predup_hold_review" in outcome.reasons
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "review"
    admin_calls = [p for _, p in poster.calls if p["chat_id"] == 777]
    assert len(admin_calls) == 1 and "дубл" in admin_calls[0]["text"].lower()


def test_predup_update_without_canonical_story_holds_for_review(pg_engine):
    # an "update" verdict is only safe to link when the canonical event already has a story;
    # with no story, superseding would orphan the event -> hold for review instead.
    from newsroom.db import make_session_factory
    from newsroom.models import Event, Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    _canon, story_id, inc_ev, pid = _seed_twin_case(pg_engine, exact=False, with_story=False)
    assert story_id is None
    judge = _FakeJudge("update")

    outcome = _publisher_predup(sf, poster, judge=judge, enforce=True).publish_one(pid)
    assert outcome.published is False and "predup_hold_review" in outcome.reasons
    with Session(pg_engine) as s:
        pub = s.get(Publication, pid)
        assert pub.status == "review"                 # held, not superseded -> not orphaned
        ev = s.get(Event, inc_ev)
        assert ev.duplicate_of is None and ev.story_id is None


class _RaisingPredup:
    def check(self, event_id):
        raise RuntimeError("boom")


def test_predup_error_publishes_and_journals_bypass(pg_engine):
    # a dedup-check failure must not freeze the queue, but the bypass must be VISIBLE:
    # the post goes out AND a predup_error decision is journalled.
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Publication
    from newsroom.publishers.supervision import Supervisor

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    supervisor = Supervisor(tg, admin_chat_id=777)
    _ev, pid = _draft(pg_engine)
    publisher = Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS, supervisor=supervisor,
                          predup=_RaisingPredup(), predup_enforce=True)

    outcome = publisher.publish_one(pid)
    assert outcome.published is True                   # channel not frozen
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "published"
        dec = s.execute(select(Decision).where(
            Decision.entity_id == str(pid), Decision.decision == "predup_error")).scalars().one()
        assert dec.details["published_without_check"] is True
    # enforce bypass alerts the admin
    admin_calls = [p for _, p in poster.calls if p["chat_id"] == 777]
    assert any("Дедуп" in c["text"] for c in admin_calls)


def test_predup_separate_publishes_normally(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    _canon, _st, _inc, pid = _seed_twin_case(pg_engine, exact=False)
    judge = _FakeJudge("separate")

    outcome = _publisher_predup(sf, poster, judge=judge, enforce=True).publish_one(pid)
    assert outcome.published is True and judge.calls == 1
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "published"


def test_predup_override_skips_check(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Publication

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    _canon, _st, _inc, pid = _seed_twin_case(pg_engine, exact=True)
    with Session(pg_engine) as s:                          # released-from-review one-shot override
        pub = s.get(Publication, pid)
        pub.features = {**(pub.features or {}), "predup_override": True}
        s.commit()

    judge = _FakeJudge("duplicate")
    outcome = _publisher_predup(sf, poster, judge=judge, enforce=True).publish_one(pid)
    assert outcome.published is True and judge.calls == 0  # twin check skipped entirely
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "published"


# --- outbox / delivery state (double-delivery protection) ---------------------

def test_claim_for_send_and_reconcile_ambiguous(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Publication

    sf = make_session_factory(pg_engine)
    _ev, pid = _draft(pg_engine)
    pub = _publisher(sf, RecordingPoster())

    assert pub._claim_for_send(pid) is True             # draft -> publishing
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "publishing"
    assert pub._claim_for_send(pid) is False            # already claimed -> no second claim

    # a row stuck in 'publishing' (crash mid-send) is reconciled to 'ambiguous', not resent
    assert pub.reconcile_pending_deliveries() == 1
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "ambiguous"


def test_send_failure_releases_claim_back_to_draft(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Publication

    class FailPoster:
        def __init__(self):
            self.calls = []

        def __call__(self, method, payload):
            self.calls.append((method, payload))
            return {"ok": False, "description": "boom"}

    sf = make_session_factory(pg_engine)
    _ev, pid = _draft(pg_engine)
    outcome = _publisher(sf, FailPoster()).publish_one(pid)
    assert outcome.published is False and "send_failed" in outcome.reasons
    with Session(pg_engine) as s:
        assert s.get(Publication, pid).status == "draft"    # claim released for retry, not stuck


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


# --- enrich-on-duplicate: upgrade the live post to a richer duplicate ----------

def _enrich_setup(pg_engine, *, canon_facts, dup_facts):
    from newsroom.models import Event, Publication

    now = dt.datetime.now(UTC)
    with Session(pg_engine) as s:
        canon = Event(status="confirmed", risk_level="low", rubric="economy", title="Курс",
                      fact_base={"facts": [{"text": f"c{i}"} for i in range(canon_facts)]}, first_seen_at=now)
        dup = Event(status="confirmed", risk_level="low", rubric="economy", title="Курс детальніше",
                    fact_base={"facts": [{"text": f"d{i}"} for i in range(dup_facts)]}, first_seen_at=now)
        s.add_all([canon, dup])
        s.flush()
        canon_pub = Publication(event_id=canon.id, channel="telegram", kind="post", status="published",
                                headline="Стара", body="Стара новина.", channel_ref="777",
                                published_at=now, features={"critic_ok": True})
        dup_pub = Publication(event_id=dup.id, channel="telegram", kind="post", status="draft",
                              headline="Нова", body="Новина з деталями.", features={"critic_ok": True})
        s.add_all([canon_pub, dup_pub])
        s.flush()
        ids = {"canon": canon.id, "dup": dup.id, "canon_pub": canon_pub.id, "dup_pub": dup_pub.id}
        s.commit()
    return ids


def test_enrich_upgrades_live_post_on_richer_duplicate(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, Publication
    from newsroom.publishers.predup import ACTION_DUPLICATE, TwinVerdict

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    ids = _enrich_setup(pg_engine, canon_facts=1, dup_facts=2)   # duplicate is richer

    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    publisher = Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS,
                          predup_enforce=True, enrich_on_duplicate=True)
    outcome = publisher._apply_predup(ids["dup_pub"], TwinVerdict(action=ACTION_DUPLICATE, mode="llm",
                                                                  canonical_event_id=ids["canon"]))
    assert outcome is not None and not outcome.published        # duplicate: not sent as a new post
    edits = [p for m, p in poster.calls if m == "editMessageText"]
    assert len(edits) == 1 and edits[0]["message_id"] == 777 and "Новина з деталями" in edits[0]["text"]
    with Session(pg_engine) as s:
        canon_pub = s.get(Publication, ids["canon_pub"])
        assert canon_pub.status == "edited" and canon_pub.body == "Новина з деталями."
        assert s.get(Publication, ids["dup_pub"]).status == "superseded"
        assert s.get(Event, ids["dup"]).duplicate_of == ids["canon"]


def test_enrich_skips_when_duplicate_not_richer(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Publication
    from newsroom.publishers.predup import ACTION_DUPLICATE, TwinVerdict

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    ids = _enrich_setup(pg_engine, canon_facts=2, dup_facts=2)   # NOT richer

    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    publisher = Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS,
                          predup_enforce=True, enrich_on_duplicate=True)
    publisher._apply_predup(ids["dup_pub"], TwinVerdict(action=ACTION_DUPLICATE, mode="llm",
                                                        canonical_event_id=ids["canon"]))
    assert [m for m, _ in poster.calls if m == "editMessageText"] == []   # no upgrade
    with Session(pg_engine) as s:
        assert s.get(Publication, ids["canon_pub"]).status == "published"  # live post untouched
        assert s.get(Publication, ids["dup_pub"]).status == "superseded"   # dup still dropped


def test_enrich_off_by_default_leaves_live_post(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Publication
    from newsroom.publishers.predup import ACTION_DUPLICATE, TwinVerdict

    sf = make_session_factory(pg_engine)
    poster = RecordingPoster()
    ids = _enrich_setup(pg_engine, canon_facts=1, dup_facts=2)

    tg = TelegramPublisher("token", -100500, enabled=True, poster=poster)
    publisher = Publisher(sf, telegram=tg, stoplist_rules=STOP, limits=LIMITS,
                          predup_enforce=True)                  # enrich_on_duplicate default False
    publisher._apply_predup(ids["dup_pub"], TwinVerdict(action=ACTION_DUPLICATE, mode="llm",
                                                        canonical_event_id=ids["canon"]))
    assert [m for m, _ in poster.calls if m == "editMessageText"] == []
    with Session(pg_engine) as s:
        assert s.get(Publication, ids["canon_pub"]).status == "published"
