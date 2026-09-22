from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.media.moderation import (
    ImageVerdict,
    _guess_image_mime,
    moderate_event_media,
    moderate_pending,
    moderation_source,
    parse_image_verdict,
)

UTC = dt.timezone.utc


# --- parse_image_verdict (offline) --------------------------------------------

def test_parse_verdict_full():
    v = parse_image_verdict('{"blocked": true, "labels": ["Тіла"], "reason": "жах"}')
    assert v.blocked is True and v.labels == ["тіла"] and v.reason == "жах"


def test_parse_verdict_missing_blocked_defaults_blocked():
    assert parse_image_verdict('{"labels": []}').blocked is True     # default-deny


def test_parse_verdict_clean():
    assert parse_image_verdict('{"blocked": false}').blocked is False


def test_parse_verdict_invalid_is_none():
    assert parse_image_verdict("not json") is None
    assert parse_image_verdict(None) is None
    assert parse_image_verdict("[1,2]") is None


# --- moderation_source / mime (offline) ---------------------------------------

def test_guess_image_mime():
    assert _guess_image_mime(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert _guess_image_mime(b"\x89PNG\r\n\x1a\nrest") == "image/png"
    assert _guess_image_mime(b"RIFF????WEBPvp8") == "image/webp"
    assert _guess_image_mime(b"unknown") == "image/jpeg"          # conservative default


class DictStore:
    def __init__(self, files=None):
        self.files = dict(files or {})

    def get(self, key):
        return self.files.get(key)


def test_moderation_source_prefers_url_then_stored_bytes():
    store = DictStore({"k1": b"\xff\xd8\xffdata"})
    assert moderation_source(store, "http://x/a.jpg", "k1") == "http://x/a.jpg"   # url wins
    src = moderation_source(store, None, "k1")
    assert src.startswith("data:image/jpeg;base64,")
    assert moderation_source(store, None, "missing") is None                     # no file
    assert moderation_source(None, None, "k1") is None                           # no store


# --- moderate_event_media (pg) ------------------------------------------------

class FakeModerator:
    model = "fake-vision"

    def __init__(self, blocked_urls=()):
        self._blocked = set(blocked_urls)

    def check(self, image_url):
        if image_url in self._blocked:
            return ImageVerdict(blocked=True, labels=["violence"], reason="graphic")
        return ImageVerdict(blocked=False)


def _event_with_images(pg_engine, urls, *, status="confirmed", approved=False):
    from newsroom.models import Event, EventItem, Item, MediaAsset, Publication, Source

    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url=f"https://v/{urls}", name="V", origin="ua", tier="media")
        s.add(src)
        s.flush()
        ev = Event(status=status, title="e", first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        for i, url in enumerate(urls):
            it = Item(source_id=src.id, external_id=f"{urls}-{i}", content_hash=f"{urls}-{i}", title="t")
            s.add(it)
            s.flush()
            s.add(MediaAsset(item_id=it.id, kind="image", url=url,
                             storage_key=f"stored/{i}" if approved else None,
                             download_status="ready" if approved else "pending"))
            s.add(EventItem(event_id=ev.id, item_id=it.id))
        if approved:
            s.add(Publication(
                event_id=ev.id, channel="telegram", kind="post", status="draft",
                headline="h", body="b", features={"critic_ok": True},
                media_approved_at=dt.datetime.now(UTC),
            ))
        s.commit()
        return ev.id


@pytest.mark.pg
def test_moderate_clean_and_blocked(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision

    sf = make_session_factory(pg_engine)
    clean_ev = _event_with_images(pg_engine, ["http://x/ok.jpg"])
    bad_ev = _event_with_images(pg_engine, ["http://x/ok2.jpg", "http://x/bad.jpg"])
    mod = FakeModerator(blocked_urls=["http://x/bad.jpg"])

    r1 = moderate_event_media(sf, mod, clean_ev)
    r2 = moderate_event_media(sf, mod, bad_ev)
    assert r1.checked == 1 and r1.blocked == 0
    assert r2.checked == 2 and r2.blocked == 1

    with Session(pg_engine) as s:
        d1 = s.execute(select(Decision).where(Decision.entity_id == str(clean_ev))).scalars().one()
        d2 = s.execute(select(Decision).where(Decision.entity_id == str(bad_ev))).scalars().one()
        assert d1.decision == "media_vision_ok"
        assert d2.decision == "media_vision_block" and d2.details["flags"][0]["labels"] == ["violence"]

    # idempotent
    assert moderate_event_media(sf, mod, clean_ev).skipped


@pytest.mark.pg
def test_moderate_no_images_is_noop(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        ev = Event(status="confirmed", title="text only", first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        eid = ev.id
        s.commit()
    result = moderate_event_media(sf, FakeModerator(), eid)
    assert result.skipped and result.checked == 0
    with Session(pg_engine) as s:
        assert s.scalar(select(Decision).where(Decision.entity_id == str(eid))) is None


@pytest.mark.pg
def test_moderate_pending_selects_events_with_images(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    _event_with_images(pg_engine, ["http://x/a.jpg"], approved=True)
    _event_with_images(pg_engine, ["http://x/b.jpg"], status="signal", approved=True)   # not publishable
    stats = moderate_pending(sf, FakeModerator(), limit=50)
    assert stats["events"] == 1


@pytest.mark.pg
def test_moderate_stored_telegram_image_via_store(pg_engine):
    # a url-less Telegram image (storage_key set) is moderated from the stored bytes
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset, Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = Source(kind="telegram", handle_or_url="@ch", name="ch", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id="m1", content_hash="m1".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        s.add(MediaAsset(item_id=it.id, kind="image", url=None, storage_key="ab/cd", source_ref="1"))
        ev = Event(status="confirmed", title="e", first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id))
        eid = ev.id
        s.commit()

    store = DictStore({"ab/cd": b"\xff\xd8\xffphoto"})
    # without a store the url-less image cannot be moderated -> skipped, no decision
    assert moderate_event_media(sf, FakeModerator(), eid, store=None).skipped
    with Session(pg_engine) as s:
        assert s.scalar(select(Decision).where(Decision.entity_id == str(eid))) is None
    # with the store it is moderated from the bytes
    r = moderate_event_media(sf, FakeModerator(), eid, store=store)
    assert r.checked == 1 and r.blocked == 0
    with Session(pg_engine) as s:
        d = s.execute(select(Decision).where(Decision.entity_id == str(eid))).scalars().one()
        assert d.decision == "media_vision_ok"
