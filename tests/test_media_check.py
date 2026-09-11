from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.factcheck.media import (
    MediaChecker,
    phash_distance,
    phash_to_int,
    select_reused,
)

UTC = dt.timezone.utc


# --- phash primitives (offline) -----------------------------------------------

def test_phash_to_int_and_distance():
    assert phash_to_int("ff00ff00ff00ff00") == 0xFF00FF00FF00FF00
    assert phash_distance("0000000000000000", "0000000000000001") == 1
    assert phash_distance("0000000000000000", "000000000000000f") == 4
    assert phash_distance("00", None) is None       # missing hash
    assert phash_distance("xyz", "00") is None       # unparsable


def test_phash_distance_identical_is_zero():
    assert phash_distance("abcdef0123456789", "abcdef0123456789") == 0


# --- select_reused (offline) --------------------------------------------------

def _cand(media_id, item_id, phash, when=None):
    return (media_id, item_id, phash, when)


def test_select_reused_within_distance_and_excludes_own():
    target = "0000000000000000"
    candidates = [
        _cand(1, 10, "0000000000000000"),   # dist 0 (near-identical, other item) -> match
        _cand(2, 11, "0000000000000003"),   # dist 2 -> match
        _cand(3, 12, "ffffffffffffffff"),   # dist 64 -> too far
        _cand(4, 99, "0000000000000000"),   # excluded item
    ]
    matches = select_reused(candidates, target, max_distance=6, exclude_item_ids=[99])
    assert [m.item_id for m in matches] == [10, 11]   # sorted by distance
    assert matches[0].distance == 0 and matches[1].distance == 2


def test_select_reused_skips_unparsable_phash():
    matches = select_reused([_cand(1, 10, "not-hex")], "0000000000000000", max_distance=6)
    assert matches == []


# --- MediaChecker (pg) --------------------------------------------------------

def _item_with_media(session, source_id, ext, phash, first_seen):
    from newsroom.models import Item, MediaAsset

    it = Item(source_id=source_id, external_id=ext, content_hash=ext, title=ext)
    session.add(it)
    session.flush()
    ma = MediaAsset(item_id=it.id, kind="image", phash=phash, first_seen_at=first_seen)
    session.add(ma)
    session.flush()
    return it.id


@pytest.mark.pg
def test_media_checker_flags_recycled_image(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Source

    sf = make_session_factory(pg_engine)
    old = dt.datetime(2026, 1, 1, tzinfo=UTC)
    new = dt.datetime(2026, 9, 1, tzinfo=UTC)
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://m/feed", name="M", origin="ua", tier="media")
        s.add(src)
        s.flush()
        # an older, unrelated item used the same image
        _old_item = _item_with_media(s, src.id, "old", "0000000000000000", old)
        # the event's item reuses that image now
        new_item = _item_with_media(s, src.id, "new", "0000000000000001", new)
        ev = Event(status="confirmed", title="Подія", first_seen_at=new)
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=new_item))
        s.commit()
        event_id = ev.id

    result = MediaChecker(sf, max_distance=6).check_event(event_id)
    assert result.checked == 1 and result.reused == 1 and not result.skipped

    with Session(pg_engine) as s:
        dec = s.execute(select(Decision).where(Decision.entity_id == str(event_id),
                                               Decision.stage == "verify")).scalars().one()
        assert dec.decision == "media_reuse" and dec.details["checked"] == 1

    # idempotent: already checked -> skipped, no second decision
    assert MediaChecker(sf).check_event(event_id).skipped
    with Session(pg_engine) as s:
        from sqlalchemy import func
        assert s.scalar(select(func.count()).select_from(Decision)
                        .where(Decision.entity_id == str(event_id), Decision.stage == "verify")) == 1


@pytest.mark.pg
def test_media_checker_clean_when_no_reuse(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, EventItem, Source

    sf = make_session_factory(pg_engine)
    now = dt.datetime(2026, 9, 1, tzinfo=UTC)
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://m2/feed", name="M2", origin="ua", tier="media")
        s.add(src)
        s.flush()
        item = _item_with_media(s, src.id, "solo", "abcdef0123456789", now)
        ev = Event(status="confirmed", title="Подія2", first_seen_at=now)
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=item))
        s.commit()
        event_id = ev.id

    result = MediaChecker(sf).check_event(event_id)
    assert result.checked == 1 and result.reused == 0
    with Session(pg_engine) as s:
        dec = s.execute(select(Decision).where(Decision.entity_id == str(event_id))).scalars().one()
        assert dec.decision == "media_clean"
