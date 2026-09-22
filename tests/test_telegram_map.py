from __future__ import annotations

import asyncio
import datetime as dt
from types import SimpleNamespace

from newsroom.collectors.telegram import (
    AlbumBuffer,
    extract_forwarded_from,
    extract_media,
    floodwait_seconds,
    merge_album,
    message_to_raw_item,
    run_with_floodwait,
)

UTC = dt.timezone.utc


def _msg(**kw):
    return SimpleNamespace(**kw)


def test_message_maps_core_fields():
    when = dt.datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
    raw = message_to_raw_item(_msg(id=42, message="Привіт", date=when, grouped_id=None),
                              source_id=3, channel_username="mychan")
    assert raw.source_id == 3
    assert raw.external_id == "42"
    assert raw.text == "Привіт"
    assert raw.published_at == when
    assert raw.url == "https://t.me/mychan/42"
    assert raw.grouped_id is None
    assert raw.fetched_at is not None            # head-start metric (§12)


def test_forwarded_from_username_and_title():
    m1 = _msg(id=1, forward=SimpleNamespace(chat=SimpleNamespace(username="origchan")))
    assert extract_forwarded_from(m1) == "@origchan"
    m2 = _msg(id=2, forward=SimpleNamespace(sender=SimpleNamespace(title="Генштаб ЗСУ")))
    assert extract_forwarded_from(m2) == "Генштаб ЗСУ"
    assert extract_forwarded_from(_msg(id=3)) is None


def test_extract_media_photo_and_video():
    photo = extract_media(_msg(id=1, photo=object()))
    assert [m.kind for m in photo] == ["image"]
    video = extract_media(_msg(id=2, video=SimpleNamespace(size=2048)))
    assert video[0].kind == "video" and video[0].size_bytes == 2048
    doc_video = extract_media(_msg(id=3, document=SimpleNamespace(mime_type="video/mp4", size=99)))
    assert doc_video[0].kind == "video"
    doc_image = extract_media(_msg(id=4, document=SimpleNamespace(mime_type="image/jpeg", size=77)))
    assert doc_image[0].kind == "image" and doc_image[0].size_bytes == 77
    assert extract_media(_msg(id=5)) == []


def test_album_merged_into_single_item():
    gid = 555
    when = dt.datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
    msgs = [
        message_to_raw_item(_msg(id=10, grouped_id=gid, date=when, photo=object()), source_id=1),
        message_to_raw_item(_msg(id=11, grouped_id=gid, message="підпис альбому", date=when, photo=object()), source_id=1),
        message_to_raw_item(_msg(id=12, grouped_id=gid, date=when, photo=object()), source_id=1),
    ]
    merged = merge_album(msgs)
    assert merged.external_id == "10"            # earliest id is the stable identity
    assert merged.text == "підпис альбому"        # caption preserved
    assert len(merged.media) == 3                 # all photos combined
    assert merged.raw_payload["album_size"] == 3


def test_album_buffer_debounce():
    gid = 7
    t0 = dt.datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
    buf = AlbumBuffer(debounce_seconds=2.0)
    for mid in (20, 21):
        buf.add(message_to_raw_item(_msg(id=mid, grouped_id=gid, photo=object()), source_id=1), now=t0)

    assert buf.flush_ready(now=t0 + dt.timedelta(seconds=1)) == []      # album still filling
    ready = buf.flush_ready(now=t0 + dt.timedelta(seconds=3))           # quiet for 3s -> complete
    assert len(ready) == 1 and len(ready[0].media) == 2
    assert buf.flush_ready(now=t0 + dt.timedelta(seconds=10)) == []     # nothing left


def test_floodwait_detection():
    class FloodWaitError(Exception):
        def __init__(self, seconds):
            self.seconds = seconds

    assert floodwait_seconds(FloodWaitError(30)) == 30
    assert floodwait_seconds(ValueError("nope")) is None


def test_run_with_floodwait_waits_then_succeeds():
    class FloodWaitError(Exception):
        def __init__(self, seconds):
            self.seconds = seconds

    calls = {"n": 0}
    slept: list[float] = []

    async def op():
        calls["n"] += 1
        if calls["n"] == 1:
            raise FloodWaitError(5)
        return "ok"

    async def fake_sleeper(sec):
        slept.append(sec)

    out = asyncio.run(run_with_floodwait(op, sleeper=fake_sleeper))
    assert out == "ok" and slept == [5] and calls["n"] == 2


def test_run_with_floodwait_reraises_other_errors():
    async def op():
        raise ValueError("real bug")

    async def fake_sleeper(sec):  # pragma: no cover — must not be called
        raise AssertionError("should not sleep on non-floodwait")

    try:
        asyncio.run(run_with_floodwait(op, sleeper=fake_sleeper))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "real bug" in str(e)
