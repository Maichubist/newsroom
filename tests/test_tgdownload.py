from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy.orm import Session

from newsroom.media.download import persist_media_bytes
from newsroom.media.store import LocalMediaStore
from newsroom.media.tgdownload import select_pending_tg_media

pytestmark = pytest.mark.pg


class FakeDecoder:
    def __init__(self, pixels):
        self._pixels = pixels

    def to_grayscale(self, data, size):
        return self._pixels


def _tg_source(s, handle="@ch"):
    from newsroom.models import Source

    src = Source(kind="telegram", handle_or_url=handle, name=handle, origin="ua", tier="media")
    s.add(src)
    s.flush()
    return src.id


def _item(s, source_id, ext, *, status="clustered"):
    from newsroom.models import Item

    it = Item(source_id=source_id, external_id=ext, content_hash=ext.ljust(64, "0"),
              title="t", status=status)
    s.add(it)
    s.flush()
    return it.id


def test_select_pending_tg_media_filters(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import MediaAsset, Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        tg = _tg_source(s, "@ch")
        # eligible: telegram, url-less, has source_ref, item filter-passed, not stored
        good_item = _item(s, tg, "m1", status="clustered")
        good = MediaAsset(item_id=good_item, kind="image", url=None, source_ref="10")
        # skipped: already stored
        stored = MediaAsset(item_id=good_item, kind="image", url=None, source_ref="11", storage_key="x/y")
        # skipped: no source_ref
        noref = MediaAsset(item_id=good_item, kind="image", url=None, source_ref=None)
        # skipped: item not filter-passed
        new_item = _item(s, tg, "m2", status="new")
        onnew = MediaAsset(item_id=new_item, kind="image", url=None, source_ref="12")
        # skipped: RSS source with a URL (handled by the HTTP downloader)
        rss = Source(kind="rss", handle_or_url="https://r", name="r", origin="ua", tier="media")
        s.add(rss)
        s.flush()
        rss_item = _item(s, rss.id, "r1", status="clustered")
        withurl = MediaAsset(item_id=rss_item, kind="image", url="http://x/a.jpg")
        s.add_all([good, stored, noref, onnew, withurl])
        s.flush()
        good_id = good.id
        s.commit()

    rows = select_pending_tg_media(sf, limit=50)
    assert [r[0] for r in rows] == [good_id]           # only the eligible asset
    mid, item_id, kind, ref, handle = rows[0]
    assert kind == "image" and ref == "10" and handle == "@ch"


def test_persist_media_bytes_stores_and_hashes(pg_engine, tmp_path):
    from newsroom.db import make_session_factory
    from newsroom.models import MediaAsset

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        tg = _tg_source(s, "@ch2")
        item_id = _item(s, tg, "m9")
        asset = MediaAsset(item_id=item_id, kind="image", url=None, source_ref="99")
        s.add(asset)
        s.flush()
        mid = asset.id
        s.commit()

    store = LocalMediaStore(tmp_path)
    pixels = np.full((32, 32), 128.0)
    key, hashed = persist_media_bytes(sf, store, FakeDecoder(pixels),
                                      media_id=mid, item_id=item_id,
                                      key_source="tg:@ch2:99", kind="image", data=b"\xff\xd8\xffbytes")
    assert hashed is True and (tmp_path / key).read_bytes() == b"\xff\xd8\xffbytes"
    with Session(pg_engine) as s:
        a = s.get(MediaAsset, mid)
        assert a.storage_key == key and a.size_bytes == len(b"\xff\xd8\xffbytes") and a.phash is not None
