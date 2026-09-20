from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from sqlalchemy.orm import Session

from newsroom.factcheck.media import phash_distance
from newsroom.media.download import MediaDownloader
from newsroom.media.phash import phash_bytes, phash_from_pixels
from newsroom.media.store import LocalMediaStore, media_key

UTC = dt.timezone.utc


# --- phash_from_pixels (offline) ----------------------------------------------

def _pattern(size=32, seed=7):
    """A photo-like image: many low-frequency components with random amplitude and
    phase, so the DCT low-frequency block is dense and its coefficients sit well
    away from the median (stable pHash bits, like a real photo)."""
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.full((size, size), 128.0)
    for _ in range(24):
        fx, fy = rng.integers(1, 6), rng.integers(1, 6)
        amp, phase = rng.uniform(6, 22), rng.uniform(0, 2 * np.pi)
        img += amp * np.sin(2 * np.pi * (fx * x + fy * y) / size + phase)
    return img


def test_phash_is_16_hex_and_stable():
    h = phash_from_pixels(_pattern())
    assert len(h) == 16 and int(h, 16) >= 0
    assert phash_from_pixels(_pattern()) == h               # deterministic


def test_phash_similar_images_are_close():
    base = _pattern()
    noisy = base + np.random.default_rng(0).normal(0, 0.8, base.shape)   # light re-encode-like jitter
    assert phash_distance(phash_from_pixels(base), phash_from_pixels(noisy)) <= 6


def test_phash_different_images_are_far():
    a = phash_from_pixels(_pattern())
    checker = (np.indices((32, 32)).sum(axis=0) % 2) * 255.0     # checkerboard (all high-freq)
    assert phash_distance(a, phash_from_pixels(checker)) >= 12


def test_phash_wrong_shape_raises():
    with pytest.raises(ValueError):
        phash_from_pixels(np.zeros((10, 10)))


class FakeDecoder:
    def __init__(self, array):
        self._array = array

    def to_grayscale(self, data, size):
        return self._array


class BrokenDecoder:
    def to_grayscale(self, data, size):
        raise ValueError("cannot decode")


def test_phash_bytes_uses_decoder_and_survives_failure():
    assert phash_bytes(b"x", FakeDecoder(_pattern())) == phash_from_pixels(_pattern())
    assert phash_bytes(b"x", BrokenDecoder()) is None


# --- LocalMediaStore (offline) ------------------------------------------------

def test_local_store_writes_and_reports(tmp_path):
    store = LocalMediaStore(tmp_path)
    key = media_key(1, 2, "http://x/a.jpg")
    returned = store.put(key, b"bytes")
    assert returned == key and store.exists(key)
    assert (tmp_path / key).read_bytes() == b"bytes"


def test_media_key_is_stable_and_distinct():
    assert media_key(1, 2, "u") == media_key(1, 2, "u")
    assert media_key(1, 2, "u") != media_key(1, 3, "u")


def test_local_store_delete_is_idempotent_and_prunes_shard(tmp_path):
    store = LocalMediaStore(tmp_path)
    key = media_key(1, 2, "http://x/a.jpg")
    store.put(key, b"bytes")
    assert store.delete(key) is True          # deleted
    assert not store.exists(key)
    assert not (tmp_path / key).parent.exists()   # empty shard dir pruned
    assert store.delete(key) is False         # already gone -> idempotent


# --- MediaDownloader (pg) -----------------------------------------------------

def _seed_asset(pg_engine, *, item_status="accepted", url="http://x/a.jpg", kind="image"):
    from newsroom.models import Item, MediaAsset, Source

    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url=f"https://d/{url}", name="D", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id=url, content_hash=url, title="t", status=item_status)
        s.add(it)
        s.flush()
        ma = MediaAsset(item_id=it.id, kind=kind, url=url)
        s.add(ma)
        s.flush()
        mid = ma.id
        s.commit()
        return mid


@pytest.mark.pg
def test_download_stores_and_hashes_image(pg_engine, tmp_path):
    from newsroom.db import make_session_factory
    from newsroom.models import MediaAsset

    sf = make_session_factory(pg_engine)
    mid = _seed_asset(pg_engine)
    dl = MediaDownloader(sf, store=LocalMediaStore(tmp_path), decoder=FakeDecoder(_pattern()),
                         fetch=lambda url: b"imagebytes")

    result = dl.download_asset(mid)
    assert result.stored and result.hashed
    with Session(pg_engine) as s:
        asset = s.get(MediaAsset, mid)
        assert asset.storage_key and asset.size_bytes == len(b"imagebytes")
        assert asset.phash and len(asset.phash) == 16

    # idempotent: already stored -> skipped
    assert dl.download_asset(mid).skipped


@pytest.mark.pg
def test_download_pending_only_filter_passed(pg_engine, tmp_path):
    from newsroom.db import make_session_factory
    from newsroom.models import MediaAsset

    sf = make_session_factory(pg_engine)
    kept = _seed_asset(pg_engine, item_status="accepted")
    dropped = _seed_asset(pg_engine, item_status="filtered_out", url="http://x/b.jpg")
    dl = MediaDownloader(sf, store=LocalMediaStore(tmp_path), decoder=FakeDecoder(_pattern()),
                         fetch=lambda url: b"data")

    stats = dl.download_pending(limit=50)
    assert stats["stored"] == 1
    with Session(pg_engine) as s:
        assert s.get(MediaAsset, kept).storage_key is not None
        assert s.get(MediaAsset, dropped).storage_key is None   # noise media never downloaded


@pytest.mark.pg
def test_download_skips_oversized(pg_engine, tmp_path):
    from newsroom.db import make_session_factory
    from newsroom.models import MediaAsset

    sf = make_session_factory(pg_engine)
    mid = _seed_asset(pg_engine)
    dl = MediaDownloader(sf, store=LocalMediaStore(tmp_path), decoder=FakeDecoder(_pattern()),
                         fetch=lambda url: b"x" * 100, max_bytes=10)
    result = dl.download_asset(mid)
    assert result.skipped and not result.stored
    with Session(pg_engine) as s:
        assert s.get(MediaAsset, mid).storage_key is None


# --- purge_stale_media (pg) ----------------------------------------------------

class _RecordingStore:
    def __init__(self):
        self.deleted: list[str] = []

    def delete(self, key):
        self.deleted.append(key)
        return True


@pytest.mark.pg
def test_purge_stale_media_removes_only_old_unpublished(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.media.purge import purge_stale_media
    from newsroom.models import (
        Event, EventItem, Item, MediaAsset, Publication, Source,
    )

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(UTC)
    old = now - dt.timedelta(hours=72)
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://p/f", name="P", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id="i1", content_hash="c1".ljust(64, "0"), status="clustered")
        it_pending = Item(source_id=src.id, external_id="i2", content_hash="c2".ljust(64, "0"), status="clustered")
        s.add_all([it, it_pending])
        s.flush()
        old_asset = MediaAsset(item_id=it.id, kind="image", storage_key="aa/old", first_seen_at=old)
        new_asset = MediaAsset(item_id=it.id, kind="image", storage_key="bb/new", first_seen_at=now)
        # old media, but its event has an in-flight draft -> must be kept
        pending_asset = MediaAsset(item_id=it_pending.id, kind="image", storage_key="cc/pending", first_seen_at=old)
        s.add_all([old_asset, new_asset, pending_asset])
        ev = Event(status="confirmed", title="e", first_seen_at=old)
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it_pending.id))
        s.add(Publication(event_id=ev.id, channel="telegram", kind="post", status="draft",
                          headline="h", body="b", features={"critic_ok": True}))
        s.flush()
        old_id, new_id, pending_id = old_asset.id, new_asset.id, pending_asset.id
        s.commit()

    store = _RecordingStore()
    removed = purge_stale_media(sf, store, older_than_hours=48)

    assert removed == 1 and store.deleted == ["aa/old"]
    with Session(pg_engine) as s:
        assert s.get(MediaAsset, old_id).purged_at is not None      # old, unpublished -> purged
        assert s.get(MediaAsset, new_id).purged_at is None          # too recent -> kept
        assert s.get(MediaAsset, pending_id).purged_at is None      # in-flight draft -> kept
