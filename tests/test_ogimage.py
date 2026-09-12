from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.orm import Session

from newsroom.media.ogimage import OgImageResolver, extract_og_image, resolve_pending


# --- extract_og_image (offline, pure) -----------------------------------------

def test_extracts_og_image_property_then_content():
    html = '<head><meta property="og:image" content="https://x/pic.jpg"></head>'
    assert extract_og_image(html) == "https://x/pic.jpg"


def test_extracts_when_content_comes_before_property():
    html = '<meta content="https://x/a.png" property="og:image"/>'
    assert extract_og_image(html) == "https://x/a.png"


def test_prefers_og_image_over_twitter():
    html = ('<meta name="twitter:image" content="https://x/tw.jpg">'
            '<meta property="og:image" content="https://x/og.jpg">')
    assert extract_og_image(html) == "https://x/og.jpg"


def test_falls_back_to_twitter_image():
    html = '<meta name="twitter:image" content="https://x/tw.jpg">'
    assert extract_og_image(html) == "https://x/tw.jpg"


def test_none_when_no_meta_image():
    assert extract_og_image("<html><body>no meta</body></html>") is None
    assert extract_og_image(None) is None


# --- resolve_pending (pg) -----------------------------------------------------

@pytest.mark.pg
def test_resolve_pending_creates_asset_from_og_image(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Item, MediaAsset, Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="ep", name="ЕП", origin="ua", tier="media")
        s.add(src)
        s.flush()
        # accepted item with a URL but no media (like Економічна Правда)
        it = Item(source_id=src.id, external_id="e1", content_hash="e1".ljust(64, "0"),
                  url="https://epravda.com.ua/a1", title="t", status="accepted")
        # an item that already has media -> must be skipped
        it2 = Item(source_id=src.id, external_id="e2", content_hash="e2".ljust(64, "0"),
                   url="https://epravda.com.ua/a2", title="t2", status="accepted")
        s.add_all([it, it2])
        s.flush()
        s.add(MediaAsset(item_id=it2.id, kind="image", url="https://x/already.jpg"))
        s.commit()
        want_id, skip_id = it.id, it2.id

    pages = {"https://epravda.com.ua/a1": '<meta property="og:image" content="https://img.ep/pic.jpg">'}
    resolver = OgImageResolver(sf, fetch=lambda url: pages.get(url, "<html></html>"))

    stats = resolve_pending(sf, resolver, limit=50)
    assert stats["checked"] == 1 and stats["found"] == 1     # only the media-less item was processed

    with Session(pg_engine) as s:
        assets = {a.item_id: a.url for a in s.query(MediaAsset).all()}
        assert assets[want_id] == "https://img.ep/pic.jpg"
        assert assets[skip_id] == "https://x/already.jpg"    # untouched
        dec = s.query(Decision).filter_by(entity_id=str(want_id), stage="media").one()
        assert dec.decision == "og_image"

    # idempotent: the checked item is not fetched again
    assert resolve_pending(sf, resolver, limit=50)["checked"] == 0


@pytest.mark.pg
def test_resolve_pending_marks_og_none_when_absent(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Item, MediaAsset, Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="s", name="S", origin="ua", tier="media")
        s.add(src)
        s.flush()
        it = Item(source_id=src.id, external_id="n1", content_hash="n1".ljust(64, "0"),
                  url="https://x/none", title="t", status="clustered")
        s.add(it)
        s.commit()
        item_id = it.id

    resolver = OgImageResolver(sf, fetch=lambda url: "<html>no image here</html>")
    stats = resolve_pending(sf, resolver, limit=50)
    assert stats["checked"] == 1 and stats["found"] == 0

    with Session(pg_engine) as s:
        assert s.query(MediaAsset).count() == 0
        assert s.query(Decision).filter_by(entity_id=str(item_id)).one().decision == "og_none"
    # marked og_none -> not retried
    assert resolve_pending(sf, resolver, limit=50)["checked"] == 0
