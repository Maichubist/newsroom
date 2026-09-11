from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsroom.monitoring import (
    find_source_retractions,
    issue_correction,
    monitor_publications,
)

pytestmark = pytest.mark.pg
UTC = dt.timezone.utc


def _published_event_with_source_item(pg_engine, *, deleted: bool, status="published"):
    """A published post over an event fed by one source item (optionally deleted)."""
    from newsroom.models import Event, EventItem, Item, Publication, Source

    with Session(pg_engine) as s:
        src = Source(kind="telegram", handle_or_url="t.me/x", name="X", origin="ua", tier="media")
        s.add(src)
        s.flush()
        item = Item(source_id=src.id, external_id="i1", content_hash="i1", title="post",
                    deleted_at=(dt.datetime.now(UTC) if deleted else None))
        s.add(item)
        s.flush()
        ev = Event(status="confirmed", title="Подія", first_seen_at=dt.datetime.now(UTC))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=item.id, role="origin"))
        pub = Publication(event_id=ev.id, channel="telegram", kind="post", status=status,
                          headline="h", body="b", channel_ref="123")
        s.add(pub)
        s.flush()
        s.commit()
        return {"event_id": ev.id, "item_id": item.id, "pub_id": pub.id, "source_id": src.id}


def test_find_retractions_only_for_deleted_source_items(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    live = _published_event_with_source_item(pg_engine, deleted=False)
    gone = _published_event_with_source_item(pg_engine, deleted=True)

    retractions = find_source_retractions(sf, limit=50)
    pubs = {r.publication_id for r in retractions}
    assert gone["pub_id"] in pubs and live["pub_id"] not in pubs


def test_find_retractions_ignores_unpublished(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    draft = _published_event_with_source_item(pg_engine, deleted=True, status="draft")
    retractions = find_source_retractions(sf, limit=50)
    assert draft["pub_id"] not in {r.publication_id for r in retractions}


def test_issue_correction_creates_draft_and_journals_once(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Publication

    sf = make_session_factory(pg_engine)
    info = _published_event_with_source_item(pg_engine, deleted=True)
    (retraction,) = [r for r in find_source_retractions(sf, limit=50) if r.publication_id == info["pub_id"]]

    corr_id = issue_correction(sf, retraction)
    assert corr_id is not None

    with Session(pg_engine) as s:
        corr = s.get(Publication, corr_id)
        assert corr.kind == "correction" and corr.status == "draft"
        assert corr.reply_to_publication_id == info["pub_id"]
        assert corr.body is None and corr.features["needs_generation"] is True
        dec = s.execute(select(Decision).where(Decision.entity_id == str(info["pub_id"]),
                                               Decision.stage == "publish")).scalars().one()
        assert dec.decision == "correction_drafted"

    # idempotent: a correction already exists -> no second one
    assert issue_correction(sf, retraction) is None
    with Session(pg_engine) as s:
        assert s.scalar(select(func.count()).select_from(Publication)
                        .where(Publication.kind == "correction",
                               Publication.reply_to_publication_id == info["pub_id"])) == 1


def test_monitor_publications_end_to_end(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Publication

    sf = make_session_factory(pg_engine)
    info = _published_event_with_source_item(pg_engine, deleted=True)

    stats = monitor_publications(sf, limit=50)
    assert stats["retractions"] >= 1 and stats["corrections"] >= 1

    # second tick: nothing new to correct
    assert monitor_publications(sf, limit=50)["corrections"] == 0
    with Session(pg_engine) as s:
        assert s.scalar(select(func.count()).select_from(Publication)
                        .where(Publication.reply_to_publication_id == info["pub_id"])) == 1
