from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.sources.discovery import sync_subscriptions


@pytest.mark.pg
def test_sync_subscriptions_adds_new_skips_existing_private_and_dups(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        s.add(Source(kind="telegram", handle_or_url="@existing", name="Existing",
                     origin="ua", tier="media", active=True))
        s.commit()

    channels = [
        {"username": "existing", "title": "Existing"},       # already tracked -> skip (not counted)
        {"username": "NewChan", "title": "New Channel"},     # add (normalized case-insensitively)
        {"username": None, "title": "Private no username"},  # no username -> skipped
        {"username": "newchan", "title": "dup in batch"},    # duplicate within batch -> skip
    ]
    stats = sync_subscriptions(sf, channels)
    assert stats["added"] == 1 and stats["skipped"] == 1

    with Session(pg_engine) as s:
        handles = set(s.execute(
            select(Source.handle_or_url).where(Source.kind == "telegram")).scalars().all())
        assert handles == {"@existing", "@newchan"}
        new = s.execute(select(Source).where(Source.handle_or_url == "@newchan")).scalar_one()
        assert new.active is True and new.name == "New Channel" and new.origin == "ua"

    # idempotent: re-running adds nothing
    assert sync_subscriptions(sf, channels)["added"] == 0


@pytest.mark.pg
def test_sync_subscriptions_empty(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    assert sync_subscriptions(sf, [])["added"] == 0
