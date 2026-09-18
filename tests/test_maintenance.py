from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsroom.maintenance import prune_decisions

UTC = dt.timezone.utc


@pytest.mark.pg
def test_prune_decisions_removes_only_old_dedup_stages(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision

    sf = make_session_factory(pg_engine)
    old = dt.datetime(2020, 1, 1, tzinfo=UTC)
    with Session(pg_engine) as s:
        s.add(Decision(entity_type="event", entity_id="1", stage="predup",
                       decision="predup_separate", created_at=old))          # old dedup -> pruned
        s.add(Decision(entity_type="event", entity_id="2", stage="ingest_dedup",
                       decision="ingest_separate", created_at=old))          # old dedup -> pruned
        s.add(Decision(entity_type="publication", entity_id="3", stage="publish",
                       decision="published", created_at=old))                # old AUDIT -> kept
        s.add(Decision(entity_type="event", entity_id="4", stage="predup",
                       decision="predup_duplicate"))                         # recent dedup -> kept
        s.commit()

    deleted = prune_decisions(sf, older_than_days=30)
    assert deleted == 2
    with Session(pg_engine) as s:
        assert s.scalar(select(func.count()).select_from(Decision)) == 2     # audit + recent survive
        stages = set(s.execute(select(Decision.stage)).scalars().all())
        assert stages == {"publish", "predup"}


@pytest.mark.pg
def test_prune_decisions_noop_when_nothing_old(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        s.add(Decision(entity_type="event", entity_id="9", stage="predup", decision="predup_separate"))
        s.commit()
    assert prune_decisions(sf, older_than_days=30) == 0
