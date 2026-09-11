from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.editorial.updates import (
    ROUTE_POST,
    ROUTE_SUMMARY,
    UPDATE_CONSEQUENCE,
    UPDATE_MINOR,
    UPDATE_NEW_FACT,
    UpdateDecision,
    parse_update,
    route_update,
)


# --- route_update (offline) ---------------------------------------------------

@pytest.mark.parametrize("utype,significant,expected", [
    ("new_fact", False, ROUTE_POST),
    ("refutation", False, ROUTE_POST),
    ("consequence", True, ROUTE_POST),
    ("consequence", False, ROUTE_SUMMARY),
    ("confirmation", True, ROUTE_SUMMARY),
    ("reaction", False, ROUTE_SUMMARY),
    ("minor", True, ROUTE_SUMMARY),
])
def test_route_update(utype, significant, expected):
    assert route_update(utype, significant) == expected


# --- parse_update (offline) ---------------------------------------------------

def test_parse_update_full():
    d = parse_update('{"update_type": "new_fact", "significant": true, '
                     '"position_changed": true, "summary": "що відомо"}')
    assert d.update_type == UPDATE_NEW_FACT and d.significant is True
    assert d.position_changed is True and d.summary == "що відомо"


def test_parse_update_unknown_type_collapses_to_minor():
    d = parse_update('{"update_type": "breaking!!!", "summary": "x"}')
    assert d.update_type == UPDATE_MINOR and route_update(d.update_type, d.significant) == ROUTE_SUMMARY


def test_parse_update_invalid_is_none():
    assert parse_update("not json") is None
    assert parse_update(None) is None
    assert parse_update("[1,2]") is None


# --- StoryUpdater (pg) --------------------------------------------------------

class FakeClassifier:
    def __init__(self, decision: UpdateDecision):
        self.model = "fake-update"
        self._decision = decision

    def classify(self, current_summary, event_title, fact_base):
        return self._decision


def _linked_event(pg_engine, *, status="confirmed"):
    """A story with one event and the bare StoryVersion the linker would create."""
    from newsroom.models import Event, Story, StoryVersion

    with Session(pg_engine) as s:
        story = Story(slug="s1", title="Сюжет", state="developing",
                      last_event_at=dt.datetime.now(dt.timezone.utc))
        s.add(story)
        s.flush()
        ev = Event(status=status, title="Нова подія", story_id=story.id,
                   first_seen_at=dt.datetime.now(dt.timezone.utc))
        s.add(ev)
        s.flush()
        s.add(StoryVersion(story_id=story.id, version=1, reason_event_id=ev.id))  # no summary yet
        s.commit()
        return story.id, ev.id


@pytest.mark.pg
def test_classify_event_posts_and_fills_summary_and_version(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Decision, Event, Story, StoryVersion

    sf = make_session_factory(pg_engine)
    story_id, event_id = _linked_event(pg_engine)

    updater = StoryUpdaterFactory(sf, UpdateDecision(
        update_type=UPDATE_NEW_FACT, significant=False, summary="Що відомо зараз.", position_changed=True))
    result = updater.classify_event(event_id)
    assert result.route == ROUTE_POST and result.update_type == UPDATE_NEW_FACT and not result.skipped

    with Session(pg_engine) as s:
        assert s.get(Event, event_id).update_type == UPDATE_NEW_FACT
        assert s.get(Story, story_id).current_summary == "Що відомо зараз."
        # the pre-existing bare version got the summary (no duplicate version created)
        versions = s.execute(select(StoryVersion).where(StoryVersion.story_id == story_id)).scalars().all()
        assert len(versions) == 1 and versions[0].summary == "Що відомо зараз."
        dec = s.execute(select(Decision).where(Decision.entity_id == str(event_id),
                                               Decision.stage == "edit")).scalars().one()
        assert dec.decision == UPDATE_NEW_FACT and dec.details["route"] == ROUTE_POST
        assert dec.details["position_changed"] is True

    # idempotent: update_type already set -> skipped
    assert updater.classify_event(event_id).skipped


@pytest.mark.pg
def test_classify_event_summary_update_route(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event

    sf = make_session_factory(pg_engine)
    _story_id, event_id = _linked_event(pg_engine)
    updater = StoryUpdaterFactory(sf, UpdateDecision(update_type=UPDATE_CONSEQUENCE, significant=False,
                                                     summary="уточнення"))
    result = updater.classify_event(event_id)
    assert result.route == ROUTE_SUMMARY
    with Session(pg_engine) as s:
        assert s.get(Event, event_id).update_type == UPDATE_CONSEQUENCE


@pytest.mark.pg
def test_classify_pending_counts_posts_and_updates(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.editorial.updates import classify_pending

    sf = make_session_factory(pg_engine)
    _s1, e1 = _linked_event(pg_engine)
    _s2, e2 = _linked_event(pg_engine)

    updater = StoryUpdaterFactory(sf, UpdateDecision(update_type=UPDATE_NEW_FACT, summary="s"))
    stats = classify_pending(sf, updater, limit=50)
    assert stats["classified"] == 2 and stats["posts"] == 2 and stats["summary_updates"] == 0


def StoryUpdaterFactory(sf, decision):
    from newsroom.editorial.updates import StoryUpdater

    return StoryUpdater(sf, classifier=FakeClassifier(decision))
