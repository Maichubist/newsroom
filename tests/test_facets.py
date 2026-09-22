from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select

from newsroom.analyze.facets import (
    FacetAssignment,
    compute_facet_metrics,
    ingest_event_facets,
    load_event_facet_signals,
    normalise_assignment,
    refresh_facet_metrics,
)


def test_normalise_assignment_rejects_unknown_axis_and_clamps_confidence():
    assert normalise_assignment(FacetAssignment("invented", ("x",))) is None
    got = normalise_assignment(FacetAssignment(
        " geography ", ("Україна", "Київська область", "Київська область"),
        4.2, " факт "))
    assert got is not None
    assert got.dimension == "geography"
    assert got.path == ("Україна", "Київська область")
    assert got.confidence == 1.0 and got.evidence == "факт"


@pytest.mark.pg
def test_ingest_event_facets_is_typed_evidenced_and_idempotent(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, EventFacet, FacetDimension, FacetValue

    sf = make_session_factory(pg_engine)
    with sf() as s:
        event = Event(status="confirmed", rubric="war", first_seen_at=dt.datetime.now(dt.timezone.utc))
        s.add(event)
        s.flush()
        assignments = [
            FacetAssignment("event_type", ("воєнна дія", "атака"), 0.95, "через атаку"),
            FacetAssignment("geography", ("Україна", "Київська область"), 0.9, "на Київщині"),
        ]
        assert ingest_event_facets(s, event.id, assignments, primary_rubric="war") == 3
        assert ingest_event_facets(s, event.id, assignments, primary_rubric="war") == 0
        s.commit()

    with sf() as s:
        links = list(s.execute(select(EventFacet)).scalars())
        assert len(links) == 3
        dims = {d.id: d.key for d in s.execute(select(FacetDimension)).scalars()}
        values = list(s.execute(select(FacetValue)).scalars())
        by_label = {v.label: v for v in values}
        assert dims[by_label["war"].dimension_id] == "domain"
        assert by_label["war"].status == "active"
        assert by_label["атака"].parent_id == by_label["воєнна дія"].id
        attack_link = next(x for x in links if x.facet_value_id == by_label["атака"].id)
        assert attack_link.confidence == pytest.approx(0.95)
        assert attack_link.evidence_text == "через атаку"


def _facet_event(sf, *, source_handle: str, facet_value: str, views: int,
                 reactions: int, forwards: int, comments: int = 0):
    from newsroom.models import Event, EventItem, Item, ItemMetric, Source, SourceMetric

    now = dt.datetime.now(dt.timezone.utc)
    with sf() as s:
        source = Source(kind="telegram", handle_or_url=source_handle, name=source_handle,
                        origin="ua", tier="media")
        s.add(source)
        s.flush()
        s.add(SourceMetric(source_id=source.id, subscribers=1000))
        item = Item(source_id=source.id, external_id="1", content_hash=(source_handle * 20)[:64],
                    title=facet_value, published_at=now - dt.timedelta(hours=1))
        s.add(item)
        s.flush()
        event = Event(status="confirmed", rubric="war", first_seen_at=now - dt.timedelta(hours=1))
        s.add(event)
        s.flush()
        s.add(EventItem(event_id=event.id, item_id=item.id, role="origin"))
        ingest_event_facets(s, event.id, [
            FacetAssignment("event_type", (facet_value,), 0.95, facet_value),
            FacetAssignment("geography", ("Україна",), 0.95, "Україна"),
        ], source_item_id=item.id, primary_rubric="war")
        s.add(ItemMetric(item_id=item.id, views=views, reactions={"👍": reactions},
                         forwards=forwards, comments=comments, measured_at=now))
        s.commit()
        return event.id


@pytest.mark.pg
def test_facet_heat_demand_and_curation_signal(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import FacetDimension, FacetValue

    sf = make_session_factory(pg_engine)
    hot_event = _facet_event(sf, source_handle="@hot", facet_value="атака",
                             views=900, reactions=100, forwards=40, comments=20)
    _facet_event(sf, source_handle="@cold", facet_value="навчання",
                 views=100, reactions=1, forwards=0)

    with sf() as s:
        metrics = compute_facet_metrics(s, min_pair_events=1, min_pair_sources=1)
        dim = s.execute(select(FacetDimension).where(FacetDimension.key == "event_type")).scalar_one()
        attack = s.execute(select(FacetValue).where(
            FacetValue.dimension_id == dim.id, FacetValue.slug == "атака")).scalar_one()
        training = s.execute(select(FacetValue).where(
            FacetValue.dimension_id == dim.id, FacetValue.slug == "навчання")).scalar_one()
        assert metrics["values"][attack.id]["demand"] == 1.0
        assert metrics["values"][training.id]["demand"] == 0.0

    refresh_facet_metrics(sf)
    with sf() as s:
        signals = load_event_facet_signals(s, [hot_event])
        assert hot_event in signals
        assert signals[hot_event][1] > 0
