from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from sqlalchemy.orm import Session

from newsroom.factbase import (
    FactBaseBuilder,
    MergedFact,
    SourceFact,
    VectorFact,
    build_pending,
    fact_base_json,
    merge_facts,
    parse_facts,
)


def _axis_vec(axis: int, dim: int = 8) -> list[float]:
    v = [0.0] * dim
    v[axis] = 1.0
    return v


# --- parse_facts (offline) ----------------------------------------------------

def test_parse_facts_objects_and_numbers():
    facts = parse_facts('{"facts": [{"text": "ставка знижена", "kind": "fact", "number": 13, "unit": "%"},'
                        ' {"text": "реакція банку", "kind": "reaction"}]}')
    assert len(facts) == 2
    assert facts[0].number == 13.0 and facts[0].unit == "%" and facts[0].kind == "fact"
    assert facts[1].kind == "reaction" and facts[1].number is None


def test_parse_facts_bare_list_and_strings():
    facts = parse_facts('["перший факт", {"text": "другий"}]')
    assert [f.text for f in facts] == ["перший факт", "другий"]
    assert all(f.kind == "fact" for f in facts)


def test_parse_facts_invalid_is_none():
    assert parse_facts("nope") is None
    assert parse_facts(None) is None
    assert parse_facts('{"facts": "not a list"}') is None


def test_parse_facts_modality_attribution_time_frame():
    facts = parse_facts(
        '{"facts": [{"text": "Росія готує мобілізацію 600 тис.", "modality": "statement",'
        ' "attribution": "українська розвідка", "time_frame": "у 2026-2027"},'
        ' {"text": "Естонія може закрити кордон", "modality": "MARTIAN"}]}')
    assert facts[0].modality == "statement"
    assert facts[0].attribution == "українська розвідка" and facts[0].time_frame == "у 2026-2027"
    assert facts[1].modality == "fact"          # unknown modality clamps to fact
    assert facts[1].attribution is None and facts[1].time_frame is None


# --- merge_facts (offline, injected vectors) ----------------------------------

def test_merge_counts_independent_sources():
    # two sources assert the same fact (same axis) -> confirmed_by 2
    items = [
        VectorFact(SourceFact("НБУ знизив ставку"), source_id=1, vector=_axis_vec(0)),
        VectorFact(SourceFact("Нацбанк зменшив ставку"), source_id=2, vector=_axis_vec(0)),
        VectorFact(SourceFact("Курс долара зріс"), source_id=1, vector=_axis_vec(3)),
    ]
    merged = merge_facts(items, threshold=0.85)
    by_conf = {m.confirmed_by for m in merged}
    assert len(merged) == 2 and by_conf == {2, 1}
    two = next(m for m in merged if m.confirmed_by == 2)
    assert sorted(two.source_ids) == [1, 2] and len(two.variants) == 2


def test_merge_same_source_twice_is_one_source():
    items = [
        VectorFact(SourceFact("те саме"), source_id=5, vector=_axis_vec(0)),
        VectorFact(SourceFact("те саме знову"), source_id=5, vector=_axis_vec(0)),
    ]
    merged = merge_facts(items)
    assert len(merged) == 1 and merged[0].confirmed_by == 1


def test_merge_detects_number_divergence():
    items = [
        VectorFact(SourceFact("ставка", number=13.0), source_id=1, vector=_axis_vec(0)),
        VectorFact(SourceFact("ставка", number=15.0), source_id=2, vector=_axis_vec(0)),
    ]
    merged = merge_facts(items, threshold=0.85)
    assert len(merged) == 1
    assert merged[0].divergent is True and sorted(merged[0].values) == [13.0, 15.0]


def test_merge_agreeing_numbers_not_divergent():
    items = [
        VectorFact(SourceFact("ставка", number=13.0), source_id=1, vector=_axis_vec(0)),
        VectorFact(SourceFact("ставка", number=13.0), source_id=2, vector=_axis_vec(0)),
    ]
    merged = merge_facts(items, threshold=0.85)
    assert merged[0].divergent is False and merged[0].values == [13.0]


def test_merge_skips_reactions():
    items = [
        VectorFact(SourceFact("реакція", kind="reaction"), source_id=1, vector=_axis_vec(0)),
        VectorFact(SourceFact("факт", kind="fact"), source_id=1, vector=_axis_vec(1)),
    ]
    merged = merge_facts(items)
    assert [m.text for m in merged] == ["факт"]


def test_merge_keeps_cautious_modality_and_fills_attribution():
    # one source hedges (statement) -> the merged fact is NOT presented as an established fact,
    # regardless of order; attribution/time_frame are filled from whichever variant has them.
    a = VectorFact(SourceFact("Росія готує мобілізацію", modality="statement",
                              attribution="розвідка", time_frame="2026-2027"), source_id=1, vector=_axis_vec(0))
    b = VectorFact(SourceFact("РФ мобілізує", modality="fact"), source_id=2, vector=_axis_vec(0))
    for items in ([a, b], [b, a]):
        merged = merge_facts(items, threshold=0.85)
        assert len(merged) == 1 and merged[0].modality == "statement"
        assert merged[0].attribution == "розвідка" and merged[0].time_frame == "2026-2027"


def test_fact_base_json_shape():
    merged = [MergedFact(text="ф", source_ids=[1, 2], confirmed_by=2, values=[13.0],
                         divergent=False, variants=["ф"], modality="statement",
                         attribution="Мінфін", time_frame="2026")]
    base = fact_base_json(merged, reactions=[(3, SourceFact("реакція", kind="reaction"))])
    assert base["facts"][0]["confirmed_by"] == 2 and base["facts"][0]["source_ids"] == [1, 2]
    assert base["facts"][0]["modality"] == "statement" and base["facts"][0]["attribution"] == "Мінфін"
    assert base["facts"][0]["time_frame"] == "2026"
    assert base["reactions"] == [{"source_id": 3, "text": "реакція", "unit": None}]


# --- FactBaseBuilder + build_pending (pg) -------------------------------------

class FakeFactExtractor:
    """Each source asserts the shared fact plus one reaction; deterministic."""

    model = "fake-fact"

    def extract(self, source_name, title, text):
        return [SourceFact("НБУ знизив ставку", kind="fact", number=13.0),
                SourceFact(f"{source_name} прокоментував", kind="reaction")]


class AxisEmbedder:
    """Maps any text to the same axis so both sources' facts merge into one."""

    model = "fake-embed"

    def embed(self, text):
        from newsroom.db.base import EMBEDDING_DIM

        v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v[0] = 1.0
        return v


@pytest.mark.pg
def test_build_event_merges_across_sources_and_stores(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, EventItem, Item, Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        s1 = Source(kind="rss", handle_or_url="https://a/feed", name="A", origin="ua", tier="media")
        s2 = Source(kind="rss", handle_or_url="https://b/feed", name="B", origin="ua", tier="official",
                    is_official=True)
        s.add_all([s1, s2])
        s.flush()
        i1 = Item(source_id=s1.id, external_id="a1", content_hash="a1", title="A", text="НБУ знизив ставку.")
        i2 = Item(source_id=s2.id, external_id="b1", content_hash="b1", title="B", text="Нацбанк зменшив ставку.")
        s.add_all([i1, i2])
        s.flush()
        ev = Event(status="confirmed", title="НБУ знизив ставку",
                   first_seen_at=dt.datetime.now(dt.timezone.utc))
        s.add(ev)
        s.flush()
        s.add_all([EventItem(event_id=ev.id, item_id=i1.id), EventItem(event_id=ev.id, item_id=i2.id)])
        s.commit()
        event_id, s1_id, s2_id = ev.id, s1.id, s2.id

    builder = FactBaseBuilder(sf, extractor=FakeFactExtractor(), embedder=AxisEmbedder(), threshold=0.85)
    result = builder.build_event(event_id)
    assert not result.skipped and result.sources == 2 and result.reactions == 2
    assert result.facts == 1   # both sources' facts merged into one

    with Session(pg_engine) as s:
        ev = s.get(Event, event_id)
        base = ev.fact_base
        assert base["facts"][0]["confirmed_by"] == 2
        assert sorted(base["facts"][0]["source_ids"]) == sorted([s1_id, s2_id])
        assert len(base["reactions"]) == 2

    # idempotent: fact_base already set -> skipped
    assert builder.build_event(event_id).skipped
    # build_pending also skips it now
    assert build_pending(sf, builder, limit=50)["events"] == 0
