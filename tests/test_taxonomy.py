from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy.orm import Session

from newsroom.analyze.taxonomy import (
    compute_node_heat,
    ingest_path,
    load_top_nodes,
    normalize_label,
    normalize_path,
    refresh_taxonomy_heat,
)

_NOW = None
_SEQ = 0


# --- normalization (offline) --------------------------------------------------

def test_normalize_label_lowercases_and_strips_punctuation():
    assert normalize_label("  Удар БпЛА!  ") == "удар бпла"
    assert normalize_label("Атака РФ") == "атака рф"
    assert normalize_label("iPhone 17") == "iphone 17"
    assert normalize_label(None) == "" and normalize_label("  ") == ""


def test_normalize_path_drops_empties_and_caps_depth():
    assert normalize_path(["Війна", " ", "Атака РФ"]) == [("війна", "Війна"), ("атака рф", "Атака РФ")]
    assert len(normalize_path([f"l{i}" for i in range(10)], max_depth=3)) == 3
    assert normalize_path(None) == []


# --- ingest_path (pg) ---------------------------------------------------------

@pytest.mark.pg
def test_ingest_path_builds_and_reuses_tree(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import TaxonomyNode

    sf = make_session_factory(pg_engine)
    with sf() as s:
        leaf1 = ingest_path(s, ["Війна", "Атака РФ", "удар бпла", "Одеса"])
        s.commit()
    with sf() as s:
        # a second event on a shorter shared prefix reuses the top nodes, forks at level 3
        leaf2 = ingest_path(s, ["Війна", "Атака РФ", "удар КАБ"])
        s.commit()

    with Session(pg_engine) as s:
        nodes = {(_slug(n), n.depth): n for n in s.query(TaxonomyNode).all()}
        # shared prefix -> one node each, forked leaves -> two
        assert ("війна", 0) in nodes and ("атака рф", 1) in nodes
        assert ("удар бпла", 2) in nodes and ("удар каб", 2) in nodes
        war = nodes[("війна", 0)]
        ataka = nodes[("атака рф", 1)]
        assert ataka.parent_id == war.id                 # chain linked
        assert war.event_count == 2 and ataka.event_count == 2   # both events roll up here
        assert nodes[("удар бпла", 2)].event_count == 1          # only the first
        assert leaf1 == nodes[("одеса", 3)].id and leaf2 == nodes[("удар каб", 2)].id


@pytest.mark.pg
def test_ingest_path_empty_returns_none(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with sf() as s:
        assert ingest_path(s, []) is None
        assert ingest_path(s, ["  ", "!"]) is None      # nothing normalizable
        s.commit()


def _slug(node):
    return node.slug


# --- Phase 2: node heat (pg) --------------------------------------------------

def _bare_event(sf, leaf_id, *, age_min):
    """An event placed at a taxonomy leaf, no engagement metrics (eng contribution 0)."""
    from newsroom.models import Event

    now = dt.datetime.now(dt.timezone.utc)
    with sf() as s:
        s.add(Event(status="confirmed", title="e", topic_leaf_id=leaf_id,
                    first_seen_at=now - dt.timedelta(minutes=age_min)))
        s.commit()


def _competitor_event(sf, leaf_id, *, subs, reacts, forwards, age_min=5):
    """A competitor event with engagement metrics under a taxonomy leaf."""
    global _SEQ
    _SEQ += 1
    tag = _SEQ
    from newsroom.models import Event, EventItem, Item, ItemMetric, Source, SourceMetric

    now = dt.datetime.now(dt.timezone.utc)
    with sf() as s:
        src = Source(kind="telegram", handle_or_url=f"@c{tag}", name="C", origin="ua", tier="media")
        s.add(src)
        s.flush()
        s.add(SourceMetric(source_id=src.id, subscribers=subs))
        it = Item(source_id=src.id, external_id=f"x{tag}", content_hash=f"h{tag}".ljust(64, "0"), title="t")
        s.add(it)
        s.flush()
        s.add(ItemMetric(item_id=it.id, views=1000, reactions={"👍": reacts}, forwards=forwards))
        ev = Event(status="confirmed", title="e", topic_leaf_id=leaf_id,
                   first_seen_at=now - dt.timedelta(minutes=age_min))
        s.add(ev)
        s.flush()
        s.add(EventItem(event_id=ev.id, item_id=it.id, role="origin"))
        s.commit()


@pytest.mark.pg
def test_node_heat_rolls_up_and_normalizes_per_level(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import TaxonomyNode

    sf = make_session_factory(pg_engine)
    with sf() as s:
        busy = ingest_path(s, ["війна", "атака рф", "удар бпла"])
        quiet = ingest_path(s, ["війна", "атака рф", "удар каб"])
        s.commit()
    _bare_event(sf, busy, age_min=5)
    _bare_event(sf, busy, age_min=10)      # busy leaf: 2 events
    _bare_event(sf, quiet, age_min=5)      # quiet leaf: 1 event

    with sf() as s:
        heat = compute_node_heat(s, window_hours=24, halflife_hours=3.0)
        by_slug = {n.slug: n.id for n in s.query(TaxonomyNode).all()}

    war, ataka = by_slug["війна"], by_slug["атака рф"]
    # per-level min-max at depth 2: the busier leaf is 1.0, the quiet one 0.0
    assert heat[busy]["heat"] == 1.0 and heat[quiet]["heat"] == 0.0
    assert heat[busy]["events"] == 2 and heat[quiet]["events"] == 1
    # roll-up: both events count toward the shared ancestors
    assert heat[ataka]["events"] == 3 and heat[war]["events"] == 3
    assert heat[war]["depth"] == 0 and heat[ataka]["depth"] == 1


@pytest.mark.pg
def test_node_heat_favours_recent(pg_engine):
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with sf() as s:
        fresh = ingest_path(s, ["економіка", "ринок", "акції"])
        stale = ingest_path(s, ["економіка", "ринок", "облігації"])
        s.commit()
    _bare_event(sf, fresh, age_min=5)          # ~now
    _bare_event(sf, stale, age_min=20 * 60)    # 20h ago -> heavily decayed

    with sf() as s:
        heat = compute_node_heat(s, window_hours=48, halflife_hours=3.0)
    assert heat[fresh]["heat"] == 1.0 and heat[stale]["heat"] == 0.0


@pytest.mark.pg
def test_node_heat_reflects_engagement_and_refresh_persists(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import TaxonomyNode

    sf = make_session_factory(pg_engine)
    with sf() as s:
        hot = ingest_path(s, ["технології", "пристрої", "смартфон"])
        cold = ingest_path(s, ["технології", "пристрої", "ноутбук"])
        s.commit()
    # equal recency + count, but the smartphone leaf drew far more engagement
    _competitor_event(sf, hot, subs=1000, reacts=200, forwards=50, age_min=5)
    _competitor_event(sf, cold, subs=1000, reacts=2, forwards=0, age_min=5)

    with sf() as s:
        heat = compute_node_heat(s, window_hours=24, halflife_hours=3.0)
    assert heat[hot]["heat"] == 1.0 and heat[cold]["heat"] == 0.0   # engagement decides at this level

    # refresh writes onto the nodes; load_top_nodes returns the hot path first
    assert refresh_taxonomy_heat(sf)["nodes"] >= 3
    with sf() as s:
        node = s.get(TaxonomyNode, hot)
        assert node.heat == 1.0 and node.heat_at is not None
        top = load_top_nodes(s, limit=10)
    assert top and top[0]["path"][-1] == "смартфон" and top[0]["path"][0] == "технології"


@pytest.mark.pg
def test_node_demand_computed_and_l2_signals_load(pg_engine):
    # demand is engagement-only (no recency); curation reads (heat, demand) from the L2 node
    from newsroom.analyze.taxonomy import load_event_signals
    from newsroom.db import make_session_factory
    from newsroom.models import Event, TaxonomyNode

    sf = make_session_factory(pg_engine)
    with sf() as s:
        hot = ingest_path(s, ["технології", "пристрої", "смартфон"])
        cold = ingest_path(s, ["технології", "пристрої", "ноутбук"])
        s.commit()
    _competitor_event(sf, hot, subs=1000, reacts=200, forwards=50, age_min=5)
    _competitor_event(sf, cold, subs=1000, reacts=2, forwards=0, age_min=5)

    with sf() as s:
        result = compute_node_heat(s, window_hours=24, halflife_hours=3.0)
    # within the leaf depth level, the high-engagement leaf beats the low-engagement one
    assert result[hot]["demand"] == 1.0 and result[cold]["demand"] == 0.0

    refresh_taxonomy_heat(sf)
    with sf() as s:
        l2 = s.get(TaxonomyNode, s.get(TaxonomyNode, hot).parent_id)   # "пристрої" (depth 1)
        ev = Event(status="confirmed", rubric="tech_science", title="t", topic_leaf_id=hot,
                   first_seen_at=dt.datetime.now(dt.timezone.utc))
        s.add(ev)
        s.flush()
        sig = load_event_signals(s, [ev.id])
        # the event's signal comes from the L2 node, NOT the leaf (leaf demand was 1.0)
        assert sig[ev.id][0] == pytest.approx(l2.heat)
        assert sig[ev.id][1] == pytest.approx(l2.demand)
        assert sig[ev.id][1] != pytest.approx(result[hot]["demand"])


@pytest.mark.pg
def test_refresh_clears_stale_heat(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Event, TaxonomyNode

    sf = make_session_factory(pg_engine)
    with sf() as s:
        leaf = ingest_path(s, ["спорт", "футбол", "матч"])
        s.commit()
    _bare_event(sf, leaf, age_min=5)
    refresh_taxonomy_heat(sf)
    with sf() as s:
        assert s.get(TaxonomyNode, leaf).heat > 0

    # the event ages out of the window -> heat resets to 0
    with sf() as s:
        ev = s.query(Event).one()
        ev.first_seen_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=48)
        s.commit()
    refresh_taxonomy_heat(sf, window_hours=24)
    with sf() as s:
        assert s.get(TaxonomyNode, leaf).heat == 0.0 and s.get(TaxonomyNode, leaf).heat_events == 0


# --- synonym-node merge (pg) --------------------------------------------------

class FakeSynonymGrouper:
    model = "fake-syn"

    def __init__(self, label_groups):
        self.label_groups = [set(g) for g in label_groups]
        self.seen: list[list[str]] = []

    def group(self, nodes):
        self.seen.append([lab for _, lab in nodes])
        by_label = {lab: nid for nid, lab in nodes}
        out = []
        for grp in self.label_groups:
            ids = [by_label[lab] for lab in grp if lab in by_label]
            if len(ids) >= 2:
                out.append(ids)
        return out


@pytest.mark.pg
def test_merge_synonym_nodes_folds_siblings(pg_engine):
    from sqlalchemy import select

    from newsroom.analyze.taxonomy import merge_synonym_nodes
    from newsroom.db import make_session_factory
    from newsroom.models import Event, TaxonomyNode

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with sf() as s:
        a = ingest_path(s, ["війна", "атака рф", "удар бпла"])
        b = ingest_path(s, ["війна", "атака рф", "атака дронів"])
        s.get(TaxonomyNode, a).event_count = 5      # canonical-to-be (more events)
        s.get(TaxonomyNode, b).event_count = 2
        s.add(Event(status="confirmed", title="e1", topic_leaf_id=a, first_seen_at=now))
        s.add(Event(status="confirmed", title="e2", topic_leaf_id=b, first_seen_at=now))
        s.commit()

    grouper = FakeSynonymGrouper([{"удар бпла", "атака дронів"}])
    stats = merge_synonym_nodes(sf, grouper)
    assert stats["merged"] == 1 and stats["groups"] == 1

    with sf() as s:
        slugs = {n.slug for n in s.execute(select(TaxonomyNode)).scalars()}
        assert "атака дронів" not in slugs and "удар бпла" in slugs      # victim gone, canonical kept
        canon = s.execute(select(TaxonomyNode).where(TaxonomyNode.slug == "удар бпла")).scalar_one()
        assert canon.event_count == 7                                    # 5 + 2 summed
        assert {e.topic_leaf_id for e in s.execute(select(Event)).scalars()} == {canon.id}


@pytest.mark.pg
def test_taxonomy_merge_skips_unchanged_sibling_set(pg_engine):
    from newsroom.analyze.taxonomy import merge_synonym_nodes
    from newsroom.db import make_session_factory

    sf = make_session_factory(pg_engine)
    with sf() as s:
        ingest_path(s, ["економіка", "податки"])
        ingest_path(s, ["економіка", "банки"])
        s.commit()

    grouper = FakeSynonymGrouper([])
    first = merge_synonym_nodes(sf, grouper)
    calls_after_first = len(grouper.seen)
    second = merge_synonym_nodes(sf, grouper)
    assert first["cached"] == 0 and calls_after_first >= 1
    assert second["cached"] >= 1 and len(grouper.seen) == calls_after_first


@pytest.mark.pg
def test_merge_node_recurses_on_slug_collision(pg_engine):
    from sqlalchemy import select

    from newsroom.analyze.taxonomy import merge_node
    from newsroom.db import make_session_factory
    from newsroom.models import Event, TaxonomyNode

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with sf() as s:
        # two sibling subtrees under "війна", each with a grandchild "одеса"
        ingest_path(s, ["війна", "атака дронів", "одеса"])
        ingest_path(s, ["війна", "удар бпла", "одеса"])
        s.commit()
    with sf() as s:
        victim = s.execute(select(TaxonomyNode).where(TaxonomyNode.slug == "атака дронів")).scalar_one().id
        canon = s.execute(select(TaxonomyNode).where(TaxonomyNode.slug == "удар бпла")).scalar_one().id
        vic_od = s.execute(select(TaxonomyNode).where(
            TaxonomyNode.parent_id == victim, TaxonomyNode.slug == "одеса")).scalar_one().id
        can_od = s.execute(select(TaxonomyNode).where(
            TaxonomyNode.parent_id == canon, TaxonomyNode.slug == "одеса")).scalar_one().id
        s.get(TaxonomyNode, vic_od).event_count = 3
        s.get(TaxonomyNode, can_od).event_count = 4
        s.add(Event(status="confirmed", title="e", topic_leaf_id=vic_od, first_seen_at=now))
        s.commit()
    with sf() as s:
        merge_node(s, victim, canon)
        s.commit()

    with sf() as s:
        assert s.get(TaxonomyNode, victim) is None                       # victim removed
        odesas = s.execute(select(TaxonomyNode).where(
            TaxonomyNode.parent_id == canon, TaxonomyNode.slug == "одеса")).scalars().all()
        assert len(odesas) == 1 and odesas[0].id == can_od               # grandchildren folded into one
        assert odesas[0].event_count == 7                                # 4 + 3
        assert s.execute(select(Event)).scalar_one().topic_leaf_id == can_od   # event re-pointed
