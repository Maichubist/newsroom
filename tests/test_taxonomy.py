from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from newsroom.analyze.taxonomy import ingest_path, normalize_label, normalize_path


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
