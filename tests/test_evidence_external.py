from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session

from newsroom.factcheck.evidence import CompositeEvidenceSearcher, EvidenceRef
from newsroom.factcheck.external import (
    ExternalHit,
    FactCheckDBSearcher,
    hits_to_evidence,
    load_factcheck_sources,
)

CONFIG = Path(__file__).resolve().parents[1] / "config"


# --- config registry (offline) ------------------------------------------------

def test_load_factcheck_sources():
    dbs = {d.key for d in load_factcheck_sources(CONFIG / "factcheck_sources.yaml")}
    assert {"stopfake", "voxcheck", "cpd", "detector_media"} <= dbs
    assert all(d.enabled is False for d in load_factcheck_sources(CONFIG / "factcheck_sources.yaml"))


# --- hits_to_evidence (offline) -----------------------------------------------

def test_hits_to_evidence_maps_and_ranks():
    hits = [
        ExternalHit(url="https://voxcheck.org/a", title="A", stance="refutes", score=0.6),
        ExternalHit(url="https://voxcheck.org/b", title="B", stance="refutes", score=0.9),
        ExternalHit(url="", title="no url"),  # dropped: no url
    ]
    refs = hits_to_evidence(hits, min_score=0.0, max_hits=5)
    assert [r.external_ref for r in refs] == ["https://voxcheck.org/b", "https://voxcheck.org/a"]
    assert all(r.evidence_kind == "factcheck_db" and r.stance == "refutes" and r.item_id is None
               for r in refs)


def test_hits_to_evidence_respects_min_score_and_cap():
    hits = [ExternalHit(url=f"https://x/{i}", score=s) for i, s in enumerate([0.2, 0.5, 0.8])]
    refs = hits_to_evidence(hits, min_score=0.5, max_hits=1)
    assert len(refs) == 1 and refs[0].external_ref == "https://x/2"  # highest, above floor


# --- FactCheckDBSearcher + composite (offline) --------------------------------

class FakeClient:
    def __init__(self, db_key, hits):
        self.db_key = db_key
        self._hits = hits

    def query(self, claim_text):
        return self._hits


class BoomClient:
    db_key = "boom"

    def query(self, claim_text):
        raise RuntimeError("network down")


def test_factcheck_db_searcher_survives_a_failing_client():
    searcher = FactCheckDBSearcher([
        BoomClient(),
        FakeClient("voxcheck", [ExternalHit(url="https://voxcheck.org/x", stance="refutes")]),
    ])
    refs = searcher.search("твердження")
    assert [r.external_ref for r in refs] == ["https://voxcheck.org/x"]


class StaticSearcher:
    def __init__(self, refs):
        self._refs = refs

    def search(self, claim_text, *, exclude_item_ids=()):
        return list(self._refs)


class ExplodingSearcher:
    def search(self, claim_text, *, exclude_item_ids=()):
        raise RuntimeError("boom")


def test_composite_concatenates_dedups_and_isolates_failures():
    corpus = StaticSearcher([EvidenceRef(evidence_kind="corpus", item_id=1, score=0.9)])
    official = StaticSearcher([
        EvidenceRef(evidence_kind="official", item_id=1, score=0.95),  # same item, different kind -> kept
        EvidenceRef(evidence_kind="official", item_id=1, score=0.95),  # exact dup -> dropped
    ])
    external = StaticSearcher([EvidenceRef(evidence_kind="factcheck_db",
                                           external_ref="https://voxcheck.org/x", stance="refutes")])
    comp = CompositeEvidenceSearcher([corpus, ExplodingSearcher(), official, external])
    refs = comp.search("твердження")
    kinds = [(r.evidence_kind, r.item_id, r.external_ref) for r in refs]
    assert kinds == [
        ("corpus", 1, None),
        ("official", 1, None),
        ("factcheck_db", None, "https://voxcheck.org/x"),
    ]


def test_composite_caps_total():
    many = StaticSearcher([EvidenceRef(evidence_kind="corpus", item_id=i, score=0.9) for i in range(10)])
    assert len(CompositeEvidenceSearcher([many], max_total=3).search("t")) == 3


# --- OfficialRegistrySearcher (pg) --------------------------------------------

class AxisEmbedder:
    model = "fake-embed"

    def __init__(self, axis: int):
        self.axis = axis

    def embed(self, text):
        from newsroom.db.base import EMBEDDING_DIM

        v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v[self.axis] = 1.0
        return v


@pytest.mark.pg
def test_official_registry_searcher_restricts_to_official_sources(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.factcheck import OfficialRegistrySearcher
    from newsroom.models import Item, ItemEmbedding, Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        official = Source(kind="rss", handle_or_url="https://gov/feed", name="Gov",
                          origin="ua", tier="official", is_official=True)
        media = Source(kind="rss", handle_or_url="https://media/feed", name="Media",
                       origin="ua", tier="media", is_official=False)
        s.add_all([official, media])
        s.flush()
        off_item = Item(source_id=official.id, external_id="o", content_hash="o", title="o")
        med_item = Item(source_id=media.id, external_id="m", content_hash="m", title="m")
        s.add_all([off_item, med_item])
        s.flush()
        for it in (off_item, med_item):
            v = np.zeros(1536, dtype=np.float32)
            v[3] = 1.0
            s.add(ItemEmbedding(item_id=it.id, model="fake-embed", vector=v.tolist()))
        s.commit()
        off_id, med_id = off_item.id, med_item.id

    refs = OfficialRegistrySearcher(sf, AxisEmbedder(axis=3), min_similarity=0.5).search("твердження")
    found = {r.item_id for r in refs}
    assert off_id in found and med_id not in found
    assert all(r.evidence_kind == "official" for r in refs)
