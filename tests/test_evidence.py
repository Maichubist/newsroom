from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.factcheck.evidence import (
    CorpusEvidenceSearcher,
    EvidenceRef,
    select_corpus_evidence,
    store_evidence,
)


# --- select_corpus_evidence (offline) -----------------------------------------

def test_select_orders_by_similarity_and_caps_top_k():
    rows = [(1, 0.80), (2, 0.95), (3, 0.90), (4, 0.99)]
    refs = select_corpus_evidence(rows, top_k=2, min_similarity=0.75)
    assert [r.item_id for r in refs] == [4, 2]
    assert all(r.stance == "neutral" and r.evidence_kind == "corpus" for r in refs)


def test_select_drops_below_threshold():
    rows = [(1, 0.60), (2, 0.74999), (3, 0.76)]
    refs = select_corpus_evidence(rows, top_k=5, min_similarity=0.75)
    assert [r.item_id for r in refs] == [3]


def test_select_empty_when_nothing_qualifies():
    assert select_corpus_evidence([(1, 0.1), (2, 0.2)], top_k=5, min_similarity=0.75) == []


# --- CorpusEvidenceSearcher + store_evidence (pg) ------------------------------

class FakeEmbedder:
    """Deterministic 1536-dim vectors: a one-hot on the given axis, so cosine
    similarity is 1.0 for the same axis and 0.0 otherwise."""

    model = "fake-embed"

    def __init__(self, axis: int):
        self.axis = axis

    def embed(self, text: str) -> np.ndarray:
        from newsroom.db.base import EMBEDDING_DIM

        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        vec[self.axis] = 1.0
        return vec


@pytest.mark.pg
def test_corpus_search_finds_similar_and_excludes(pg_engine):
    from newsroom.db import make_session_factory
    from newsroom.models import Item, ItemEmbedding, Source

    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        src = Source(kind="rss", handle_or_url="https://ev.example/feed", name="Ev",
                     origin="ua", tier="media")
        s.add(src)
        s.flush()
        near1 = Item(source_id=src.id, external_id="near1", content_hash="h1", title="a")
        near2 = Item(source_id=src.id, external_id="near2", content_hash="h2", title="b")
        far = Item(source_id=src.id, external_id="far", content_hash="h3", title="c")
        s.add_all([near1, near2, far])
        s.flush()
        emb = FakeEmbedder(axis=7)
        for it, axis in ((near1, 7), (near2, 7), (far, 500)):
            v = np.zeros(1536, dtype=np.float32)
            v[axis] = 1.0
            s.add(ItemEmbedding(item_id=it.id, model=emb.model, vector=v.tolist()))
        s.commit()
        near1_id, near2_id, far_id = near1.id, near2.id, far.id

    searcher = CorpusEvidenceSearcher(sf, FakeEmbedder(axis=7), top_k=5, min_similarity=0.75)
    refs = searcher.search("будь-яке твердження")
    found = {r.item_id for r in refs}
    assert found == {near1_id, near2_id} and far_id not in found

    # excluding the claim's own item leaves only the other near item
    refs2 = searcher.search("будь-яке твердження", exclude_item_ids=[near1_id])
    assert {r.item_id for r in refs2} == {near2_id}


@pytest.mark.pg
def test_store_evidence_persists(pg_engine):
    from newsroom.db import make_session_factory  # noqa: F401 (parity w/ other pg tests)
    from newsroom.models import Claim, ClaimEvidence, Event

    with Session(pg_engine) as s:
        ev = Event(status="reported", title="Подія", first_seen_at=dt.datetime.now(dt.timezone.utc))
        s.add(ev)
        s.flush()
        claim = Claim(event_id=ev.id, text="твердження")
        s.add(claim)
        s.flush()
        ids = store_evidence(s, claim.id, [
            EvidenceRef(stance="neutral", evidence_kind="corpus", item_id=None, score=0.9),
            EvidenceRef(stance="refutes", evidence_kind="factcheck_db",
                        external_ref="https://voxcheck.org/x"),
        ])
        s.commit()

        assert len(ids) == 2
        rows = s.execute(select(ClaimEvidence).where(ClaimEvidence.claim_id == claim.id)
                         .order_by(ClaimEvidence.id)).scalars().all()
        assert rows[0].evidence_kind == "corpus" and rows[0].stance == "neutral"
        assert rows[1].evidence_kind == "factcheck_db"
        assert rows[1].external_ref == "https://voxcheck.org/x"
