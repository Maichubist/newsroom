from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsroom.analyze.risk import load_risk_matrix
from newsroom.analyze.signal import load_filters
from newsroom.analyze.stoplist import load_stoplist
from newsroom.analyze.verify import Classification, Verifier, verify_pending
from newsroom.db import make_session_factory
from newsroom.db.base import EMBEDDING_DIM
from newsroom.models import Item, Source

pytestmark = pytest.mark.pg

CONFIG = Path(__file__).resolve().parents[1] / "config"


def _vec(*idx):
    v = [0.0] * EMBEDDING_DIM
    for i in idx:
        v[i] = 1.0
    return v


class FakeEmbedder:
    model = "fake-embed"

    def embed(self, text):
        return _vec(1, 2) if "iran" in text.lower() else _vec(30)


class FakeClassifier:
    model = "fake-clf"

    def classify(self, title, text):
        if "advert" in (text or "").lower():
            return Classification(is_event=False)
        return Classification(is_event=True, rubrics=["politics"])


def test_verify_pending_processes_new_items_only(pg_engine):
    sf = make_session_factory(pg_engine)
    with Session(pg_engine) as s:
        sid = _source_id = None
        src = Source(kind="rss", handle_or_url="vp-src", name="S", origin="ua", tier="media")
        s.add(src)
        s.flush()
        for i, (title, text) in enumerate([
            ("Іран 1", "iran talks"),
            ("Іран 2", "iran talks continue"),
            ("Реклама", "advert promo here"),
        ]):
            s.add(Item(source_id=src.id, external_id=f"vp-{i}", title=title, text=text,
                       content_hash=f"vp{i}".ljust(64, "0"), status="new"))
        # one already-processed item must be ignored
        s.add(Item(source_id=src.id, external_id="vp-done", title="x", text="x",
                   content_hash="done".ljust(64, "0"), status="clustered"))
        s.commit()

    verifier = Verifier(
        sf, classifier=FakeClassifier(), embedder=FakeEmbedder(),
        risk_matrix=load_risk_matrix(CONFIG / "risk.yaml"),
        filters=load_filters(CONFIG / "filters.yaml"),
        stoplist_rules=load_stoplist(CONFIG / "stoplist.yaml"),
    )

    stats = verify_pending(sf, verifier, limit=50)

    assert stats["processed"] == 3                      # only the 3 new items
    with Session(pg_engine) as s:
        remaining_new = s.scalar(select(func.count()).select_from(Item).where(Item.status == "new"))
        assert remaining_new == 0                        # all moved off 'new'
        # the two iran items clustered together; the advert was filtered out
        assert stats.get("clustered", 0) == 2
        assert stats.get("filtered_out", 0) == 1
