from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsroom.analyze.risk import load_risk_matrix
from newsroom.analyze.signal import load_filters
from newsroom.analyze.stoplist import load_stoplist
from newsroom.analyze.verify import Classification, Verifier
from newsroom.db import make_session_factory
from newsroom.db.base import EMBEDDING_DIM
from newsroom.models import Decision, Event, EventItem, Item, Source

pytestmark = pytest.mark.pg

CONFIG = Path(__file__).resolve().parents[1] / "config"
RISK = load_risk_matrix(CONFIG / "risk.yaml")
FILTERS = load_filters(CONFIG / "filters.yaml")
STOP = load_stoplist(CONFIG / "stoplist.yaml")


def _vec(*idx: int) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    for i in idx:
        v[i] = 1.0
    return v


class FakeEmbedder:
    model = "fake-embed"

    def embed(self, text: str):
        if "iran" in text.lower():
            return _vec(1, 2, 3)
        if "budget" in text.lower():
            return _vec(10, 11)
        return _vec(20)


class FakeClassifier:
    model = "fake-clf"

    def __init__(self, result):
        self._result = result

    def classify(self, title, text):
        return self._result(title, text) if callable(self._result) else self._result


class CountingClassifier:
    model = "count-clf"

    def __init__(self, result):
        self._result = result
        self.calls = 0

    def classify(self, title, text):
        self.calls += 1
        return self._result


def _source(s: Session, handle: str, *, official: bool = False, tier: str = "media") -> int:
    src = Source(kind="rss", handle_or_url=handle, name=handle, origin="ua",
                 tier="official" if official else tier, is_official=official)
    s.add(src)
    s.flush()
    return src.id


def _item(s: Session, source_id: int, ext: str, title: str, text: str) -> int:
    it = Item(source_id=source_id, external_id=ext, title=title, text=text,
              content_hash=ext.ljust(64, "0"))
    s.add(it)
    s.flush()
    return it.id


def _verifier(pg_engine, classification) -> Verifier:
    return Verifier(
        make_session_factory(pg_engine),
        classifier=FakeClassifier(classification),
        embedder=FakeEmbedder(),
        risk_matrix=RISK, filters=FILTERS, stoplist_rules=STOP,
    )


def _status(pg_engine, item_id) -> str:
    with Session(pg_engine) as s:
        return s.get(Item, item_id).status


def test_noise_item_is_filtered_out(pg_engine):
    with Session(pg_engine) as s:
        sid = _source(s, "vrf-noise")
        iid = _item(s, sid, "n1", "Огляд", "На правах реклами. Знижки сьогодні.")
        s.commit()
    v = _verifier(pg_engine, Classification(is_event=True, rubrics=["economy"]))
    r = v.verify_item(iid)
    assert r.item_status == "filtered_out" and _status(pg_engine, iid) == "filtered_out"


def test_non_event_is_filtered_out(pg_engine):
    with Session(pg_engine) as s:
        sid = _source(s, "vrf-nonevent")
        iid = _item(s, sid, "ne1", "Роздуми", "Просто колонка без новини.")
        s.commit()
    v = _verifier(pg_engine, Classification(is_event=False))
    assert v.verify_item(iid).item_status == "filtered_out"


def test_high_risk_single_source_is_signal(pg_engine):
    with Session(pg_engine) as s:
        sid = _source(s, "vrf-single")
        iid = _item(s, sid, "h1", "Політика", "iran deal news, політичне рішення")
        s.commit()
    v = _verifier(pg_engine, Classification(is_event=True, rubrics=["politics"]))
    r = v.verify_item(iid)
    assert r.item_status == "clustered" and r.event_status == "signal"
    with Session(pg_engine) as s:
        ev = s.get(Event, r.event_id)
        assert ev.risk_level == "high" and ev.independent_source_count == 1


def test_high_risk_two_independent_sources_becomes_reported(pg_engine):
    with Session(pg_engine) as s:
        s1 = _source(s, "vrf-a")
        s2 = _source(s, "vrf-b")
        i1 = _item(s, s1, "a1", "Іран", "iran talks, політика")
        i2 = _item(s, s2, "b1", "Іран-2", "iran talks continue, політика")
        s.commit()
    v = _verifier(pg_engine, Classification(is_event=True, rubrics=["politics"]))
    r1 = v.verify_item(i1)
    r2 = v.verify_item(i2)          # same embedding -> same event, second independent source
    assert r2.event_id == r1.event_id
    assert r2.event_status == "reported"
    with Session(pg_engine) as s:
        assert s.get(Event, r2.event_id).independent_source_count == 2


def test_critical_with_official_source_is_confirmed(pg_engine):
    with Session(pg_engine) as s:
        sid = _source(s, "vrf-official", official=True)
        iid = _item(s, sid, "c1", "Удар", "iran strike, оборона, офіційно")
        s.commit()
    v = _verifier(pg_engine, Classification(is_event=True, rubrics=["war"]))
    r = v.verify_item(iid)
    assert r.event_status == "confirmed"


def test_classifier_runs_once_per_event(pg_engine):
    # two reprints of the same news -> one event; the LLM classifier is paid for
    # once (per event), not once per item. This is the cost saving of the reorder.
    with Session(pg_engine) as s:
        s1 = _source(s, "vrf-once-a")
        s2 = _source(s, "vrf-once-b")
        i1 = _item(s, s1, "o1", "Іран", "iran talks, політика")
        i2 = _item(s, s2, "o2", "Іран-2", "iran talks continue, політика")
        s.commit()
    clf = CountingClassifier(Classification(is_event=True, rubrics=["politics"]))
    v = Verifier(make_session_factory(pg_engine), classifier=clf, embedder=FakeEmbedder(),
                 risk_matrix=RISK, filters=FILTERS, stoplist_rules=STOP)
    r1 = v.verify_item(i1)
    r2 = v.verify_item(i2)
    assert r1.event_id == r2.event_id          # same embedding -> same event
    assert clf.calls == 1                       # classified once per event, not per item
    with Session(pg_engine) as s:
        assert s.get(Event, r1.event_id).classifier_model == "count-clf"


def test_non_event_cluster_is_retired(pg_engine):
    # an item that clears the noise filter but the classifier deems a non-event:
    # its cluster is retired (centroid cleared) so it never attracts more items.
    with Session(pg_engine) as s:
        sid = _source(s, "vrf-retire")
        iid = _item(s, sid, "r1", "Роздуми", "Просто колонка без новини про життя.")
        s.commit()
    v = _verifier(pg_engine, Classification(is_event=False))
    r = v.verify_item(iid)
    assert r.item_status == "filtered_out" and _status(pg_engine, iid) == "filtered_out"
    with Session(pg_engine) as s:
        ev_id = s.execute(select(EventItem.event_id).where(EventItem.item_id == iid)).scalar_one()
        ev = s.get(Event, ev_id)
        assert ev.status == "filtered_out" and ev.centroid is None


def test_decisions_are_journalled(pg_engine):
    with Session(pg_engine) as s:
        sid = _source(s, "vrf-journal")
        iid = _item(s, sid, "j1", "Бюджет", "budget approved, економіка")
        s.commit()
    v = _verifier(pg_engine, Classification(is_event=True, rubrics=["economy"]))
    v.verify_item(iid)
    with Session(pg_engine) as s:
        rows = s.execute(select(Decision).where(Decision.entity_type == "item", Decision.entity_id == str(iid))).scalars().all()
        assert any(d.stage == "verify" and d.charter_version == "0.3" for d in rows)


def test_verify_records_topic_path_and_leaf(pg_engine):
    # the classifier's learned topic path lands on the event and links a taxonomy leaf node
    from newsroom.models import TaxonomyNode

    with Session(pg_engine) as s:
        sid = _source(s, "vrf-taxo")
        iid = _item(s, sid, "t1", "Ударний БпЛА по Одесі", "Внаслідок удару БпЛА в Одесі пошкоджено будівлю.")
        s.commit()
    v = _verifier(pg_engine, Classification(
        is_event=True, rubrics=["war"], topic_path=["війна", "атака рф", "удар бпла", "одеса"]))
    v.verify_item(iid)
    with Session(pg_engine) as s:
        ev_id = s.execute(select(EventItem.event_id).where(EventItem.item_id == iid)).scalar_one()
        ev = s.get(Event, ev_id)
        assert ev.topic_path == ["війна", "атака рф", "удар бпла", "одеса"]
        assert ev.topic_leaf_id is not None
        leaf = s.get(TaxonomyNode, ev.topic_leaf_id)
        assert leaf.slug == "одеса" and leaf.depth == 3
