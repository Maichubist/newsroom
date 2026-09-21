from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from newsroom.analyze.spine import SpineConfigError, load_spine, spine_proposals

CONFIG = Path(__file__).resolve().parents[1] / "config" / "taxonomy_spine.yaml"

# the confirmed oversight set (2026-09-20): war-family + politics + geopolitics + corruption
OVERSIGHT = {"war", "defense", "security", "mobilization", "politics", "geopolitics", "corruption"}


def test_real_spine_loads_14_rubrics():
    s = load_spine(CONFIG)
    assert len(s.rubrics) == 14
    # every rubric has a valid floor and a display name
    for r in s.rubrics.values():
        assert r.floor in ("low", "high", "critical")
        assert r.display


def test_slugs_resolve_to_themselves():
    s = load_spine(CONFIG)
    for slug in s.rubrics:
        assert s.resolve(slug) == slug


def test_old_flat_rubrics_map_to_spine():
    s = load_spine(CONFIG)
    assert s.resolve("war") == "war"
    assert s.resolve("strikes") == "war"          # strikes folded into war
    assert s.resolve("defense") == "defense"
    assert s.resolve("accusations") == "law_crime"
    assert s.resolve("crime") == "law_crime"
    assert s.resolve("lifestyle") == "culture"
    assert s.resolve("technology") == "tech_science"
    assert s.resolve("science") == "tech_science"


def test_old_pyramid_roots_map_to_spine():
    s = load_spine(CONFIG)
    assert s.resolve("війна") == "war"
    assert s.resolve("оборона") == "defense"
    assert s.resolve("безпека") == "security"
    assert s.resolve("корупція") == "corruption"
    assert s.resolve("право") == "law_crime"
    assert s.resolve("транспорт") == "society"
    assert s.resolve("погода") == "environment"


def test_resolve_is_case_and_space_insensitive():
    s = load_spine(CONFIG)
    assert s.resolve("  WAR ") == "war"
    assert s.resolve("Корупція") == "corruption"


def test_unknown_resolves_to_none_but_floor_is_conservative():
    s = load_spine(CONFIG)
    assert s.resolve("astrology") is None
    assert s.floor_for("astrology") == "high"      # default_floor, not low
    assert s.floor_for(None) == "high"


def test_floor_preserves_critical_for_war_family():
    s = load_spine(CONFIG)
    assert s.floor_for("war") == "critical"
    assert s.floor_for("strikes") == "critical"    # via alias
    assert s.floor_for("defense") == "critical"
    assert s.floor_for("security") == "critical"
    assert s.floor_for("corruption") == "high"
    assert s.floor_for("sport") == "low"


def test_oversight_set_is_exactly_the_confirmed_seven():
    s = load_spine(CONFIG)
    assert set(s.oversight_slugs()) == OVERSIGHT
    for slug in OVERSIGHT:
        assert s.needs_oversight(slug)
    assert not s.needs_oversight("economy")
    assert not s.needs_oversight("society")
    assert not s.needs_oversight("astrology")      # unknown never triggers oversight


def test_floor_for_any_takes_highest():
    # replaces RiskMatrix.level_for: highest floor among the rubrics, conservative default.
    s = load_spine(CONFIG)
    assert s.floor_for_any(["economy", "war"]) == "critical"
    assert s.floor_for_any(["sport", "politics"]) == "high"
    assert s.floor_for_any(["війна", "економіка"]) == "critical"   # old roots resolve too
    assert s.floor_for_any([]) == "high"                            # default
    assert s.floor_for_any(["astrology"]) == "high"


def test_corruption_is_high_floor_but_overseen():
    # the key decision: oversight set != critical set — corruption is high yet supervised.
    s = load_spine(CONFIG)
    assert s.floor_for("corruption") == "high"
    assert s.needs_oversight("corruption")


def test_spine_floors_mirror_risk_yaml():
    # guard against drift: every flat rubric in risk.yaml must resolve to the SAME floor
    # in the spine, so the two policy sources can't disagree while both exist.
    from newsroom.analyze.risk import load_risk_matrix

    s = load_spine(CONFIG)
    m = load_risk_matrix(Path(__file__).resolve().parents[1] / "config" / "risk.yaml")
    for rubric, level in m.rubric_level.items():
        assert s.floor_for(rubric) == level, f"{rubric}: spine {s.floor_for(rubric)} != risk.yaml {level}"


# --- validation ---------------------------------------------------------------

def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "spine.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_bad_floor_rejected(tmp_path):
    p = _write(tmp_path, """
        rubrics:
          - {slug: war, display: X, floor: extreme, oversight: true}
    """)
    with pytest.raises(SpineConfigError):
        load_spine(p)


def test_duplicate_slug_rejected(tmp_path):
    p = _write(tmp_path, """
        rubrics:
          - {slug: war, display: A, floor: critical}
          - {slug: war, display: B, floor: high}
    """)
    with pytest.raises(SpineConfigError):
        load_spine(p)


def test_alias_collision_rejected(tmp_path):
    p = _write(tmp_path, """
        rubrics:
          - {slug: war, display: A, floor: critical, aliases: [udar]}
          - {slug: defense, display: B, floor: critical, aliases: [udar]}
    """)
    with pytest.raises(SpineConfigError):
        load_spine(p)


def test_empty_rubrics_rejected(tmp_path):
    p = _write(tmp_path, "rubrics: []\n")
    with pytest.raises(SpineConfigError):
        load_spine(p)


# --- semi-automatic evolution proposals (offline) -----------------------------

def test_spine_proposals_flags_unmapped_high_volume():
    s = load_spine(CONFIG)
    props = spine_proposals({"war": 100, "новамегатема": 40, "дрібнинова": 5}, s, min_events=20)
    adds = [p for p in props if p.kind == "add"]
    assert any(p.topic == "новамегатема" and p.count == 40 for p in adds)
    assert not any(p.topic == "war" for p in adds)          # war maps to a rubric -> not proposed
    assert not any(p.topic == "дрібнинова" for p in adds)   # below min_events -> not proposed


def test_spine_proposals_cold_only_with_enough_data():
    s = load_spine(CONFIG)
    # not enough total data -> no cold proposals (a fresh deploy must not flag everything)
    assert [p for p in spine_proposals({"war": 50}, s) if p.kind == "cold"] == []
    # enough data, only war drew events -> the other 13 rubrics are cold candidates
    cold = [p.topic for p in spine_proposals({"war": 300}, s) if p.kind == "cold"]
    assert "war" not in cold and "sport" in cold and len(cold) == 13


def test_admin_spine_command_dispatches():
    from newsroom.publishers.admin import AdminConsole

    # spine=None short-circuits before any DB use, so a dummy session_factory is fine
    console = AdminConsole(lambda: None, spine=None)
    assert "не завантажено" in console.handle("/spine")


@pytest.mark.pg
def test_propose_spine_changes_reads_events(pg_engine):
    import datetime as dt

    from sqlalchemy.orm import Session

    from newsroom.analyze.spine import propose_spine_changes
    from newsroom.db import make_session_factory
    from newsroom.models import Event

    sf = make_session_factory(pg_engine)
    now = dt.datetime.now(dt.timezone.utc)
    with Session(pg_engine) as s:
        for i in range(25):
            s.add(Event(status="confirmed", rubric="загадковатема", title=f"t{i}", first_seen_at=now))
        s.add(Event(status="confirmed", rubric="war", title="w", first_seen_at=now))
        s.commit()
    spine = load_spine(CONFIG)
    with sf() as s:
        props = propose_spine_changes(s, spine, min_events=20)
    assert any(p.kind == "add" and p.topic == "загадковатема" and p.count == 25 for p in props)
