from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from newsroom.analyze.spine import SpineConfigError, load_spine

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
