from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from newsroom.config import TIERS, SourceConfigError, load_sources

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_real_sources_yaml_is_valid():
    sources = load_sources(CONFIG_DIR / "sources.yaml")
    assert len(sources) == 25
    assert all(s.kind == "rss" for s in sources)
    assert all(s.origin in {"ua", "world"} for s in sources)
    assert all(s.tier in TIERS for s in sources)
    # bootstrap invariant: 12 UA + 13 world feeds
    assert sum(s.origin == "ua" for s in sources) == 12
    assert sum(s.origin == "world" for s in sources) == 13
    # official sources present (Ukrinform, Suspilne, KMDA) and flagged
    official = [s for s in sources if s.tier == "official"]
    assert {s.name for s in official} == {"Укрінформ", "Суспільне", "КМДА (Київ)"}
    assert all(s.is_official for s in official)
    # every (kind, handle) pair is unique
    keys = {(s.kind, s.handle_or_url) for s in sources}
    assert len(keys) == len(sources)


def _write(tmp_path, body: str):
    p = tmp_path / "sources.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_defaults_applied(tmp_path):
    p = _write(tmp_path, """
        sources:
          - {name: "X", kind: rss, handle_or_url: "https://x/rss", origin: ua, tier: media}
    """)
    (s,) = load_sources(p)
    assert s.poll_interval == 300 and s.active is True and s.is_official is False


@pytest.mark.parametrize("body, needle", [
    ("sources:\n  - {name: X, kind: carrier_pigeon, handle_or_url: h, origin: ua, tier: media}", "kind"),
    ("sources:\n  - {name: X, kind: rss, handle_or_url: h, origin: mars, tier: media}", "origin"),
    ("sources:\n  - {name: X, kind: rss, handle_or_url: h, origin: ua, tier: gossip}", "tier"),
    ("sources:\n  - {kind: rss, handle_or_url: h, origin: ua, tier: media}", "name"),
    ("sources:\n  - {name: X, kind: rss, handle_or_url: h, origin: ua, tier: media, poll_interval: 0}", "poll_interval"),
])
def test_invalid_rows_raise(tmp_path, body, needle):
    p = _write(tmp_path, body)
    with pytest.raises(SourceConfigError) as exc:
        load_sources(p)
    assert needle in str(exc.value)


def test_duplicate_source_rejected(tmp_path):
    p = _write(tmp_path, """
        sources:
          - {name: A, kind: rss, handle_or_url: "https://x/rss", origin: ua, tier: media}
          - {name: B, kind: rss, handle_or_url: "https://x/rss", origin: ua, tier: media}
    """)
    with pytest.raises(SourceConfigError) as exc:
        load_sources(p)
    assert "duplicate" in str(exc.value)


def test_missing_file(tmp_path):
    with pytest.raises(SourceConfigError):
        load_sources(tmp_path / "nope.yaml")


def test_empty_sources_rejected(tmp_path):
    p = _write(tmp_path, "sources: []\n")
    with pytest.raises(SourceConfigError):
        load_sources(p)
