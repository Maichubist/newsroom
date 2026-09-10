"""Loader and validator for config/sources.yaml.

Pure and offline: parses YAML, validates, returns dataclasses. No DB, no network.
Validation is strict so a broken config fails fast at startup, not mid-collection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

KINDS = {"rss", "telegram", "site", "x", "threads"}
ORIGINS = {"ua", "world"}
TIERS = {"official", "media", "aggregator", "leak", "anonymous"}


class SourceConfigError(ValueError):
    """Raised when sources.yaml is malformed or violates a rule."""


@dataclass(frozen=True)
class SourceConfig:
    kind: str
    handle_or_url: str
    name: str
    origin: str
    tier: str
    is_official: bool = False
    lang: str | None = None
    region: str | None = None
    active: bool = True
    poll_interval: int = 300


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SourceConfigError(msg)


def _parse_one(idx: int, raw: dict) -> SourceConfig:
    where = f"sources[{idx}]"
    _require(isinstance(raw, dict), f"{where}: must be a mapping")

    name = str(raw.get("name") or "").strip()
    kind = str(raw.get("kind") or "").strip().lower()
    handle = str(raw.get("handle_or_url") or "").strip()
    origin = str(raw.get("origin") or "").strip().lower()
    tier = str(raw.get("tier") or "").strip().lower()

    _require(bool(name), f"{where}: 'name' is required")
    _require(kind in KINDS, f"{where} ({name}): 'kind' must be one of {sorted(KINDS)}, got {kind!r}")
    _require(bool(handle), f"{where} ({name}): 'handle_or_url' is required")
    _require(origin in ORIGINS, f"{where} ({name}): 'origin' must be one of {sorted(ORIGINS)}, got {origin!r}")
    _require(tier in TIERS, f"{where} ({name}): 'tier' must be one of {sorted(TIERS)}, got {tier!r}")

    poll_interval = raw.get("poll_interval", 300)
    _require(isinstance(poll_interval, int) and poll_interval > 0,
             f"{where} ({name}): 'poll_interval' must be a positive integer")

    lang = raw.get("lang")
    region = raw.get("region")
    return SourceConfig(
        kind=kind,
        handle_or_url=handle,
        name=name,
        origin=origin,
        tier=tier,
        is_official=bool(raw.get("is_official", False)),
        lang=str(lang).strip() if lang else None,
        region=str(region).strip() if region else None,
        active=bool(raw.get("active", True)),
        poll_interval=int(poll_interval),
    )


def load_sources(path: str | Path) -> list[SourceConfig]:
    path = Path(path)
    _require(path.exists(), f"sources config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _require(isinstance(data, dict) and "sources" in data,
             "sources.yaml must have a top-level 'sources:' list")
    raw_list = data["sources"]
    _require(isinstance(raw_list, list) and raw_list, "'sources' must be a non-empty list")

    parsed = [_parse_one(i, raw) for i, raw in enumerate(raw_list)]

    seen: set[tuple[str, str]] = set()
    for sc in parsed:
        key = (sc.kind, sc.handle_or_url)
        _require(key not in seen, f"duplicate source (kind={sc.kind}, handle_or_url={sc.handle_or_url})")
        seen.add(key)
    return parsed
