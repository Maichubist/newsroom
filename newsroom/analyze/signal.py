"""Deterministic signal filter (charter §2.5, §3.7.5).

Two independent, pure/offline checks driven by config/filters.yaml:
  * classify_noise  — obvious ad/promo -> drop. Kept narrow and high-precision;
    the semantic "event vs noise" judgment is an LLM step (stage 1в).
  * ipso_markers    — information-operation red flags (calls to share, panic
    urgency, anonymous insiders). These are MARKERS, not a drop: they raise the
    verification bar downstream, they do not block on their own.

Every deny pattern is guarded by false-positive tests (CLAUDE.md).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


class FilterConfigError(ValueError):
    """Raised when filters.yaml is malformed or a pattern fails to compile."""


@dataclass(frozen=True)
class FiltersConfig:
    noise: dict[str, list[re.Pattern]]
    ipso: dict[str, list[re.Pattern]]


def _compile_section(section: object, where: str) -> dict[str, list[re.Pattern]]:
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise FilterConfigError(f"{where} must be a mapping of category -> [regex]")
    out: dict[str, list[re.Pattern]] = {}
    for category, patterns in section.items():
        if not isinstance(patterns, list):
            raise FilterConfigError(f"{where}.{category} must be a list of regexes")
        compiled: list[re.Pattern] = []
        for pat in patterns:
            try:
                compiled.append(re.compile(str(pat)))
            except re.error as exc:
                raise FilterConfigError(f"{where}.{category}: bad regex {pat!r}: {exc}") from exc
        out[str(category)] = compiled
    return out


def load_filters(path: str | Path) -> FiltersConfig:
    path = Path(path)
    if not path.exists():
        raise FilterConfigError(f"filters config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return FiltersConfig(
        noise=_compile_section(data.get("noise"), "noise"),
        ipso=_compile_section(data.get("ipso"), "ipso"),
    )


def _combined(title: str | None, text: str | None) -> str:
    return f"{title or ''}\n{text or ''}"


def _matched_categories(blob: str, section: dict[str, list[re.Pattern]]) -> list[str]:
    hits = [cat for cat, pats in section.items() if any(p.search(blob) for p in pats)]
    return sorted(hits)


@dataclass(frozen=True)
class NoiseVerdict:
    is_noise: bool
    reasons: list[str]


def classify_noise(title: str | None, text: str | None, filters: FiltersConfig) -> NoiseVerdict:
    reasons = _matched_categories(_combined(title, text), filters.noise)
    return NoiseVerdict(is_noise=bool(reasons), reasons=reasons)


def ipso_markers(title: str | None, text: str | None, filters: FiltersConfig) -> list[str]:
    return _matched_categories(_combined(title, text), filters.ipso)
