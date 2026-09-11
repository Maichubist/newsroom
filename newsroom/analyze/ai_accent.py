"""Deterministic "AI-accent" critic (charter §6.1).

Checks a GENERATED post for the cliches, calques, template constructions and
weak endings the charter forbids, so the LLM critic (1в.4) can rewrite. Every
pattern is high-precision and guarded by false-positive tests (CLAUDE.md): the
calque patterns target the wrong form only, so correct Ukrainian does not trip.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


class AiAccentConfigError(ValueError):
    """Raised when ai_accent.yaml is malformed or a pattern fails to compile."""


@dataclass(frozen=True)
class AccentHit:
    category: str
    phrase: str


def load_ai_accent(path: str | Path) -> dict[str, list[re.Pattern]]:
    path = Path(path)
    if not path.exists():
        raise AiAccentConfigError(f"ai_accent config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, list[re.Pattern]] = {}
    for category, patterns in data.items():
        if category == "version":
            continue
        if not isinstance(patterns, list):
            raise AiAccentConfigError(f"{category} must be a list of regexes")
        compiled: list[re.Pattern] = []
        for pat in patterns:
            try:
                compiled.append(re.compile(str(pat)))
            except re.error as exc:
                raise AiAccentConfigError(f"{category}: bad regex {pat!r}: {exc}") from exc
        out[str(category)] = compiled
    return out


def check_ai_accent(text: str | None, patterns: dict[str, list[re.Pattern]]) -> list[AccentHit]:
    """All AI-accent hits in the text (category + the exact matched span)."""
    blob = text or ""
    hits: list[AccentHit] = []
    for category, pats in patterns.items():
        for pat in pats:
            for m in pat.finditer(blob):
                hits.append(AccentHit(category, m.group(0)))
    return hits
