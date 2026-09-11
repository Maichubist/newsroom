"""Stop-list engine (charter §4).

Config-driven rules with a scope (all | ua_side | ru_side) and an action
(block | needs_official | review). `check` returns the violations that apply to
the content's side. Deterministic/offline and deliberately high-precision — the
nuanced, context-dependent parts of §4 are enforced by the LLM critic (1в), and
prisoner-of-war / graphic-media checks are media (vision) checks, not text.

Side handling is conservative (architecture §3.5): when the side is unknown,
ua_side rules still apply (blocking a legitimate item is cheaper than leaking a
forbidden one). ru_side rules apply only when the content is known to be about
the Russian side (that side is permissive by the charter).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ACTIONS = ("block", "needs_official", "review")
SCOPES = ("all", "ua_side", "ru_side")
SIDES = ("ua", "ru", "unknown")
_ACTION_SEVERITY = {"review": 0, "needs_official": 1, "block": 2}


class StopListConfigError(ValueError):
    """Raised when stoplist.yaml is malformed or a pattern fails to compile."""


@dataclass(frozen=True)
class StopRule:
    id: str
    scope: str
    action: str
    description: str
    patterns: list[re.Pattern]


@dataclass(frozen=True)
class Violation:
    rule_id: str
    scope: str
    action: str
    description: str


def load_stoplist(path: str | Path) -> list[StopRule]:
    path = Path(path)
    if not path.exists():
        raise StopListConfigError(f"stoplist config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise StopListConfigError("stoplist.yaml must have a non-empty 'rules' list")

    rules: list[StopRule] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_rules):
        if not isinstance(raw, dict):
            raise StopListConfigError(f"rules[{i}] must be a mapping")
        rid = str(raw.get("id") or "").strip()
        scope = str(raw.get("scope") or "").strip()
        action = str(raw.get("action") or "").strip()
        if not rid or rid in seen:
            raise StopListConfigError(f"rules[{i}]: missing or duplicate id {rid!r}")
        seen.add(rid)
        if scope not in SCOPES:
            raise StopListConfigError(f"rule {rid}: scope must be one of {SCOPES}")
        if action not in ACTIONS:
            raise StopListConfigError(f"rule {rid}: action must be one of {ACTIONS}")
        pats = raw.get("any") or []
        if not isinstance(pats, list) or not pats:
            raise StopListConfigError(f"rule {rid}: 'any' must be a non-empty list of regexes")
        compiled: list[re.Pattern] = []
        for pat in pats:
            try:
                compiled.append(re.compile(str(pat)))
            except re.error as exc:
                raise StopListConfigError(f"rule {rid}: bad regex {pat!r}: {exc}") from exc
        rules.append(StopRule(rid, scope, action, str(raw.get("description") or ""), compiled))
    return rules


def _applies(scope: str, side: str) -> bool:
    if scope == "all":
        return True
    if scope == "ua_side":
        return side in ("ua", "unknown")   # conservative when unknown
    if scope == "ru_side":
        return side == "ru"
    return False


def check(title: str | None, text: str | None, rules: list[StopRule], *, side: str = "unknown") -> list[Violation]:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    blob = f"{title or ''}\n{text or ''}"
    violations: list[Violation] = []
    for rule in rules:
        if not _applies(rule.scope, side):
            continue
        if any(p.search(blob) for p in rule.patterns):
            violations.append(Violation(rule.id, rule.scope, rule.action, rule.description))
    return violations


def is_blocked(violations: list[Violation]) -> bool:
    """True if anything must hard-block publication."""
    return any(v.action == "block" for v in violations)


def worst_action(violations: list[Violation]) -> str | None:
    """Most severe action among violations (block > needs_official > review)."""
    if not violations:
        return None
    return max(violations, key=lambda v: _ACTION_SEVERITY[v.action]).action
