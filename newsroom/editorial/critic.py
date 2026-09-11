"""Deterministic critic: check a rendered post against the charter before it can
be a publishable draft. Reuses the stop-list (1б.4) and AI-accent (1в.4a) engines.

Hard issues (never publish): stop-list block, a rumor without its label, an empty
post. Soft issues (should rewrite): AI-accent, stop-list review/needs_official.
The LLM critic (semantic charter compliance) layers on top of this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from newsroom.analyze.ai_accent import check_ai_accent
from newsroom.analyze.stoplist import check as stoplist_check


@dataclass(frozen=True)
class CriticReport:
    ok: bool                                  # True when there are no hard issues
    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)


def critic_check(
    post_text: str,
    *,
    is_rumor: bool = False,
    side: str = "unknown",
    stoplist_rules,
    ai_accent_patterns,
) -> CriticReport:
    hard: list[str] = []
    soft: list[str] = []

    violations = stoplist_check(None, post_text, stoplist_rules, side=side)
    blocked = [v.rule_id for v in violations if v.action == "block"]
    review = [v.rule_id for v in violations if v.action in ("review", "needs_official")]
    if blocked:
        hard.append("stoplist_block:" + ",".join(sorted(blocked)))
    if review:
        soft.append("stoplist_review:" + ",".join(sorted(review)))

    accent = check_ai_accent(post_text, ai_accent_patterns)
    if accent:
        soft.append("ai_accent:" + ",".join(sorted({h.category for h in accent})))

    if is_rumor and "чутка" not in (post_text or "")[:60].lower():
        hard.append("missing_rumor_label")

    if not re.search(r"\S", post_text or ""):
        hard.append("empty_post")

    return CriticReport(ok=not hard, hard=hard, soft=soft)
