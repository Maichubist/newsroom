"""Post draft: prose content, JSON parsing, and rendering to a Telegram post.

A post is written as живу прозу, not filled into fixed labelled slots — the
generator finds the angle and writes 1–3 natural paragraphs itself, folding "why
it matters" into the text only when it is non-obvious. compose_post renders the
charter §10 anatomy — (rumor label) → headline → body → (a bare forward line) →
hashtags → sources — deterministically, so it is unit-tested without an LLM.
parse_draft turns the generator's JSON into DraftContent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DraftContent:
    headline: str
    body: str                                # 1–3 natural paragraphs (\n\n between them)
    watching: str = ""                       # optional concrete forward line, rendered unlabelled
    rubrics: list[str] = field(default_factory=list)
    fallback: bool = False                   # True = generation failed, minimal placeholder (never publish)


# A body this short is not journalism — reject it so a thin generation holds and
# retries rather than emitting a one-word post.
MIN_BODY_CHARS = 12


def content_is_publishable(content: DraftContent) -> bool:
    """A draft may become a publishable post only if it is real content: not the
    generation fallback, with a body that says something of its own (non-empty,
    long enough, and not just the headline repeated). Everything else must hold
    and retry — never publish a placeholder (§3.5)."""
    if content.fallback:
        return False
    body = content.body.strip()
    if len(body) < MIN_BODY_CHARS:
        return False
    if body.casefold() == content.headline.strip().casefold():
        return False
    return True


def parse_draft(raw: str | None) -> DraftContent | None:
    if not raw:
        return None
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    headline = str(obj.get("headline") or "").strip()
    # accept "body"; tolerate an old-style "lead" key so a stray response still parses
    body = str(obj.get("body") or obj.get("lead") or "").strip()
    if not headline or not body:
        return None
    # a body that just repeats the headline is a degenerate generation: treat it
    # as unparsable so the caller retries instead of shipping the title twice.
    if body.casefold() == headline.casefold():
        return None
    return DraftContent(
        headline=headline,
        body=body,
        watching=str(obj.get("watching") or "").strip(),
        rubrics=[str(r).strip().lower() for r in (obj.get("rubrics") or []) if str(r).strip()],
    )


def _tag(value: str) -> str:
    v = value.strip()
    return v if v.startswith("#") else "#" + v.lstrip("#")


def compose_post(
    content: DraftContent,
    *,
    is_rumor: bool = False,
    reported: bool = False,
    hashtags: list[str] | None = None,
    sources: list[str] | None = None,
    max_hashtags: int = 2,
) -> str:
    """Render a DraftContent into the final post text (charter §10).

    `reported` prefixes the body with "Повідомляють:" — the caller sets it only
    where the confidence level matters (a single-source high-risk item), not on
    routine news, so the hedge does not appear on every post.
    """
    parts: list[str] = []
    if is_rumor:
        parts.append("Чутка")                # label at the start (charter §3.7.6)

    parts.append(content.headline.strip())

    body = content.body.strip()
    if reported and body:
        body = "Повідомляють: " + body        # status shown where it matters (charter §3.2)
    if body:
        parts.append(body)

    if content.watching.strip():
        parts.append(content.watching.strip())   # bare forward line, no "За чим стежити:" label

    text = "\n\n".join(p for p in parts if p)

    tags = [_tag(t) for t in (hashtags or []) if str(t).strip()][:max_hashtags]
    if tags:
        text += "\n\n" + " ".join(tags)
    if sources:
        text += "\n\n" + "Джерела: " + ", ".join(sources)
    return text
