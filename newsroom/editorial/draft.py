"""Post draft: structured content, JSON parsing, and rendering to a Telegram post.

compose_post assembles the charter §10 anatomy — (rumor label) → headline → lead
(+status) → why it matters → what it means → what we're watching → hashtags →
sources — deterministically, so it is unit-tested without an LLM. parse_draft
turns the generator's JSON into DraftContent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DraftContent:
    headline: str
    lead: str
    why_important: str = ""
    what_it_means: str = ""
    watching: str = ""                       # "за чим стежимо далі" (story posts)
    rubrics: list[str] = field(default_factory=list)
    fallback: bool = False                    # True = generation failed, minimal placeholder (never publish)


# A lead this short is not journalism — reject it so a thin generation holds and
# retries rather than emitting a one-word post.
MIN_LEAD_CHARS = 12


def content_is_publishable(content: DraftContent) -> bool:
    """A draft may become a publishable post only if it is real content: not the
    generation fallback, with a lead that says something of its own (non-empty,
    long enough, and not just the headline repeated). Everything else must hold
    and retry — never publish a placeholder (§3.5)."""
    if content.fallback:
        return False
    lead = content.lead.strip()
    if len(lead) < MIN_LEAD_CHARS:
        return False
    if lead.casefold() == content.headline.strip().casefold():
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
    lead = str(obj.get("lead") or "").strip()
    if not headline or not lead:
        return None
    # A lead that just repeats the headline is a degenerate generation: treat it
    # as unparsable so the caller retries instead of shipping the title twice.
    if lead.casefold() == headline.casefold():
        return None
    return DraftContent(
        headline=headline,
        lead=lead,
        why_important=str(obj.get("why_important") or "").strip(),
        what_it_means=str(obj.get("what_it_means") or "").strip(),
        watching=str(obj.get("watching") or "").strip(),
        rubrics=[str(r).strip().lower() for r in (obj.get("rubrics") or []) if str(r).strip()],
    )


def _tag(value: str) -> str:
    v = value.strip()
    return v if v.startswith("#") else "#" + v.lstrip("#")


def compose_post(
    content: DraftContent,
    *,
    status: str = "confirmed",
    is_rumor: bool = False,
    hashtags: list[str] | None = None,
    sources: list[str] | None = None,
    max_hashtags: int = 2,
) -> str:
    """Render a DraftContent into the final post text (charter §10)."""
    parts: list[str] = []
    if is_rumor:
        parts.append("Чутка")                # label at the start (charter §3.7.6)

    parts.append(content.headline.strip())

    lead = content.lead.strip()
    if status == "reported" and lead:
        lead = "Повідомляють: " + lead        # status shown in the post (charter §3.2)
    if lead:
        parts.append(lead)

    if content.why_important.strip():
        parts.append(content.why_important.strip())
    if content.what_it_means.strip():
        parts.append("Що це означає: " + content.what_it_means.strip())
    if content.watching.strip():
        parts.append("За чим стежити: " + content.watching.strip())

    body = "\n\n".join(p for p in parts if p)

    tags = [_tag(t) for t in (hashtags or []) if str(t).strip()][:max_hashtags]
    if tags:
        body += "\n\n" + " ".join(tags)
    if sources:
        body += "\n\n" + "Джерела: " + ", ".join(sources)
    return body
