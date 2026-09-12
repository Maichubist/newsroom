"""Telegram HTML rendering for a published post.

Kept separate from compose_post (which produces the channel-neutral plain text
stored on the publication and checked by the critic and the shadow report): here
we add Telegram-specific presentation — a **bold** headline and sources rendered
as links instead of a "Джерела:" line — with every dynamic part HTML-escaped so a
stray <, > or & in the text can never break Telegram's HTML parser.
"""
from __future__ import annotations

from html import escape


def _tag(value: str) -> str:
    v = value.strip()
    return v if v.startswith("#") else "#" + v.lstrip("#")


def render_telegram_html(
    *,
    headline: str,
    body: str = "",
    watching: str = "",
    hashtags: list[str] | None = None,
    source_links: list[tuple[str, str | None]] | None = None,
    is_rumor: bool = False,
    reported: bool = False,
    max_hashtags: int = 2,
) -> str:
    """Render the post for Telegram (parse_mode=HTML). Mirrors the block order of
    compose_post — (Чутка) → headline → body → forward line → hashtags → sources —
    but the headline is bold and the sources are links, not a labelled line."""
    parts: list[str] = []
    if is_rumor:
        parts.append(escape("Чутка"))
    parts.append("<b>" + escape(headline.strip()) + "</b>")

    text_body = body.strip()
    if reported and text_body:
        text_body = "Повідомляють: " + text_body      # status shown where it matters (charter §3.2)
    if text_body:
        parts.append(escape(text_body))
    if watching.strip():
        parts.append(escape(watching.strip()))

    out = "\n\n".join(p for p in parts if p)

    tags = [_tag(t) for t in (hashtags or []) if str(t).strip()][:max_hashtags]
    if tags:
        out += "\n\n" + " ".join(escape(t) for t in tags)

    links = _render_sources(source_links or [])
    if links:
        out += "\n\n" + links
    return out


def _render_sources(source_links: list[tuple[str, str | None]]) -> str:
    """Source names as links (no "Джерела:" label); a name without a URL stays
    plain text. href is attribute-escaped."""
    rendered: list[str] = []
    for name, url in source_links:
        name = (name or "").strip()
        if not name:
            continue
        if url:
            rendered.append(f'<a href="{escape(url, quote=True)}">{escape(name)}</a>')
        else:
            rendered.append(escape(name))
    return " · ".join(rendered)
