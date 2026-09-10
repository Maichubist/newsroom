"""RSS collector. `parse_feed` is pure (bytes -> RawItem list) and offline-testable;
`RssCollector` adds fetching (injectable) and persistence with source-health updates.
"""
from __future__ import annotations

import calendar
import datetime as dt
import html
import logging
import re
from dataclasses import dataclass
from typing import Callable

import feedparser

from newsroom.collectors.base import RawItem, RawMedia

# NB: DB-touching imports (ingest, models) are done lazily inside RssCollector
# so that `parse_feed` stays importable with only stdlib + feedparser (pure,
# offline). SQLAlchemy is pulled in only when you actually persist.

log = logging.getLogger("newsroom.collectors.rss")

_TAG_RE = re.compile(r"<[^>]+>")
_USER_AGENT = "Mozilla/5.0 (compatible; newsroom/0.1; +https://example.org)"


def _strip_html(value: str | None) -> str | None:
    if not value:
        return None
    return html.unescape(_TAG_RE.sub("", value)).strip() or None


def _parse_date(entry) -> dt.datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        struct = entry.get(key)
        if struct:
            return dt.datetime.fromtimestamp(calendar.timegm(struct), tz=dt.timezone.utc)
    return None


def _external_id(entry) -> str | None:
    for key in ("id", "guid", "link"):
        val = entry.get(key)
        if val:
            return str(val)
    return None


def _media(entry) -> list[RawMedia]:
    out: list[RawMedia] = []

    def kind_for(mime: str | None, url: str | None) -> str:
        m = (mime or "").lower()
        u = (url or "").lower()
        if m.startswith("image") or u.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
            return "image"
        if m.startswith("video") or u.endswith((".mp4", ".webm", ".mov")):
            return "video"
        return "embed"

    for enc in entry.get("enclosures", []) or []:
        url = enc.get("href") or enc.get("url")
        if not url:
            continue
        size = enc.get("length")
        out.append(RawMedia(kind=kind_for(enc.get("type"), url), url=url,
                            size_bytes=int(size) if str(size or "").isdigit() else None))
    for mc in entry.get("media_content", []) or []:
        url = mc.get("url")
        if not url:
            continue
        out.append(RawMedia(
            kind=kind_for(mc.get("type") or ("video" if mc.get("medium") == "video" else None), url),
            url=url,
            width=int(mc["width"]) if str(mc.get("width") or "").isdigit() else None,
            height=int(mc["height"]) if str(mc.get("height") or "").isdigit() else None,
        ))
    for th in entry.get("media_thumbnail", []) or []:
        url = th.get("url")
        if url:
            out.append(RawMedia(kind="image", url=url))
    # de-duplicate by url, keep first
    seen, uniq = set(), []
    for m in out:
        if m.url and m.url not in seen:
            seen.add(m.url)
            uniq.append(m)
    return uniq


def parse_feed(source_id: int, raw_bytes: bytes, *, default_lang: str | None = None) -> list[RawItem]:
    """Parse feed bytes into unified RawItems. Skips entries without a stable id."""
    feed = feedparser.parse(raw_bytes)
    feed_lang = (feed.feed.get("language") if getattr(feed, "feed", None) else None) or default_lang
    items: list[RawItem] = []
    for entry in feed.entries:
        ext = _external_id(entry)
        if not ext:
            continue  # cannot dedup a source item without a stable external id
        text = _strip_html(
            (entry.get("content", [{}])[0].get("value") if entry.get("content") else None)
            or entry.get("summary")
            or entry.get("description")
        )
        items.append(RawItem(
            source_id=source_id,
            external_id=ext,
            url=entry.get("link"),
            title=_strip_html(entry.get("title")),
            text=text,
            lang=feed_lang,
            published_at=_parse_date(entry),
            media=_media(entry),
            raw_payload={
                "id": ext,
                "link": entry.get("link"),
                "title": entry.get("title"),
                "published": entry.get("published") or entry.get("updated"),
                "summary": entry.get("summary"),
            },
        ))
    return items


@dataclass
class CollectResult:
    source_id: int
    ok: bool
    total: int = 0
    created: int = 0
    updated: int = 0
    error: str | None = None


def _http_fetch(url: str, *, timeout: float = 20.0) -> bytes:
    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                     headers={"User-Agent": _USER_AGENT})
    resp.raise_for_status()
    return resp.content


class RssCollector:
    def __init__(self, session_factory, fetch: Callable[[str], bytes] | None = None):
        self.session_factory = session_factory
        self._fetch = fetch or _http_fetch

    def collect_source(self, source_id: int) -> CollectResult:
        from newsroom.collectors.ingest import (
            mark_source_failure,
            mark_source_success,
            upsert_raw_item,
        )
        from newsroom.models import Source

        with self.session_factory() as s:
            src = s.get(Source, source_id)
            if src is None:
                return CollectResult(source_id, ok=False, error="source not found")
            handle, default_lang = src.handle_or_url, src.lang

        try:
            raw = self._fetch(handle)  # network happens outside any DB transaction
        except Exception as exc:  # noqa: BLE001 — a dead source must degrade, not crash
            log.warning("rss fetch failed", extra={"source_id": source_id, "error": str(exc)})
            with self.session_factory() as s:
                mark_source_failure(s.get(Source, source_id), exc)
                s.commit()
            return CollectResult(source_id, ok=False, error=str(exc))

        items = parse_feed(source_id, raw, default_lang=default_lang)
        created = updated = 0
        with self.session_factory() as s:
            src = s.get(Source, source_id)
            try:
                for ri in items:
                    _, was_created = upsert_raw_item(s, ri)
                    created += int(was_created)
                    updated += int(not was_created)
                mark_source_success(src)
                s.commit()
            except Exception as exc:  # noqa: BLE001
                s.rollback()
                mark_source_failure(s.get(Source, source_id), exc)
                s.commit()
                log.exception("rss persist failed", extra={"source_id": source_id})
                return CollectResult(source_id, ok=False, total=len(items), error=str(exc))
        log.info("rss collected", extra={"source_id": source_id, "total": len(items),
                                         "created": created, "updated": updated})
        return CollectResult(source_id, ok=True, total=len(items), created=created, updated=updated)
