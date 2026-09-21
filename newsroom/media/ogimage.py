"""og:image resolver (architecture §13).

Many feeds (e.g. Економічна Правда) carry no media in the RSS itself — the article
image lives only as an og:image meta tag on the page. The RSS collector reads only
feed media, so those items arrive with no MediaAsset and never get a picture. This
step fills the gap: for an item that passed the filter (accepted/clustered) and has
no media yet, fetch the article page and record its og:image as an image asset,
which then flows through the normal download → reuse-check → vision pipeline.

Fetching the page only for post-filter items honours §12 ("media only after the
filter"). The extractor is pure and offline-tested; the fetch is injectable.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("newsroom.media.ogimage")

# A real browser UA: many sites (Cloudflare, Суспільне, …) 403 a bot-ish UA, which was
# failing the og:image page fetch ~600 times (no real picture reached the post).
_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_BROWSER_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "uk,en;q=0.8",
}
_META_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_ATTR_RE = re.compile(r'(property|name)\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_CONTENT_RE = re.compile(r'content\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
# most-preferred first
_IMAGE_KEYS = ("og:image", "og:image:url", "og:image:secure_url", "twitter:image", "twitter:image:src")


def extract_og_image(html: str | None) -> str | None:
    """Return the best social-preview image URL from a page's <meta> tags, or None.
    Pure; tolerant of attribute order (property/name before or after content)."""
    if not html:
        return None
    found: dict[str, str] = {}
    for tag in _META_RE.findall(html):
        key_m = _ATTR_RE.search(tag)
        content_m = _CONTENT_RE.search(tag)
        if not key_m or not content_m:
            continue
        key = key_m.group(2).strip().lower()
        if key in _IMAGE_KEYS and key not in found:
            url = content_m.group(1).strip()
            if url:
                found[key] = url
    for key in _IMAGE_KEYS:
        if key in found:
            return found[key]
    return None


def _http_fetch_text(url: str, *, timeout: float = 20.0) -> str:  # pragma: no cover - network
    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers=_BROWSER_HEADERS)
    resp.raise_for_status()
    return resp.text


@dataclass(frozen=True)
class OgResult:
    item_id: int
    url: str | None = None
    created: bool = False
    skipped: bool = False


class OgImageResolver:
    def __init__(self, session_factory, *, fetch: Callable[[str], str] | None = None):
        self.sf = session_factory
        self.fetch = fetch or _http_fetch_text

    def resolve_item(self, item_id: int) -> OgResult:
        import datetime as dt

        from newsroom.models import Decision, Item, MediaAsset

        with self.sf() as s:
            item = s.get(Item, item_id)
            if item is None or not item.url:
                return OgResult(item_id, skipped=True)
            url = item.url

        try:
            html = self.fetch(url)
        except Exception as exc:  # noqa: BLE001 - a dead page must not stall the queue
            # transient failure: don't mark, so a later tick can retry the page
            log.warning("og fetch failed", extra={"item_id": item_id, "error": str(exc)})
            return OgResult(item_id, skipped=True)

        image_url = extract_og_image(html)
        with self.sf() as s:
            if image_url:
                s.add(MediaAsset(item_id=item_id, kind="image", url=image_url,
                                 first_seen_at=dt.datetime.now(dt.timezone.utc)))
            s.add(Decision(entity_type="item", entity_id=str(item_id), stage="media",
                           decision="og_image" if image_url else "og_none",
                           details={"url": image_url} if image_url else {}))
            s.commit()
        return OgResult(item_id, url=image_url, created=bool(image_url))


def resolve_pending(session_factory, resolver: "OgImageResolver", *, limit: int = 25) -> dict[str, int]:
    """One og:image tick: for post-filter items with a URL and no media yet, resolve
    the article's og:image into an image asset. Idempotent — an item that already has
    media, or was already checked (og_image / og_none), is not fetched again."""
    from sqlalchemy import String, cast, select

    from newsroom.models import Decision, Item, MediaAsset

    with session_factory() as s:
        have_media = select(MediaAsset.item_id).where(MediaAsset.item_id.is_not(None))
        checked = select(Decision.entity_id).where(
            Decision.entity_type == "item", Decision.stage == "media",
            Decision.decision.in_(("og_image", "og_none")),
        )
        ids = list(s.execute(
            select(Item.id)
            .where(
                Item.status.in_(("accepted", "clustered")),
                Item.url.is_not(None),
                Item.id.not_in(have_media),
                cast(Item.id, String).not_in(checked),
            )
            .order_by(Item.id)
            .limit(limit)
        ).scalars().all())

    stats = {"checked": 0, "found": 0}
    for item_id in ids:
        result = resolver.resolve_item(item_id)
        if result.skipped:
            continue
        stats["checked"] += 1
        stats["found"] += int(result.created)
    return stats
