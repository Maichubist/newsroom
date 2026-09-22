"""Image moderation — the media stop-list (charter §4, architecture §9.5).

The text stop-list cannot see pictures; the charter forbids publishing graphic
violence, bodies of the dead, prisoners of war and similar. A vision model checks
each image and returns a verdict. The model is pluggable (OpenAI vision in
production, a fake in tests); parsing is pure and offline-tested.

Asymmetry of errors (§3.5): media is optional (the post falls back to text), so
anything the model cannot clear — unparsable output, an unsure answer, a missing
field — is treated as BLOCKED. We drop a doubtful image rather than publish it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger("newsroom.media.moderation")


@dataclass(frozen=True)
class ImageVerdict:
    blocked: bool = True                       # default-deny (§3.5)
    labels: list[str] = field(default_factory=list)
    reason: str = ""


def _guess_image_mime(data: bytes) -> str:
    """Best-effort content type from magic bytes (for the vision data URL)."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def moderation_source(store, url: str | None, storage_key: str | None) -> str | None:
    """What to hand the vision model: the public URL when there is one, else a base64
    data URL built from the locally stored bytes (Telegram media has no URL). None when
    neither is available (no store, or the file is gone) — the caller then skips it."""
    if url:
        return url
    if not storage_key or store is None:
        return None
    data = store.get(storage_key)
    if not data:
        return None
    import base64

    return f"data:{_guess_image_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


def parse_image_verdict(raw: str | None) -> ImageVerdict | None:
    """Parse the model's JSON verdict. A missing 'blocked' field defaults to
    blocked (conservative). None if unparsable so the caller can retry."""
    if not raw:
        return None
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    return ImageVerdict(
        blocked=bool(obj.get("blocked", True)),
        labels=[str(x).strip().lower() for x in (obj.get("labels") or []) if str(x).strip()],
        reason=str(obj.get("reason") or "").strip(),
    )


class ImageModerator(Protocol):
    model: str

    def check(self, image_url: str) -> ImageVerdict: ...


DEFAULT_MODERATION_PROMPT = """Ти модератор зображень для новинного каналу. За
редакційною хартією заборонено публікувати: тіла загиблих і поранених великим
планом, полонених (упізнавані обличчя), надмірне насильство/жах, страти,
18+ контент, документи з персональними даними. Оціни зображення.

Поверни лише JSON: {"blocked": true|false, "labels": ["..."], "reason": "..."}.
Якщо є сумнів — blocked=true."""


class OpenAIImageModerator:  # pragma: no cover - network / vision
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini",
                 prompt: str = DEFAULT_MODERATION_PROMPT):
        import os

        self.model = model
        self.prompt = prompt
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _call(self, image_url: str) -> str | None:
        try:
            from newsroom.llmutil import chat_json

            return chat_json(self._ensure_client(), model=self.model,
                             messages=[{"role": "user", "content": [
                                 {"type": "text", "text": self.prompt},
                                 {"type": "image_url", "image_url": {"url": image_url}},
                             ]}],
                             op="moderate_image", max_tokens=512)
        except Exception as exc:  # noqa: BLE001
            log.warning("image moderation failed", extra={"error": str(exc)})
            return None

    def check(self, image_url: str) -> ImageVerdict:
        for attempt in (1, 2):
            parsed = parse_image_verdict(self._call(image_url))
            if parsed is not None:
                return parsed
            log.warning("image verdict unparsable", extra={"attempt": attempt})
        return ImageVerdict(blocked=True, reason="unverifiable")   # default-deny


@dataclass(frozen=True)
class ModerationResult:
    event_id: int
    checked: int = 0
    blocked: int = 0
    skipped: bool = False


def moderate_event_media(session_factory, moderator: ImageModerator, event_id: int, *,
                         store=None, charter_version: str = "0.2") -> ModerationResult:
    """Moderate the event's images; journal media_vision_ok / media_vision_block.
    Idempotent, and a no-op (no decision) for events without images. Telegram images
    have no URL, so they are moderated from the locally stored bytes (needs `store`);
    an image with neither a URL nor readable bytes is skipped, not silently cleared."""
    from sqlalchemy import and_, func, or_, select

    from newsroom.models import Decision, EventItem, Item, MediaAsset

    with session_factory() as s:
        already = s.scalar(
            select(func.count()).select_from(Decision).where(
                Decision.entity_type == "event", Decision.entity_id == str(event_id),
                Decision.stage == "verify",
                Decision.decision.in_(("media_vision_ok", "media_vision_block")),
            )
        )
        if already:
            return ModerationResult(event_id, skipped=True)
        images = list(s.execute(
            select(MediaAsset.id, MediaAsset.url, MediaAsset.storage_key)
            .join(Item, Item.id == MediaAsset.item_id)
            .join(EventItem, EventItem.item_id == Item.id)
            .where(EventItem.event_id == event_id, MediaAsset.kind == "image",
                   or_(MediaAsset.url.is_not(None),
                       and_(MediaAsset.storage_key.is_not(None), MediaAsset.purged_at.is_(None))))
        ).all())

    if not images:
        return ModerationResult(event_id, checked=0, skipped=True)

    flags: list[dict] = []
    checked = 0
    for media_id, url, storage_key in images:
        source = moderation_source(store, url, storage_key)
        if source is None:
            continue                         # cannot fetch this image -> do not clear it
        checked += 1
        from newsroom.llmutil import llm_context

        with llm_context(event_id=event_id, media_id=media_id, stage="moderate_image"):
            verdict = moderator.check(source)
        if verdict.blocked:
            flags.append({"media_id": media_id, "labels": verdict.labels, "reason": verdict.reason})

    if checked == 0:
        return ModerationResult(event_id, checked=0, skipped=True)

    with session_factory() as s:
        s.add(Decision(
            entity_type="event", entity_id=str(event_id), stage="verify",
            decision="media_vision_block" if flags else "media_vision_ok",
            reason=f"{len(flags)} blocked of {checked}",
            details={"checked": checked, "flags": flags},
            charter_version=charter_version,
        ))
        s.commit()
    return ModerationResult(event_id, checked=checked, blocked=len(flags))


def moderate_pending(session_factory, moderator: ImageModerator, *, store=None,
                     limit: int = 25) -> dict[str, int]:
    """Moderate approved events after all downloads have settled."""
    from sqlalchemy import Integer, and_, cast, or_, select

    from newsroom.media.state import approved_event_ids_query, event_media_readiness
    from newsroom.models import Decision, Event, EventItem, Item, MediaAsset

    with session_factory() as s:
        checked = (
            select(cast(Decision.entity_id, Integer))
            .where(Decision.entity_type == "event", Decision.stage == "verify",
                   Decision.decision.in_(("media_vision_ok", "media_vision_block")))
        )
        ids = list(s.execute(
            select(Event.id)
            .join(EventItem, EventItem.event_id == Event.id)
            .join(Item, Item.id == EventItem.item_id)
            .join(MediaAsset, MediaAsset.item_id == Item.id)
            .where(Event.status.in_(("reported", "confirmed", "rumor")),
                   Event.id.in_(approved_event_ids_query()),
                   MediaAsset.kind == "image",
                   and_(MediaAsset.storage_key.is_not(None), MediaAsset.purged_at.is_(None)),
                   Event.id.not_in(checked))
            .order_by(Event.id).distinct().limit(max(limit * 4, limit))
        ).scalars().all())

        ids = [event_id for event_id in ids
               if event_media_readiness(s, event_id).downloads_settled][:limit]

    stats = {"events": 0, "blocked": 0}
    for event_id in ids:
        result = moderate_event_media(session_factory, moderator, event_id, store=store)
        if result.skipped:
            continue
        stats["events"] += 1
        stats["blocked"] += result.blocked
    return stats
