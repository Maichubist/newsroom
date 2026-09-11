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
            resp = self._ensure_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return resp.choices[0].message.content
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
                         charter_version: str = "0.2") -> ModerationResult:
    """Moderate the event's images; journal media_vision_ok / media_vision_block.
    Idempotent, and a no-op (no decision) for events without images."""
    from sqlalchemy import func, select

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
            select(MediaAsset.id, MediaAsset.url)
            .join(Item, Item.id == MediaAsset.item_id)
            .join(EventItem, EventItem.item_id == Item.id)
            .where(EventItem.event_id == event_id, MediaAsset.kind == "image",
                   MediaAsset.url.is_not(None))
        ).all())

    if not images:
        return ModerationResult(event_id, checked=0, skipped=True)

    flags: list[dict] = []
    for media_id, url in images:
        verdict = moderator.check(url)
        if verdict.blocked:
            flags.append({"media_id": media_id, "labels": verdict.labels, "reason": verdict.reason})

    with session_factory() as s:
        s.add(Decision(
            entity_type="event", entity_id=str(event_id), stage="verify",
            decision="media_vision_block" if flags else "media_vision_ok",
            reason=f"{len(flags)} blocked of {len(images)}",
            details={"checked": len(images), "flags": flags},
            charter_version=charter_version,
        ))
        s.commit()
    return ModerationResult(event_id, checked=len(images), blocked=len(flags))


def moderate_pending(session_factory, moderator: ImageModerator, *, limit: int = 25) -> dict[str, int]:
    """One moderation tick: moderate publishable events that own images and have
    no vision verdict yet."""
    from sqlalchemy import Integer, cast, select

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
                   MediaAsset.kind == "image", MediaAsset.url.is_not(None),
                   Event.id.not_in(checked))
            .order_by(Event.id).distinct().limit(limit)
        ).scalars().all())

    stats = {"events": 0, "blocked": 0}
    for event_id in ids:
        result = moderate_event_media(session_factory, moderator, event_id)
        if result.skipped:
            continue
        stats["events"] += 1
        stats["blocked"] += result.blocked
    return stats
