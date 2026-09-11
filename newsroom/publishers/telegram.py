"""Telegram Bot API publisher (ported from news_bot, adapted and gated).

Stage 1a wires it ready but nothing calls it: the channel adapter that turns an
event into a publication is stage 1г. Two safety properties matter here and are
tested:
  * PUBLISH_ENABLED master switch — when off, send_* refuse and touch no network
    (the hard-off "stop button", architecture §10).
  * long posts are split under Telegram's 4096-char limit.

Media send (photo/video) and the video→image→text cascade are deferred to 1г
(architecture §13); this port covers text, which is enough to wire and test the
gate and the transport seam.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("newsroom.publishers.telegram")

TELEGRAM_MAX_LEN = 4096
TELEGRAM_CAPTION_LEN = 1024   # media caption limit
# poster(method, payload) -> Telegram Bot API response dict ({"ok": bool, ...}).
Poster = Callable[[str, dict], dict]


def split_text(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    """Split text into chunks under `limit`, preferring paragraph, then line,
    then word boundaries; hard-cut only if a single run exceeds the limit."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c]


@dataclass(frozen=True)
class PublishResult:
    ok: bool
    message_id: int | None = None
    error: str | None = None


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


class TelegramPublisher:
    def __init__(
        self,
        token: str,
        channel_chat_id: int | None,
        *,
        enabled: bool = False,
        shadow: bool = False,
        shadow_chat_id: int | None = None,
        poster: Poster | None = None,
        max_len: int = TELEGRAM_MAX_LEN,
    ):
        self.token = token or ""
        self.chat_id = channel_chat_id
        self.enabled = bool(enabled)
        self.shadow = bool(shadow)
        self.shadow_chat_id = shadow_chat_id
        self.max_len = max_len
        self._poster = poster or self._httpx_poster

    @classmethod
    def from_env(cls, *, poster: Poster | None = None) -> "TelegramPublisher":
        return cls(
            token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            channel_chat_id=_int_or_none(os.getenv("TELEGRAM_CHANNEL_CHAT_ID")),
            enabled=os.getenv("PUBLISH_ENABLED", "false").strip().lower() in {"1", "true", "yes"},
            shadow=os.getenv("SHADOW_MODE", "false").strip().lower() in {"1", "true", "yes"},
            shadow_chat_id=_int_or_none(os.getenv("TELEGRAM_SHADOW_CHANNEL_CHAT_ID")),
            poster=poster,
        )

    @property
    def active_chat_id(self) -> int | None:
        """Where channel posts go: the closed test channel in shadow mode
        (architecture §2), otherwise the real channel."""
        return self.shadow_chat_id if self.shadow else self.chat_id

    def is_enabled(self) -> bool:
        """Master switch: publishing on AND the active target's credentials present."""
        return self.enabled and bool(self.token) and self.active_chat_id is not None

    def send_text(self, text: str, *, chat_id: int | None = None, disable_preview: bool = True,
                  reply_markup: dict | None = None, reply_to_message_id: int | None = None) -> PublishResult:
        if not self.is_enabled():
            # Hard-off: no network touched. This is the stop button.
            return PublishResult(False, error="publishing disabled (PUBLISH_ENABLED off or no credentials)")
        target = chat_id if chat_id is not None else self.active_chat_id
        chunks = split_text(text, self.max_len)
        if not chunks:
            return PublishResult(False, error="empty text")

        last_id: int | None = None
        for i, chunk in enumerate(chunks):
            payload = {
                "chat_id": target,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": disable_preview,
            }
            if reply_markup is not None and i == len(chunks) - 1:
                payload["reply_markup"] = reply_markup   # buttons only on the final chunk
            if reply_to_message_id is not None and i == 0:
                # chain only the first chunk to the parent; a missing parent must
                # not drop the post (story reply-chaining, §7)
                payload["reply_to_message_id"] = reply_to_message_id
                payload["allow_sending_without_reply"] = True
            resp = self._poster("sendMessage", payload)
            if not resp.get("ok"):
                err = resp.get("description") or resp.get("error") or str(resp)
                log.warning("telegram publish failed", extra={"error": err})
                return PublishResult(False, message_id=last_id, error=str(err))
            last_id = (resp.get("result") or {}).get("message_id", last_id)
        return PublishResult(True, message_id=last_id)

    def send_media(self, choice, caption: str, *, chat_id: int | None = None,
                   reply_to_message_id: int | None = None) -> PublishResult:
        """Send one photo/video by URL with a caption (architecture §13 cascade).
        The caller guarantees the caption fits TELEGRAM_CAPTION_LEN."""
        if not self.is_enabled():
            return PublishResult(False, error="publishing disabled (PUBLISH_ENABLED off or no credentials)")
        target = chat_id if chat_id is not None else self.active_chat_id
        payload = {
            "chat_id": target,
            choice.param: choice.url,
            "caption": caption,
            "parse_mode": "HTML",
        }
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
            payload["allow_sending_without_reply"] = True
        resp = self._poster(choice.method, payload)
        if not resp.get("ok"):
            err = resp.get("description") or resp.get("error") or str(resp)
            log.warning("telegram media publish failed", extra={"error": err})
            return PublishResult(False, error=str(err))
        return PublishResult(True, message_id=(resp.get("result") or {}).get("message_id"))

    def send_post(self, body: str, media_choice=None, *, chat_id: int | None = None,
                  reply_to_message_id: int | None = None) -> PublishResult:
        """Publish a post: media + caption when a media choice is given and the
        body fits a caption, otherwise text (the cascade falls back to text so a
        long post never loses its content to caption truncation)."""
        body = (body or "").strip()
        if media_choice is not None and 0 < len(body) <= TELEGRAM_CAPTION_LEN:
            return self.send_media(media_choice, body, chat_id=chat_id,
                                   reply_to_message_id=reply_to_message_id)
        return self.send_text(body, chat_id=chat_id, reply_to_message_id=reply_to_message_id)

    def delete_message(self, chat_id: int | None, message_id: int) -> bool:
        """Delete a channel message (used to retract a post). No enable gate: a
        retract must work even while publishing is halted."""
        if not (self.token and chat_id is not None):
            return False
        resp = self._poster("deleteMessage", {"chat_id": chat_id, "message_id": message_id})
        return bool(resp.get("ok"))

    def answer_callback_query(self, callback_query_id: str, text: str | None = None) -> bool:
        payload = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        return bool(self._poster("answerCallbackQuery", payload).get("ok"))

    def get_updates(self, *, offset: int | None = None, timeout: int = 25,
                    allowed_updates: list[str] | None = None) -> list[dict]:  # pragma: no cover - network
        payload: dict = {"timeout": timeout}
        if offset is not None:
            payload["offset"] = offset
        if allowed_updates is not None:
            payload["allowed_updates"] = allowed_updates
        resp = self._poster("getUpdates", payload)
        return resp.get("result") or [] if resp.get("ok") else []

    def _httpx_poster(self, method: str, payload: dict) -> dict:  # pragma: no cover - network
        import httpx

        url = f"https://api.telegram.org/bot{self.token}/{method}"
        try:
            r = httpx.post(url, json=payload, timeout=20.0)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"request failed: {exc}"}
        try:
            return r.json()
        except Exception:
            return {"ok": False, "error": f"HTTP {r.status_code}"}
