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
        poster: Poster | None = None,
        max_len: int = TELEGRAM_MAX_LEN,
    ):
        self.token = token or ""
        self.chat_id = channel_chat_id
        self.enabled = bool(enabled)
        self.max_len = max_len
        self._poster = poster or self._httpx_poster

    @classmethod
    def from_env(cls, *, poster: Poster | None = None) -> "TelegramPublisher":
        return cls(
            token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            channel_chat_id=_int_or_none(os.getenv("TELEGRAM_CHANNEL_CHAT_ID")),
            enabled=os.getenv("PUBLISH_ENABLED", "false").strip().lower() in {"1", "true", "yes"},
            poster=poster,
        )

    def is_enabled(self) -> bool:
        """Master switch: publishing on AND credentials present."""
        return self.enabled and bool(self.token) and self.chat_id is not None

    def send_text(self, text: str, *, chat_id: int | None = None, disable_preview: bool = True) -> PublishResult:
        if not self.is_enabled():
            # Hard-off: no network touched. This is the stop button.
            return PublishResult(False, error="publishing disabled (PUBLISH_ENABLED off or no credentials)")
        target = chat_id if chat_id is not None else self.chat_id
        chunks = split_text(text, self.max_len)
        if not chunks:
            return PublishResult(False, error="empty text")

        last_id: int | None = None
        for chunk in chunks:
            resp = self._poster("sendMessage", {
                "chat_id": target,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": disable_preview,
            })
            if not resp.get("ok"):
                err = resp.get("description") or resp.get("error") or str(resp)
                log.warning("telegram publish failed", extra={"error": err})
                return PublishResult(False, message_id=last_id, error=str(err))
            last_id = (resp.get("result") or {}).get("message_id", last_id)
        return PublishResult(True, message_id=last_id)

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
