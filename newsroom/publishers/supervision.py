"""Supervision notices to the service chat (architecture §10).

A human does not approve every post, but must be told about the risky ones. When
a critical-topic or rumor post goes out, the Supervisor sends a short notice to
the service chat (TELEGRAM_ADMIN_CHAT_ID) so a person can react — recall via the
stop button (system_state) or fix the post. It reuses the Telegram adapter, so it
is off whenever publishing is off. Interactive recall/fix buttons need a bot
callback loop and are left for a later step; this is the one-way alert.

The message text is pure and offline-tested; sending goes through the adapter.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("newsroom.publishers.supervision")


def format_publish_notice(*, headline: str | None, risk_level: str | None,
                          is_rumor: bool, channel_ref: str | None = None) -> str:
    labels: list[str] = []
    if risk_level == "critical":
        labels.append("⚠️ ризикова")
    if is_rumor:
        labels.append("чутка")
    label = ", ".join(labels) or "публікація"
    ref = f" · msg {channel_ref}" if channel_ref else ""
    head = (headline or "").strip() or "(без заголовка)"
    return (f"🛰 Опубліковано ({label}){ref}\n{head}\n\n"
            f"Відкликати — стоп-кнопка (system_state {'publish_stopped'}).")


class Supervisor:
    def __init__(self, telegram, admin_chat_id: int | None):
        self.telegram = telegram
        self.admin_chat_id = admin_chat_id

    @classmethod
    def from_env(cls, telegram) -> "Supervisor":
        raw = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
        try:
            admin = int(raw) if raw not in ("", None) else None
        except (TypeError, ValueError):
            admin = None
        return cls(telegram, admin)

    def enabled(self) -> bool:
        return bool(self.telegram.is_enabled() and self.admin_chat_id is not None)

    def notify(self, text: str) -> bool:
        if not self.enabled():
            return False
        result = self.telegram.send_text(text, chat_id=self.admin_chat_id, disable_preview=True)
        if not result.ok:
            log.warning("supervisor notice failed", extra={"error": result.error})
        return result.ok

    def notify_published(self, *, headline: str | None, risk_level: str | None,
                         is_rumor: bool, channel_ref: str | None = None) -> bool:
        """Notify only for the posts that matter: critical-topic or rumor."""
        if not (risk_level == "critical" or is_rumor):
            return False
        return self.notify(format_publish_notice(
            headline=headline, risk_level=risk_level, is_rumor=is_rumor, channel_ref=channel_ref))
