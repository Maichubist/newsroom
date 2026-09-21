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
                          is_rumor: bool, oversight: bool = False,
                          channel_ref: str | None = None) -> str:
    labels: list[str] = []
    if risk_level == "critical":
        labels.append("⚠️ ризикова")
    elif oversight:
        # a high-risk-but-supervised rubric (corruption, politics, mobilization…):
        # not "critical", but flagged for human eyes (info-attack / speculative vector).
        labels.append("наглядова")
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

    def notify(self, text: str, *, reply_markup: dict | None = None) -> bool:
        if not self.enabled():
            return False
        result = self.telegram.send_text(text, chat_id=self.admin_chat_id, disable_preview=True,
                                         reply_markup=reply_markup)
        if not result.ok:
            log.warning("supervisor notice failed", extra={"error": result.error})
        return result.ok

    def notify_published(self, *, headline: str | None, risk_level: str | None,
                         is_rumor: bool, oversight: bool = False,
                         channel_ref: str | None = None,
                         publication_id: int | None = None) -> bool:
        """Notify for the posts that need human eyes: a rubric flagged for oversight
        (the confirmed set — war/defense/security/mobilization/politics/geopolitics/
        corruption; corruption is a speculative info-attack vector), a critical-topic
        post (kept as a safety backstop even if a rubric's oversight flag is off), or a
        rumor. The notice carries Recall/Stop buttons the supervision bot acts on (§10)."""
        if not (oversight or risk_level == "critical" or is_rumor):
            return False
        text = format_publish_notice(headline=headline, risk_level=risk_level,
                                     is_rumor=is_rumor, oversight=oversight,
                                     channel_ref=channel_ref)
        return self.notify(text, reply_markup=supervision_keyboard(publication_id))

    def notify_review(self, *, headline: str | None, publication_id: int | None,
                      reason: str | None = None) -> bool:
        """A publish-time dedup check couldn't decide (LLM unavailable) and held the
        draft as a possible duplicate. Ask a human: publish anyway, or drop it."""
        text = format_review_notice(headline=headline, reason=reason)
        return self.notify(text, reply_markup=review_keyboard(publication_id))


def supervision_keyboard(publication_id: int | None) -> dict:
    """Inline keyboard for a publication notice: recall this post, or halt all
    publishing. callback_data is what the supervision bot dispatches on."""
    buttons: list[dict] = []
    if publication_id is not None:
        buttons.append({"text": "↩︎ Відкликати", "callback_data": f"retract:{publication_id}"})
    buttons.append({"text": "⏸ Стоп", "callback_data": "stop"})
    return {"inline_keyboard": [buttons]}


def format_review_notice(*, headline: str | None, reason: str | None = None) -> str:
    head = (headline or "").strip() or "(без заголовка)"
    why = f"\nПричина: {reason}" if reason else ""
    return (f"🕵 Притримано як можливий дубль (потрібне рішення){why}\n{head}\n\n"
            f"«Опублікувати» — надіслати попри це; «Відхилити» — прибрати чернетку.")


def review_keyboard(publication_id: int | None) -> dict:
    """Inline keyboard for a held-for-review draft: publish anyway, or drop it."""
    if publication_id is None:
        return {"inline_keyboard": [[{"text": "⏸ Стоп", "callback_data": "stop"}]]}
    return {"inline_keyboard": [[
        {"text": "✅ Опублікувати", "callback_data": f"review_publish:{publication_id}"},
        {"text": "🗑 Відхилити", "callback_data": f"review_drop:{publication_id}"},
    ]]}
