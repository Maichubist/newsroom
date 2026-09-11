from __future__ import annotations

import pytest

from newsroom.publishers.telegram import PublishResult, TelegramPublisher, split_text


class RecordingPoster:
    """Fake Bot API transport: records calls, returns queued responses."""

    def __init__(self, responses=None):
        self.calls: list[tuple[str, dict]] = []
        self._responses = list(responses or [])

    def __call__(self, method: str, payload: dict) -> dict:
        self.calls.append((method, payload))
        if self._responses:
            return self._responses.pop(0)
        return {"ok": True, "result": {"message_id": 100 + len(self.calls)}}


# --- split_text ----------------------------------------------------------------

def test_split_short_text_is_single_chunk():
    assert split_text("привіт") == ["привіт"]


def test_split_empty_is_no_chunks():
    assert split_text("   ") == []


def test_split_long_text_respects_limit():
    text = "\n\n".join(f"Абзац номер {i} з певним текстом." for i in range(200))
    chunks = split_text(text, limit=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # nothing lost beyond whitespace
    assert "".join(chunks).replace(" ", "").replace("\n", "") == text.replace(" ", "").replace("\n", "")


# --- gate ----------------------------------------------------------------------

def test_disabled_publisher_refuses_and_touches_no_network():
    poster = RecordingPoster()
    pub = TelegramPublisher("token", -100, enabled=False, poster=poster)
    result = pub.send_text("hello")
    assert result.ok is False and "disabled" in result.error
    assert poster.calls == []  # hard-off: transport never invoked


def test_enabled_but_missing_credentials_is_disabled():
    assert TelegramPublisher("", -100, enabled=True).is_enabled() is False
    assert TelegramPublisher("token", None, enabled=True).is_enabled() is False
    assert TelegramPublisher("token", -100, enabled=True).is_enabled() is True


# --- sending -------------------------------------------------------------------

def test_send_text_posts_and_returns_message_id():
    poster = RecordingPoster([{"ok": True, "result": {"message_id": 42}}])
    pub = TelegramPublisher("token", -100500, enabled=True, poster=poster)

    result = pub.send_text("Головна новина дня.")

    assert result == PublishResult(True, message_id=42)
    assert len(poster.calls) == 1
    method, payload = poster.calls[0]
    assert method == "sendMessage"
    assert payload["chat_id"] == -100500
    assert payload["parse_mode"] == "HTML"


def test_long_text_sent_as_multiple_messages_last_id_returned():
    poster = RecordingPoster()  # default: each call returns message_id = 100 + call number
    pub = TelegramPublisher("token", -100, enabled=True, poster=poster, max_len=100)
    text = "\n\n".join(f"Абзац {i} тут." for i in range(60))

    result = pub.send_text(text)

    assert result.ok is True
    assert len(poster.calls) >= 2                          # split into several messages
    assert result.message_id == 100 + len(poster.calls)   # id of the last message sent
    assert all(len(payload["text"]) <= 100 for _, payload in poster.calls)


def test_api_error_is_reported():
    poster = RecordingPoster([{"ok": False, "description": "chat not found"}])
    pub = TelegramPublisher("token", -100, enabled=True, poster=poster)
    result = pub.send_text("x")
    assert result.ok is False and "chat not found" in result.error


def test_from_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_CHANNEL_CHAT_ID", "-100777")
    monkeypatch.setenv("PUBLISH_ENABLED", "true")
    pub = TelegramPublisher.from_env(poster=RecordingPoster())
    assert pub.is_enabled() is True and pub.chat_id == -100777

    monkeypatch.setenv("PUBLISH_ENABLED", "false")
    assert TelegramPublisher.from_env().is_enabled() is False
