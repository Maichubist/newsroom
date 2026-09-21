from __future__ import annotations

from newsroom.publishers.supervision import Supervisor, format_publish_notice
from newsroom.publishers.telegram import TelegramPublisher


# --- format_publish_notice (offline) ------------------------------------------

def test_notice_marks_critical_and_rumor():
    txt = format_publish_notice(headline="Заголовок", risk_level="critical", is_rumor=True, channel_ref="42")
    assert "ризикова" in txt and "чутка" in txt and "42" in txt and "Заголовок" in txt


def test_notice_plain_when_not_risky():
    txt = format_publish_notice(headline="Спокійно", risk_level="low", is_rumor=False)
    assert "публікація" in txt and "ризикова" not in txt


def test_notice_marks_overseen_high_rubric():
    # a high-risk-but-supervised rubric (e.g. corruption): labelled "наглядова", not "ризикова"
    txt = format_publish_notice(headline="Корупційна схема", risk_level="high", is_rumor=False, oversight=True)
    assert "наглядова" in txt and "ризикова" not in txt


def test_critical_label_wins_over_oversight():
    txt = format_publish_notice(headline="h", risk_level="critical", is_rumor=False, oversight=True)
    assert "ризикова" in txt and "наглядова" not in txt


# --- Supervisor (offline, injected transport) ---------------------------------

class RecordingPoster:
    def __init__(self):
        self.calls = []

    def __call__(self, method, payload):
        self.calls.append((method, payload))
        return {"ok": True, "result": {"message_id": 1}}


def _sup(*, enabled=True, admin=999):
    poster = RecordingPoster()
    tg = TelegramPublisher("token", -100, enabled=enabled, poster=poster)
    return Supervisor(tg, admin), poster


def test_supervisor_disabled_when_publisher_off():
    sup, poster = _sup(enabled=False)
    assert sup.enabled() is False
    assert sup.notify("x") is False and poster.calls == []


def test_supervisor_disabled_without_admin_chat():
    sup, poster = _sup(admin=None)
    assert sup.enabled() is False and sup.notify("x") is False


def test_notify_published_only_for_risky_or_rumor():
    sup, poster = _sup()
    assert sup.notify_published(headline="h", risk_level="low", is_rumor=False) is False
    assert poster.calls == []                                  # plain post: no notice

    assert sup.notify_published(headline="h", risk_level="critical", is_rumor=False) is True
    assert sup.notify_published(headline="h", risk_level="low", is_rumor=True) is True
    assert len(poster.calls) == 2
    # notices go to the admin chat, not the channel
    assert all(payload["chat_id"] == 999 for _, payload in poster.calls)


def test_notify_published_fires_on_oversight_even_when_not_critical():
    # the confirmed change: a supervised high-risk rubric (corruption/politics/mobilization…)
    # notifies even though its risk_level is 'high', not 'critical'.
    sup, poster = _sup()
    assert sup.notify_published(headline="h", risk_level="high", is_rumor=False, oversight=False) is False
    assert poster.calls == []                                  # high + not overseen: no notice
    assert sup.notify_published(headline="h", risk_level="high", is_rumor=False, oversight=True) is True
    assert len(poster.calls) == 1


def test_critical_still_notifies_as_safety_backstop():
    # even if a rubric's oversight flag were off, a critical post must still notify.
    sup, poster = _sup()
    assert sup.notify_published(headline="h", risk_level="critical", is_rumor=False, oversight=False) is True
    assert len(poster.calls) == 1
