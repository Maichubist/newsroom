from __future__ import annotations

from newsroom.publishers.cascade import MediaItem, MediaLimits, choose_media
from newsroom.publishers.telegram import TelegramPublisher


# --- choose_media (offline) ---------------------------------------------------

def test_prefers_video_over_image():
    items = [MediaItem("image", "http://x/i.jpg", width=800), MediaItem("video", "http://x/v.mp4")]
    choice = choose_media(items)
    assert choice.method == "sendVideo" and choice.param == "video" and choice.url == "http://x/v.mp4"


def test_picks_widest_image_when_no_video():
    items = [MediaItem("image", "http://x/small.jpg", width=320),
             MediaItem("image", "http://x/big.jpg", width=1600)]
    choice = choose_media(items)
    assert choice.method == "sendPhoto" and choice.url == "http://x/big.jpg"


def test_ignores_embeds_and_urlless():
    assert choose_media([MediaItem("embed", "http://x/e"), MediaItem("image", None, width=999)]) is None


def test_respects_size_limits():
    limits = MediaLimits(photo_max_bytes=1000, video_max_bytes=1000)
    # video too big, image too big -> nothing
    assert choose_media([MediaItem("video", "http://x/v", size_bytes=5000)], limits) is None
    # image within limit chosen
    c = choose_media([MediaItem("image", "http://x/i", width=800, size_bytes=500)], limits)
    assert c is not None and c.param == "photo"


def test_unknown_size_is_allowed():
    c = choose_media([MediaItem("image", "http://x/i", width=800, size_bytes=None)])
    assert c is not None


# --- send_post routing (offline) ----------------------------------------------

class RecordingPoster:
    def __init__(self):
        self.calls = []

    def __call__(self, method, payload):
        self.calls.append((method, payload))
        return {"ok": True, "result": {"message_id": 1}}


def _pub(poster):
    return TelegramPublisher("token", -100, enabled=True, poster=poster)


def test_send_post_text_only_when_no_media():
    poster = RecordingPoster()
    _pub(poster).send_post("Коротка новина.")
    assert poster.calls[0][0] == "sendMessage"


def test_send_post_media_with_caption_when_body_fits():
    poster = RecordingPoster()
    choice = choose_media([MediaItem("image", "http://x/i.jpg", width=800)])
    _pub(poster).send_post("Коротко.", choice)
    method, payload = poster.calls[0]
    assert method == "sendPhoto" and payload["photo"] == "http://x/i.jpg" and payload["caption"] == "Коротко."


def test_send_post_falls_back_to_text_when_body_too_long_for_caption():
    poster = RecordingPoster()
    choice = choose_media([MediaItem("image", "http://x/i.jpg", width=800)])
    long_body = "х" * 1500          # exceeds the 1024 caption limit
    _pub(poster).send_post(long_body, choice)
    assert all(m == "sendMessage" for m, _ in poster.calls)   # media dropped, text preserved
