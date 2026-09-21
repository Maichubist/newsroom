from __future__ import annotations

from newsroom.publishers.cascade import MediaItem, MediaLimits, choose_media, choose_media_group
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


def test_stored_file_is_sendable_without_url():
    # Telegram media has no URL but a stored file -> still eligible (uploaded, not fetched)
    c = choose_media([MediaItem("image", url=None, width=800, storage_key="ab/abcd")])
    assert c is not None and c.param == "photo" and c.url is None and c.storage_key == "ab/abcd"
    # nothing sendable at all -> None
    assert choose_media([MediaItem("image", url=None, width=800, storage_key=None)]) is None


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


# --- choose_media_group (albums, offline) -------------------------------------

def test_group_collects_multiple_images_in_order():
    items = [MediaItem("image", "http://x/1", width=800), MediaItem("image", "http://x/2", width=800),
             MediaItem("image", "http://x/3", width=800)]
    g = choose_media_group(items)
    assert [c.url for c in g] == ["http://x/1", "http://x/2", "http://x/3"]
    assert all(c.param == "photo" for c in g)


def test_group_mixes_image_and_video():
    items = [MediaItem("image", "http://x/i", width=800), MediaItem("video", "http://x/v")]
    g = choose_media_group(items)
    assert {c.param for c in g} == {"photo", "video"} and len(g) == 2


def test_group_of_one_when_only_one_eligible():
    # a lone image -> list of 1 (caller sends it as a single, not a group)
    assert len(choose_media_group([MediaItem("image", "http://x/i", width=800)])) == 1
    assert choose_media_group([]) == []


def test_group_dedups_and_skips_tiny_and_unsendable():
    items = [
        MediaItem("image", "http://x/dup", width=800),
        MediaItem("image", "http://x/dup", width=800),      # duplicate url -> once
        MediaItem("image", "http://x/logo", width=100),     # too narrow -> skip
        MediaItem("image", url=None, storage_key=None),     # unsendable -> skip
        MediaItem("embed", "http://x/e"),                    # embeds never sent
    ]
    g = choose_media_group(items)
    assert [c.url for c in g] == ["http://x/dup"]


def test_group_respects_limits_and_cap():
    limits = MediaLimits(photo_max_bytes=1000, video_max_bytes=1000)
    items = [MediaItem("image", "http://x/big", width=800, size_bytes=5000),   # too big -> skip
             MediaItem("image", "http://x/ok", width=800, size_bytes=500)]
    assert [c.url for c in choose_media_group(items, limits)] == ["http://x/ok"]
    many = [MediaItem("image", f"http://x/{i}", width=800) for i in range(15)]
    assert len(choose_media_group(many, max_group=10)) == 10


def test_excludes_narrow_image_but_keeps_unknown_width():
    # a known-tiny image (logo/icon) is not attached...
    assert choose_media([MediaItem("image", "http://x/logo.png", width=120)]) is None
    # ...but an image without a reported width is still allowed (Telegram fetches it)
    c = choose_media([MediaItem("image", "http://x/photo.jpg", width=None)])
    assert c is not None and c.url == "http://x/photo.jpg"


def test_narrow_image_min_width_is_configurable():
    items = [MediaItem("image", "http://x/i.jpg", width=300)]
    assert choose_media(items) is None                                   # default min 400 excludes 300
    assert choose_media(items, MediaLimits(min_image_width=200)) is not None


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


def test_send_media_uploads_file_when_no_url():
    # Telegram-origin media: no URL, uploaded as a multipart file (carried in _file)
    poster = RecordingPoster()
    choice = choose_media([MediaItem("image", url=None, width=800, storage_key="ab/cd")])
    _pub(poster).send_post("Підпис.", choice, file=("photo.jpg", b"\xff\xd8\xff bytes"))
    method, payload = poster.calls[0]
    assert method == "sendPhoto"
    assert "photo" not in payload                         # not a URL send
    assert payload["_file"]["field"] == "photo" and payload["_file"]["data"] == b"\xff\xd8\xff bytes"
    assert payload["caption"] == "Підпис."


def test_send_media_errors_when_neither_url_nor_file():
    poster = RecordingPoster()
    choice = choose_media([MediaItem("image", url=None, width=800, storage_key="ab/cd")])
    result = _pub(poster).send_media(choice, "cap")       # no file passed
    assert result.ok is False and poster.calls == []
