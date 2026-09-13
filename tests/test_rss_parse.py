from __future__ import annotations

from newsroom.collectors.rss import _is_denied_media, mentions_ukraine, parse_feed


def test_mentions_ukraine_matches_forms():
    assert mentions_ukraine("Zelensky meets allies", None) is True
    assert mentions_ukraine("Russland greift Charkiw an", None) is True     # German form
    assert mentions_ukraine(None, "Обстріл Києва вночі") is True            # Cyrillic
    assert mentions_ukraine("Drone strike near Pokrovsk", "") is True
    assert mentions_ukraine("US election results", "Trump wins Ohio") is False
    assert mentions_ukraine("Apple unveils new iPhone", None) is False

_PLACEHOLDER_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>t</title>
<item><title>A</title><link>https://ex/a</link><guid>a</guid>
  <enclosure url="https://cdn4.suspilne.media/images/default.jpg" type="image/jpeg"/></item>
<item><title>B</title><link>https://ex/b</link><guid>b</guid>
  <enclosure url="https://cdn/real-photo.jpg" type="image/jpeg"/></item>
</channel></rss>"""


def test_denied_media_url():
    assert _is_denied_media("https://cdn4.suspilne.media/images/default.jpg") is True
    assert _is_denied_media("https://cdn/real-photo.jpg") is False


def test_parse_feed_drops_placeholder_media():
    items = {i.external_id: i for i in parse_feed(1, _PLACEHOLDER_FEED)}
    assert items["a"].media == []                       # default.jpg dropped -> og:image resolver will fill
    assert [m.url for m in items["b"].media] == ["https://cdn/real-photo.jpg"]

FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
<channel>
  <title>Test feed</title>
  <language>uk</language>
  <item>
    <title>&#1047;&#1072;&#1075;&#1086;&#1083;&#1086;&#1074;&#1086;&#1082; &amp; &#1090;&#1077;&#1089;&#1090;</title>
    <link>https://ex/a</link>
    <guid>guid-a</guid>
    <description>&lt;p&gt;&#1058;&#1077;&#1082;&#1089;&#1090; &lt;b&gt;&#1085;&#1086;&#1074;&#1080;&#1085;&#1080;&lt;/b&gt;&lt;/p&gt;</description>
    <pubDate>Tue, 10 Sep 2026 12:00:00 +0000</pubDate>
    <enclosure url="https://ex/a.jpg" type="image/jpeg" length="1234"/>
  </item>
  <item>
    <title>Second</title>
    <link>https://ex/b</link>
    <media:content url="https://ex/b.mp4" type="video/mp4" width="640" height="360"/>
    <pubDate>Tue, 10 Sep 2026 13:00:00 +0000</pubDate>
  </item>
  <item>
    <title>No stable id</title>
    <description>skipped: no guid and no link</description>
  </item>
</channel>
</rss>
"""


def test_parse_maps_fields_and_skips_idless():
    items = parse_feed(source_id=7, raw_bytes=FEED)
    # third item (no guid, no link) is skipped — cannot be deduped
    assert len(items) == 2

    a, b = items
    assert a.source_id == 7
    assert a.external_id == "guid-a"          # guid preferred
    assert a.url == "https://ex/a"
    assert a.title == "Заголовок & тест"       # HTML entities unescaped
    assert a.text == "Текст новини"           # HTML tags stripped
    assert a.lang == "uk"
    assert a.published_at is not None and a.published_at.tzinfo is not None
    assert a.published_at.hour == 12

    # media: image enclosure with size
    assert len(a.media) == 1
    assert a.media[0].kind == "image" and a.media[0].url == "https://ex/a.jpg"
    assert a.media[0].size_bytes == 1234


def test_external_id_falls_back_to_link():
    items = parse_feed(source_id=1, raw_bytes=FEED)
    b = items[1]
    assert b.external_id == "https://ex/b"     # no guid -> link
    assert b.media[0].kind == "video"
    assert b.media[0].width == 640 and b.media[0].height == 360


def test_content_hash_and_simhash_available_on_parsed_items():
    a = parse_feed(source_id=1, raw_bytes=FEED)[0]
    assert len(a.content_hash) == 64
    assert isinstance(a.simhash, int)


def test_empty_feed_yields_nothing():
    assert parse_feed(source_id=1, raw_bytes=b"<rss><channel></channel></rss>") == []
