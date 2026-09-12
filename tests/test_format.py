from __future__ import annotations

from newsroom.publishers.format import render_telegram_html


def test_headline_is_bold():
    out = render_telegram_html(headline="НБУ знизив ставку", body="Ставку знижено до 13%.")
    assert "<b>НБУ знизив ставку</b>" in out


def test_sources_are_links_without_label():
    out = render_telegram_html(
        headline="Заголовок", body="Текст.",
        source_links=[("Цензор.НЕТ", "https://censor.net/a1")],
    )
    assert '<a href="https://censor.net/a1">Цензор.НЕТ</a>' in out
    assert "Джерела" not in out and "Джерело" not in out


def test_multiple_sources_joined_and_urlless_stays_plain():
    out = render_telegram_html(
        headline="H", body="Текст.",
        source_links=[("BBC", "https://bbc.com/x"), ("КМДА", None)],
    )
    assert '<a href="https://bbc.com/x">BBC</a>' in out
    assert "КМДА" in out and "<a" in out.split("КМДА")[0]   # KMDA rendered plain, after the linked BBC
    assert " · " in out                                     # names joined with a separator


def test_escapes_html_special_chars_in_text_and_href():
    out = render_telegram_html(
        headline="AT&T <під загрозою>", body='Ціна впала на "5%" & більше.',
        source_links=[("R&D", "https://x/a?b=1&c=2")],
    )
    # visible text is escaped so the parser never sees stray markup
    assert "&amp;" in out and "&lt;під загрозою&gt;" in out and "&quot;5%&quot;" in out
    # the href keeps a working (attribute-escaped) URL
    assert 'href="https://x/a?b=1&amp;c=2"' in out
    assert "<b>AT&amp;T &lt;під загрозою&gt;</b>" in out


def test_rumor_label_and_reported_prefix():
    out = render_telegram_html(headline="H", body="щось сталося", is_rumor=True, reported=True)
    assert out.startswith("Чутка")
    assert "Повідомляють: щось сталося" in out


def test_hashtags_capped_and_present():
    out = render_telegram_html(headline="H", body="Текст.", hashtags=["#політика", "#сюжет", "#зайвий"])
    assert "#політика #сюжет" in out and "#зайвий" not in out
