from __future__ import annotations

from newsroom.analyze.embeddings import MAX_EMBED_CHARS, clean_for_embedding, clip_for_embedding


def test_clip_caps_length():
    long = "х" * (MAX_EMBED_CHARS + 5000)
    clipped = clip_for_embedding(long)
    assert len(clipped) == MAX_EMBED_CHARS


def test_clip_char_count_bounds_token_count():
    # a BPE token is >= 1 char, so char count is an upper bound on tokens;
    # capping at 8000 chars guarantees <= 8192 tokens (never a 400).
    assert MAX_EMBED_CHARS <= 8192


def test_clip_short_text_unchanged():
    assert clip_for_embedding("  привіт  ") == "привіт"


def test_clip_empty_is_non_empty():
    assert clip_for_embedding("") == " "
    assert clip_for_embedding(None) == " "
    assert clip_for_embedding("   ") == " "


# --- clean_for_embedding (offline) --------------------------------------------

def test_clean_strips_footer_emoji_url_keeps_content():
    text = ("🇪🇺 ЄС виділив Україні 3,3 млрд євро на оборону\n\n"
            "📰 підписатися\n"
            "Наш канал | Instagram | YouTube\n"
            "👉 https://t.me/x/1")
    cleaned = clean_for_embedding(text)
    assert "ЄС виділив Україні 3,3 млрд євро на оборону" in cleaned
    assert "підписатися" not in cleaned and "Instagram" not in cleaned
    assert "https" not in cleaned and "🇪🇺" not in cleaned


def test_clean_keeps_content_words_that_contain_footer_substrings():
    # "підтримки"/"підписав" are content, not footers -> never dropped
    t = "Уряд ухвалив програму підтримки бізнесу, яку підписав президент минулого тижня."
    cleaned = clean_for_embedding(t)
    assert "програму підтримки" in cleaned and "підписав" in cleaned


def test_clean_all_footer_falls_back_to_raw_non_empty():
    assert clean_for_embedding("📰 підписатися").strip() != ""   # never empties a post
