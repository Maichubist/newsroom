from __future__ import annotations

from newsroom.analyze.embeddings import MAX_EMBED_CHARS, clip_for_embedding


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
