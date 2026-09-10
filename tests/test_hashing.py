from __future__ import annotations

from newsroom.collectors.base import (
    compute_content_hash,
    compute_simhash,
    hamming_distance,
    normalize_text,
)


def test_normalize_collapses_and_lowercases():
    assert normalize_text("  Привіт   Світ \n") == "привіт світ"
    assert normalize_text(None) == ""


def test_content_hash_is_stable_and_whitespace_insensitive():
    a = compute_content_hash("Заголовок", "Текст новини")
    b = compute_content_hash("  заголовок ", "текст   новини")
    assert a == b  # normalization makes exact-dup detection robust
    assert len(a) == 64


def test_content_hash_differs_on_real_change():
    a = compute_content_hash("Заголовок", "Текст новини")
    c = compute_content_hash("Заголовок", "Текст новини з новою цифрою 5")
    assert a != c


def test_simhash_near_duplicate_is_closer_than_unrelated():
    original = "Уряд ухвалив новий державний бюджет на 2027 рік із рекордними видатками на оборону"
    rewrite = "Кабмін ухвалив новий держбюджет на 2027 рік з рекордними оборонними видатками"
    unrelated = "Барселона перемогла Реал у класико з рахунком три-один на Камп Ноу"

    h0 = compute_simhash("", original)
    h1 = compute_simhash("", rewrite)
    h2 = compute_simhash("", unrelated)

    assert hamming_distance(h0, h0) == 0
    assert hamming_distance(h0, h1) < hamming_distance(h0, h2)


def test_simhash_empty_is_zero():
    assert compute_simhash(None, None) == 0


def test_simhash_always_fits_signed_bigint():
    # Regression: an unsigned 64-bit simhash overflows Postgres BIGINT. Values
    # must stay within the signed range, and the high-bit case must be exercised.
    lo, hi = -(2 ** 63), 2 ** 63 - 1
    saw_negative = False
    for i in range(50):
        h = compute_simhash(f"Заголовок {i}", f"Текст новини номер {i} про подію {i * 7}")
        assert lo <= h <= hi
        saw_negative = saw_negative or h < 0
    assert saw_negative


def test_hamming_handles_signed_values():
    assert hamming_distance(-1, 0) == 64   # all 64 bits differ
    assert hamming_distance(-1, -1) == 0
