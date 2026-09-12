from __future__ import annotations

from pathlib import Path

from newsroom.analyze.ai_accent import load_ai_accent
from newsroom.analyze.stoplist import load_stoplist
from newsroom.editorial import (
    DraftContent,
    compose_post,
    content_is_publishable,
    critic_check,
    parse_draft,
)

CONFIG = Path(__file__).resolve().parents[1] / "config"
STOP = load_stoplist(CONFIG / "stoplist.yaml")
ACCENT = load_ai_accent(CONFIG / "ai_accent.yaml")


# --- compose_post --------------------------------------------------------------

def test_compose_prose_anatomy_and_hashtag_cap():
    post = compose_post(
        DraftContent(headline="Головне", body="Суть події. Наслідок для ринку конкретний.",
                     watching="Рішення очікують у четвер."),
        hashtags=["#економіка", "business", "extra"],  # third must be dropped
        sources=["Укрінформ"],
    )
    assert post.startswith("Головне")
    assert "Суть події. Наслідок для ринку конкретний." in post
    assert "Рішення очікують у четвер." in post
    assert "Що це означає:" not in post and "За чим стежити:" not in post   # no labels — prose only
    assert "#економіка #business" in post and "extra" not in post
    assert post.rstrip().endswith("Джерела: Укрінформ")


def test_compose_rumor_label_first():
    post = compose_post(DraftContent(headline="Заголовок", body="Текст події."), is_rumor=True)
    assert post.splitlines()[0] == "Чутка"


def test_compose_reported_prefixes_body_when_flagged():
    post = compose_post(DraftContent(headline="H", body="сталася подія"), reported=True)
    assert "Повідомляють: сталася подія" in post


def test_compose_no_reported_prefix_by_default():
    post = compose_post(DraftContent(headline="H", body="сталася подія"))
    assert "Повідомляють:" not in post


# --- parse_draft ---------------------------------------------------------------

def test_parse_draft_valid():
    d = parse_draft('{"headline": "H", "body": "Тіло новини тут.", "rubrics": ["War"]}')
    assert d is not None and d.headline == "H" and d.body == "Тіло новини тут." and d.rubrics == ["war"]


def test_parse_draft_accepts_legacy_lead_key():
    d = parse_draft('{"headline": "H", "lead": "Старий ключ теж парситься."}')
    assert d is not None and d.body == "Старий ключ теж парситься."


def test_parse_draft_requires_headline_and_body():
    assert parse_draft('{"headline": "H"}') is None
    assert parse_draft("not json") is None
    assert parse_draft("[1,2]") is None


def test_parse_draft_rejects_body_that_repeats_headline():
    # a body == headline is a degenerate generation (title twice); reject so the
    # caller retries instead of shipping it
    assert parse_draft('{"headline": "Стефанчук став послом", "body": "стефанчук став послом"}') is None


# --- content_is_publishable ----------------------------------------------------

def test_real_content_is_publishable():
    assert content_is_publishable(DraftContent(headline="НБУ знизив ставку", body="Ставку знижено до 13%.")) is True


def test_fallback_draft_is_not_publishable():
    # the generation fallback (headline=title, body=title/summary, fallback=True)
    assert content_is_publishable(DraftContent(headline="Новина", body="Новина", fallback=True)) is False


def test_thin_or_duplicate_body_is_not_publishable():
    assert content_is_publishable(DraftContent(headline="Подія", body="")) is False          # empty
    assert content_is_publishable(DraftContent(headline="Подія", body="коротко")) is False    # too short
    assert content_is_publishable(DraftContent(headline="Тема дня", body="Тема дня")) is False  # repeats headline


# --- critic --------------------------------------------------------------------

def test_clean_post_passes_critic():
    post = compose_post(DraftContent(headline="НБУ знизив ставку", body="Ставку знижено до 13%."),
                        sources=["НБУ"])
    r = critic_check(post, stoplist_rules=STOP, ai_accent_patterns=ACCENT)
    assert r.ok is True and r.hard == [] and r.soft == []


def test_stoplist_block_is_hard_fail():
    r = critic_check("Шахеди курсом на Київ.", stoplist_rules=STOP, ai_accent_patterns=ACCENT)
    assert r.ok is False and any(h.startswith("stoplist_block") for h in r.hard)


def test_missing_rumor_label_is_hard_fail():
    bad = critic_check("Просто текст без позначки.", is_rumor=True, stoplist_rules=STOP, ai_accent_patterns=ACCENT)
    assert bad.ok is False and "missing_rumor_label" in bad.hard
    good = critic_check("Чутка. Просто текст.", is_rumor=True, stoplist_rules=STOP, ai_accent_patterns=ACCENT)
    assert good.ok is True


def test_ai_accent_is_soft():
    r = critic_check("Таким чином, ситуація змінилася.", stoplist_rules=STOP, ai_accent_patterns=ACCENT)
    assert r.ok is True and any(sft.startswith("ai_accent") for sft in r.soft)
