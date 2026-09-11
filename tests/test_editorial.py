from __future__ import annotations

from pathlib import Path

from newsroom.analyze.ai_accent import load_ai_accent
from newsroom.analyze.stoplist import load_stoplist
from newsroom.editorial import DraftContent, compose_post, critic_check, parse_draft

CONFIG = Path(__file__).resolve().parents[1] / "config"
STOP = load_stoplist(CONFIG / "stoplist.yaml")
ACCENT = load_ai_accent(CONFIG / "ai_accent.yaml")


# --- compose_post --------------------------------------------------------------

def test_compose_full_anatomy_and_hashtag_cap():
    post = compose_post(
        DraftContent(headline="Головне", lead="Суть події.", what_it_means="Наслідок для ринку."),
        status="confirmed",
        hashtags=["#економіка", "business", "extra"],  # third must be dropped
        sources=["Укрінформ"],
    )
    assert post.startswith("Головне")
    assert "Суть події." in post
    assert "Що це означає: Наслідок для ринку." in post
    assert "#економіка #business" in post and "extra" not in post
    assert post.rstrip().endswith("Джерела: Укрінформ")


def test_compose_rumor_label_first():
    post = compose_post(DraftContent(headline="Заголовок", lead="Текст."), is_rumor=True)
    assert post.splitlines()[0] == "Чутка"


def test_compose_reported_status_prefixes_lead():
    post = compose_post(DraftContent(headline="H", lead="сталася подія"), status="reported")
    assert "Повідомляють: сталася подія" in post


# --- parse_draft ---------------------------------------------------------------

def test_parse_draft_valid():
    d = parse_draft('{"headline": "H", "lead": "L", "what_it_means": "M", "rubrics": ["War"]}')
    assert d is not None and d.headline == "H" and d.what_it_means == "M" and d.rubrics == ["war"]


def test_parse_draft_requires_headline_and_lead():
    assert parse_draft('{"headline": "H"}') is None
    assert parse_draft("not json") is None
    assert parse_draft("[1,2]") is None


# --- critic --------------------------------------------------------------------

def test_clean_post_passes_critic():
    post = compose_post(DraftContent(headline="НБУ знизив ставку", lead="Ставка — 13%."),
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
