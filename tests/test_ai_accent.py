from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from newsroom.analyze.ai_accent import AiAccentConfigError, check_ai_accent, load_ai_accent

PATTERNS = load_ai_accent(Path(__file__).resolve().parents[1] / "config" / "ai_accent.yaml")


def _cats(text):
    return {h.category for h in check_ai_accent(text, PATTERNS)}


# --- positives -----------------------------------------------------------------

def test_cliches_detected():
    assert "cliches" in _cats("Таким чином, варто зазначити, що це новий етап.")


def test_calques_detected():
    for bad in ["на протязі тижня", "приймати участь у заході", "слідуючий крок",
                "дані співпадають", "провели міроприємство"]:
        assert "calques" in _cats(bad), bad


def test_template_and_weak_ending_detected():
    assert "templates" in _cats("Це не лише швидко, а й дешево для бюджету.")
    assert "weak_endings" in _cats("Що буде далі — час покаже.")


# --- false positives (correct Ukrainian must NOT trip) -------------------------

def test_correct_forms_do_not_trigger_calques():
    good = "Протягом тижня учасники братимуть участь у заході. Наступний крок збігається з планом."
    assert check_ai_accent(good, PATTERNS) == []


def test_zyavlyaetsya_is_not_the_calque():
    # "з'являється" (appears) must not match the "являється" (=є) calque
    assert "calques" not in _cats("На небі з'являється нова зірка.")


def test_clean_news_text_has_no_hits():
    text = "НБУ знизив облікову ставку до 13%. Це здешевить кредити для бізнесу."
    assert check_ai_accent(text, PATTERNS) == []


# --- config --------------------------------------------------------------------

def test_bad_regex_rejected(tmp_path):
    p = tmp_path / "ai_accent.yaml"
    p.write_text(textwrap.dedent("""
        cliches: ['(oops']
    """), encoding="utf-8")
    with pytest.raises(AiAccentConfigError):
        load_ai_accent(p)
