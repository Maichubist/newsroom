from __future__ import annotations

from types import SimpleNamespace

from newsroom.analyze.verify import _derive_event_title


def _row(title=None, text=None):
    return (SimpleNamespace(title=title, text=text), SimpleNamespace(name="s"))


def test_prefers_item_title():
    rows = [_row(title="", text="якийсь текст"), _row(title="Справжній заголовок", text="…")]
    assert _derive_event_title(rows) == "Справжній заголовок"


def test_falls_back_to_first_line_of_text_for_titleless_telegram():
    rows = [_row(title=None, text="Ударний БпЛА зафіксовано на Черкащині\nдеталі нижче")]
    assert _derive_event_title(rows) == "Ударний БпЛА зафіксовано на Черкащині"


def test_none_when_no_title_or_text():
    assert _derive_event_title([_row(title="", text="")]) is None
    assert _derive_event_title([]) is None


def test_caps_length():
    long = "х" * 500
    assert len(_derive_event_title([_row(text=long)])) == 200
