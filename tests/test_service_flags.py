from __future__ import annotations

import pytest

from newsroom.service import (
    editorial_enabled,
    factbase_enabled,
    factcheck_enabled,
    story_updates_enabled,
    verify_enabled,
)

FLAGS = {
    "VERIFY_ENABLED": verify_enabled,
    "FACTBASE_ENABLED": factbase_enabled,
    "FACTCHECK_ENABLED": factcheck_enabled,
    "STORY_UPDATES_ENABLED": story_updates_enabled,
    "EDITORIAL_ENABLED": editorial_enabled,
}


@pytest.mark.parametrize("env", list(FLAGS))
def test_flag_defaults_off_when_unset(env, monkeypatch):
    monkeypatch.delenv(env, raising=False)
    assert FLAGS[env]() is False


@pytest.mark.parametrize("env", list(FLAGS))
@pytest.mark.parametrize("value,expected", [
    ("true", True), ("TRUE", True), ("1", True), ("yes", True), (" YeS ", True),
    ("false", False), ("0", False), ("no", False), ("", False), ("maybe", False),
])
def test_flag_parsing(env, value, expected, monkeypatch):
    monkeypatch.setenv(env, value)
    assert FLAGS[env]() is expected
