from __future__ import annotations

import pytest

from newsroom.service import (
    editorial_enabled,
    factbase_enabled,
    factcheck_enabled,
    media_check_enabled,
    media_download_enabled,
    media_moderation_enabled,
    metrics_enabled,
    monitoring_enabled,
    reputation_enabled,
    story_updates_enabled,
    verify_enabled,
)

FLAGS = {
    "VERIFY_ENABLED": verify_enabled,
    "FACTBASE_ENABLED": factbase_enabled,
    "FACTCHECK_ENABLED": factcheck_enabled,
    "STORY_UPDATES_ENABLED": story_updates_enabled,
    "EDITORIAL_ENABLED": editorial_enabled,
    "MEDIA_DOWNLOAD_ENABLED": media_download_enabled,
    "MEDIA_CHECK_ENABLED": media_check_enabled,
    "MEDIA_MODERATION_ENABLED": media_moderation_enabled,
    "REPUTATION_ENABLED": reputation_enabled,
    "MONITORING_ENABLED": monitoring_enabled,
    "METRICS_ENABLED": metrics_enabled,
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
