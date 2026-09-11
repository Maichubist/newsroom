from __future__ import annotations

from newsroom.analyze.classifier import parse_classification


def test_parse_valid_normalizes_rubrics():
    c = parse_classification(
        '{"is_event": true, "rubrics": ["War", " Politics "], "side": "ua", '
        '"is_first_source": true, "is_rumor": false}'
    )
    assert c is not None
    assert c.is_event is True
    assert c.rubrics == ["war", "politics"]
    assert c.side == "ua" and c.is_first_source is True and c.is_rumor is False


def test_parse_invalid_json_is_none():
    assert parse_classification("not json at all") is None
    assert parse_classification("") is None
    assert parse_classification(None) is None
    assert parse_classification("[1, 2, 3]") is None   # not an object


def test_parse_clamps_side_and_defaults_missing_fields():
    c = parse_classification('{"is_event": false, "side": "martian"}')
    assert c is not None
    assert c.is_event is False
    assert c.side == "unknown"          # invalid side clamped
    assert c.rubrics == [] and c.is_rumor is False and c.is_first_source is False
