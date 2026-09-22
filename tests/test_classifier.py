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
    assert c.keywords == []             # missing keywords -> empty
    assert c.topic_path == []           # missing topic_path -> empty


def test_parse_keywords_extracted_and_capped():
    kws = [f"тема{i}" for i in range(15)]
    import json
    c = parse_classification(json.dumps({"is_event": True, "keywords": kws + ["", "  "]}))
    assert c is not None
    assert c.keywords[:3] == ["тема0", "тема1", "тема2"]
    assert len(c.keywords) == 10        # capped at 10, blanks dropped


def test_parse_topic_path_lowercased_and_capped():
    import json
    c = parse_classification(json.dumps({
        "is_event": True,
        "topic_path": ["Війна", " Атака РФ ", "Удар БпЛА", "Одеса", "Затока", "зайве"]}))
    assert c is not None
    # lowercased + trimmed, capped at 5 (punctuation is normalized later, at ingest)
    assert c.topic_path == ["війна", "атака рф", "удар бпла", "одеса", "затока"]
    assert parse_classification('{"is_event": true}').topic_path == []                # missing -> empty


def test_parse_facets_keeps_only_fixed_dimensions_and_evidence():
    import json

    c = parse_classification(json.dumps({
        "is_event": True,
        "facets": [
            {"dimension": "geography", "path": ["Україна", "Київська область"],
             "confidence": 1.7, "evidence": "у Київській області"},
            {"dimension": "made_up_axis", "path": ["x"], "confidence": 1},
            {"dimension": "impact", "path": [], "confidence": 0.9},
        ],
    }, ensure_ascii=False))
    assert c is not None and len(c.facets) == 1
    assert c.facets[0].dimension == "geography"
    assert c.facets[0].path == ("Україна", "Київська область")
    assert c.facets[0].confidence == 1.0
    assert c.facets[0].evidence == "у Київській області"


def test_parse_compact_facts_preserves_modality_and_clamps_values():
    import json

    c = parse_classification(json.dumps({
        "is_event": True,
        "facts": [
            {"text": "Уряд заявив про 10 об'єктів", "modality": "statement",
             "attribution": "уряд", "number": "10", "unit": "об'єкт"},
            {"text": "Може зрости", "modality": "forecast", "time_frame": "у 2027 році"},
            {"text": "Невідомий тип", "modality": "certain"},
            {"text": ""},
        ],
    }, ensure_ascii=False))
    assert c is not None and len(c.compact_facts) == 3
    assert c.compact_facts[0]["modality"] == "statement"
    assert c.compact_facts[0]["number"] == 10.0
    assert c.compact_facts[1]["time_frame"] == "у 2027 році"
    assert c.compact_facts[2]["modality"] == "fact"
