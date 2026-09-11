from __future__ import annotations

from newsroom.analyze.independence import (
    SourceItem,
    group_materials,
    independent_source_count,
)


def test_two_independent_reports_count_as_two():
    items = [
        SourceItem(source_id=1, content_hash="a" * 64, simhash=0b0, source_name="УП"),
        SourceItem(source_id=2, content_hash="b" * 64, simhash=(1 << 40) - 1, source_name="Суспільне"),
    ]
    assert independent_source_count(items) == 2


def test_verbatim_reprint_from_two_channels_is_one_source():
    items = [
        SourceItem(source_id=1, content_hash="same" + "0" * 60, source_name="УП"),
        SourceItem(source_id=2, content_hash="same" + "0" * 60, source_name="Копіпаста"),
    ]
    assert independent_source_count(items) == 1


def test_near_duplicate_within_threshold_is_one_source():
    a = 0
    near = 0b111            # Hamming 3 <= 6
    far = (1 << 50) - 1     # Hamming 50
    assert independent_source_count([
        SourceItem(source_id=1, simhash=a, source_name="A"),
        SourceItem(source_id=2, simhash=near, source_name="B"),
    ]) == 1
    assert independent_source_count([
        SourceItem(source_id=1, simhash=a, source_name="A"),
        SourceItem(source_id=2, simhash=far, source_name="B"),
    ]) == 2


def test_same_channel_twice_is_one_source():
    items = [
        SourceItem(source_id=5, content_hash="x" * 64, source_name="Канал"),
        SourceItem(source_id=5, content_hash="y" * 64, source_name="Канал"),
    ]
    assert independent_source_count(items) == 1


def test_forward_is_not_independent_of_origin():
    # channel 2 forwards a post that originated from "УП" (channel 1)
    items = [
        SourceItem(source_id=1, content_hash="a" * 64, source_name="УП"),
        SourceItem(source_id=2, content_hash="z" * 64, source_name="Агрегатор", forwarded_from="УП"),
    ]
    assert independent_source_count(items) == 1


def test_two_forwards_of_same_origin_collapse():
    items = [
        SourceItem(source_id=2, source_name="A", forwarded_from="Генштаб"),
        SourceItem(source_id=3, source_name="B", forwarded_from="Генштаб"),
    ]
    assert independent_source_count(items) == 1


def test_mixed_cluster_origin_plus_reprints_plus_one_independent():
    items = [
        SourceItem(source_id=1, content_hash="orig" + "0" * 60, simhash=0, source_name="УП"),      # origin
        SourceItem(source_id=2, content_hash="orig" + "0" * 60, simhash=0, source_name="Repost1"),  # verbatim reprint
        SourceItem(source_id=3, simhash=0b11, source_name="Repost2"),                               # near-dup (H=2)
        SourceItem(source_id=4, content_hash="diff" + "0" * 60, simhash=(1 << 55) - 1, source_name="Суспільне"),  # independent
    ]
    groups = group_materials(items)
    assert len(groups) == 2
    assert independent_source_count(items) == 2


def test_empty_is_zero():
    assert independent_source_count([]) == 0
