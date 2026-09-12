from __future__ import annotations

import datetime as dt

from newsroom.collectors.base import is_recent

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 9, 12, 12, 0, tzinfo=UTC)


def test_recent_item_kept():
    assert is_recent(NOW - dt.timedelta(hours=5), NOW, 24) is True


def test_old_item_dropped():
    assert is_recent(NOW - dt.timedelta(hours=30), NOW, 24) is False


def test_boundary_just_inside_kept():
    assert is_recent(NOW - dt.timedelta(hours=24) + dt.timedelta(minutes=1), NOW, 24) is True


def test_missing_date_is_kept():
    # a feed with no date is usually serving current items — don't drop possibly-fresh news
    assert is_recent(None, NOW, 24) is True


def test_zero_or_negative_window_disables_filter():
    assert is_recent(NOW - dt.timedelta(days=365), NOW, 0) is True
    assert is_recent(NOW - dt.timedelta(days=365), NOW, -1) is True
