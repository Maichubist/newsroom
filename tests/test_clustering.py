from __future__ import annotations

import pytest

from newsroom.analyze.clustering import best_match, cosine, update_centroid


def test_cosine_identical_orthogonal_zero():
    assert cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0            # zero-vector guard


def test_update_centroid_is_running_mean():
    assert update_centroid([0, 0], 0, [2, 4]) == [2.0, 4.0]
    assert update_centroid([2, 0], 1, [0, 2]) == [1.0, 1.0]
    assert update_centroid([1, 1], 3, [5, 5]) == [2.0, 2.0]


def test_best_match_picks_nearest_above_threshold():
    idx, sim = best_match([1, 0], [[1, 0], [0, 1]], threshold=0.83)
    assert idx == 0 and sim == pytest.approx(1.0)


def test_best_match_returns_none_below_threshold():
    idx, sim = best_match([1, 1], [[1, 0]], threshold=0.83)
    assert idx is None and sim == pytest.approx(0.7071, abs=1e-3)


def test_best_match_empty_centroids():
    assert best_match([1, 0], []) == (None, 0.0)
