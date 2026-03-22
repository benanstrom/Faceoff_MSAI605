from __future__ import annotations

import numpy as np
import pytest

from lfw_verif.thresholds import evaluate_thresholds, select_best_threshold


def test_evaluate_thresholds_returns_metrics_for_each_threshold() -> None:
    scores = [0.9, 0.8, 0.4, 0.2]
    labels = [1, 0, 1, 0]

    results = evaluate_thresholds(scores, labels, thresholds=[0.25, 0.5, 0.75])

    assert len(results) == 3
    assert [result["threshold"] for result in results] == [0.25, 0.5, 0.75]
    assert all("balanced_accuracy" in result for result in results)
    assert all("tp" in result for result in results)


def test_select_best_threshold_maximizes_balanced_accuracy() -> None:
    scores = [0.95, 0.8, 0.45, 0.2]
    labels = [1, 0, 1, 0]

    best = select_best_threshold(scores, labels, thresholds=[0.2, 0.5, 0.9])

    assert np.isclose(best["threshold"], 0.9)
    assert np.isclose(best["balanced_accuracy"], 0.75)


def test_select_best_threshold_keeps_lowest_threshold_on_tie() -> None:
    scores = [0.9, 0.1]
    labels = [1, 0]

    best = select_best_threshold(scores, labels, thresholds=[0.2, 0.3, 0.4])

    assert np.isclose(best["threshold"], 0.2)
    assert np.isclose(best["balanced_accuracy"], 1.0)


def test_select_best_threshold_requires_at_least_one_threshold() -> None:
    with pytest.raises(ValueError, match="At least one threshold"):
        select_best_threshold([0.1, 0.9], [0, 1], thresholds=[])
