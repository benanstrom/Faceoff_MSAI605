from __future__ import annotations

import numpy as np
import pytest

from lfw_verif.metrics import accuracy, balanced_accuracy, confusion_matrix, f1, precision, recall


def test_confusion_matrix_counts_predictions_at_threshold() -> None:
    scores = [0.9, 0.8, 0.4, 0.2]
    labels = [1, 0, 1, 0]

    cm = confusion_matrix(scores, labels, threshold=0.5)

    assert cm == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_metric_functions_compute_expected_values() -> None:
    scores = [0.9, 0.8, 0.4, 0.2]
    labels = [1, 0, 1, 0]

    assert np.isclose(accuracy(scores, labels, threshold=0.5), 0.5)
    assert np.isclose(precision(scores, labels, threshold=0.5), 0.5)
    assert np.isclose(recall(scores, labels, threshold=0.5), 0.5)
    assert np.isclose(f1(scores, labels, threshold=0.5), 0.5)
    assert np.isclose(balanced_accuracy(scores, labels, threshold=0.5), 0.5)


def test_precision_and_f1_return_zero_when_no_predicted_positives() -> None:
    scores = [0.1, 0.2, 0.3]
    labels = [1, 0, 1]

    assert precision(scores, labels, threshold=0.9) == 0.0
    assert f1(scores, labels, threshold=0.9) == 0.0


def test_balanced_accuracy_handles_missing_positive_class_predictions() -> None:
    scores = [0.1, 0.2, 0.3, 0.4]
    labels = [1, 1, 0, 0]

    assert np.isclose(balanced_accuracy(scores, labels, threshold=0.9), 0.5)


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        confusion_matrix([0.1, 0.2], [1], threshold=0.5)


def test_metrics_reject_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary values"):
        confusion_matrix([0.1, 0.2], [1, 2], threshold=0.5)
