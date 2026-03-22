from __future__ import annotations

from typing import Iterable

import numpy as np


def confusion_matrix(scores: Iterable[float], labels: Iterable[int], threshold: float) -> dict[str, int]:
    score_array, label_array = _validate_inputs(scores, labels)
    predictions = score_array >= float(threshold)
    positives = label_array == 1
    negatives = label_array == 0

    true_positive = int(np.sum(predictions & positives))
    false_positive = int(np.sum(predictions & negatives))
    true_negative = int(np.sum(~predictions & negatives))
    false_negative = int(np.sum(~predictions & positives))

    return {
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def accuracy(scores: Iterable[float], labels: Iterable[int], threshold: float) -> float:
    cm = confusion_matrix(scores, labels, threshold)
    total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
    return (cm["tp"] + cm["tn"]) / total


def precision(scores: Iterable[float], labels: Iterable[int], threshold: float) -> float:
    cm = confusion_matrix(scores, labels, threshold)
    denominator = cm["tp"] + cm["fp"]
    if denominator == 0:
        return 0.0
    return cm["tp"] / denominator


def recall(scores: Iterable[float], labels: Iterable[int], threshold: float) -> float:
    cm = confusion_matrix(scores, labels, threshold)
    denominator = cm["tp"] + cm["fn"]
    if denominator == 0:
        return 0.0
    return cm["tp"] / denominator


def f1(scores: Iterable[float], labels: Iterable[int], threshold: float) -> float:
    precision_value = precision(scores, labels, threshold)
    recall_value = recall(scores, labels, threshold)
    denominator = precision_value + recall_value
    if denominator == 0.0:
        return 0.0
    return 2.0 * precision_value * recall_value / denominator


def balanced_accuracy(scores: Iterable[float], labels: Iterable[int], threshold: float) -> float:
    cm = confusion_matrix(scores, labels, threshold)
    positive_total = cm["tp"] + cm["fn"]
    negative_total = cm["tn"] + cm["fp"]

    true_positive_rate = 0.0 if positive_total == 0 else cm["tp"] / positive_total
    true_negative_rate = 0.0 if negative_total == 0 else cm["tn"] / negative_total
    return (true_positive_rate + true_negative_rate) / 2.0


def _validate_inputs(scores: Iterable[float], labels: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(list(scores), dtype=np.float64)
    label_array = np.asarray(list(labels), dtype=np.int8)

    if score_array.ndim != 1:
        raise ValueError("Scores must be a 1D sequence.")
    if label_array.ndim != 1:
        raise ValueError("Labels must be a 1D sequence.")
    if score_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            f"Scores and labels must have the same length; got {score_array.shape[0]} and {label_array.shape[0]}."
        )
    if score_array.shape[0] == 0:
        raise ValueError("Scores and labels must be non-empty.")

    unique_labels = set(label_array.tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError("Labels must contain only binary values: 0 or 1.")

    return score_array, label_array
