from __future__ import annotations

from typing import Iterable

from lfw_verif.metrics import accuracy, balanced_accuracy, confusion_matrix, f1, precision, recall


def evaluate_thresholds(
    scores: Iterable[float], labels: Iterable[int], thresholds: Iterable[float]
) -> list[dict[str, float | int]]:
    evaluated = []
    for threshold in _sorted_thresholds(thresholds):
        cm = confusion_matrix(scores, labels, threshold)
        evaluated.append(
            {
                "threshold": threshold,
                "accuracy": accuracy(scores, labels, threshold),
                "precision": precision(scores, labels, threshold),
                "recall": recall(scores, labels, threshold),
                "f1": f1(scores, labels, threshold),
                "balanced_accuracy": balanced_accuracy(scores, labels, threshold),
                "tp": cm["tp"],
                "fp": cm["fp"],
                "tn": cm["tn"],
                "fn": cm["fn"],
            }
        )
    return evaluated


def select_best_threshold(
    scores: Iterable[float], labels: Iterable[int], thresholds: Iterable[float]
) -> dict[str, float | int]:
    results = evaluate_thresholds(scores, labels, thresholds)
    if not results:
        raise ValueError("At least one threshold is required.")

    # Keep the lowest threshold on ties by iterating in sorted order.
    best = results[0]
    for result in results[1:]:
        if float(result["balanced_accuracy"]) > float(best["balanced_accuracy"]):
            best = result
    return best


def _sorted_thresholds(thresholds: Iterable[float]) -> list[float]:
    values = [float(threshold) for threshold in thresholds]
    if not values:
        return []
    return sorted(values)
