from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from lfw_verif.metrics import confusion_matrix


def plot_roc_curve(scores: Iterable[float], labels: Iterable[int], output_path: str | Path) -> Path:
    score_array, label_array = _validate_curve_inputs(scores, labels)
    fpr_values, tpr_values = _compute_roc_points(score_array, label_array)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(fpr_values, tpr_values, color="#0b6e4f", linewidth=2, label="ROC")
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1, label="Chance")
    axis.set_title("ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def plot_confusion_matrix(
    scores: Iterable[float], labels: Iterable[int], threshold: float, output_path: str | Path
) -> Path:
    matrix = confusion_matrix(scores, labels, threshold)
    values = np.asarray([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]], dtype=np.int64)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values, cmap="Blues")
    axis.set_title(f"Confusion Matrix @ threshold={threshold:.3f}")
    axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    axis.set_yticks([0, 1], labels=["True 0", "True 1"])

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(int(values[row_index, column_index])),
                ha="center",
                va="center",
                color="#111111",
            )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def _validate_curve_inputs(scores: Iterable[float], labels: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(list(scores), dtype=np.float64)
    label_array = np.asarray(list(labels), dtype=np.int8)

    if score_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError("Scores and labels must be 1D sequences.")
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


def _compute_roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_thresholds = sorted(set(scores.tolist()), reverse=True)
    thresholds = [float("inf"), *unique_thresholds, float("-inf")]

    fpr_values = []
    tpr_values = []
    positive_total = int(np.sum(labels == 1))
    negative_total = int(np.sum(labels == 0))

    for threshold in thresholds:
        predictions = scores >= threshold
        true_positive = int(np.sum(predictions & (labels == 1)))
        false_positive = int(np.sum(predictions & (labels == 0)))

        tpr = 0.0 if positive_total == 0 else true_positive / positive_total
        fpr = 0.0 if negative_total == 0 else false_positive / negative_total
        tpr_values.append(tpr)
        fpr_values.append(fpr)

    return np.asarray(fpr_values, dtype=np.float64), np.asarray(tpr_values, dtype=np.float64)
