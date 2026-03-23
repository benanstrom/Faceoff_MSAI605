from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lfw_verif.eval_config import EvalConfig, ThresholdSweep, load_eval_config
from lfw_verif.metrics import accuracy, balanced_accuracy, confusion_matrix, f1, precision, recall
from lfw_verif.plots import plot_confusion_matrix, plot_roc_curve
from lfw_verif.scoring import score_pairs
from lfw_verif.thresholds import evaluate_thresholds, select_best_threshold
from lfw_verif.validation import validate_pair_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate face verification pairs with threshold selection.")
    parser.add_argument("--pairs", required=True, help="Path to pair CSV with left_path, right_path, label, split.")
    parser.add_argument("--config", default="configs/m2_baseline.yaml", help="Milestone 2 evaluation config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON outputs and plots. Defaults to <tracked_run_dir>/<experiment_name>.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(32, 32),
        help="Image size for baseline grayscale feature extraction.",
    )
    args = parser.parse_args()

    config = load_eval_config(args.config)
    output_dir = _resolve_output_dir(config, args.output_dir)
    artifacts = run_evaluation(
        pair_csv=args.pairs,
        config=config,
        output_dir=output_dir,
        image_size=tuple(args.image_size),
    )

    print(f"Wrote scores: {artifacts['scores_json']}")
    print(f"Wrote metrics: {artifacts['metrics_json']}")
    print(f"Wrote threshold sweep: {artifacts['threshold_sweep_json']}")
    print(f"Wrote ROC plot: {artifacts['roc_png']}")
    print(f"Wrote confusion matrix plot: {artifacts['confusion_matrix_png']}")


def run_evaluation(
    pair_csv: str | Path,
    config: EvalConfig,
    output_dir: str | Path,
    image_size: tuple[int, int] = (32, 32),
) -> dict[str, Any]:
    _validate_supported_config(config)
    pair_csv_path = validate_pair_csv(pair_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pairs_df = pd.read_csv(pair_csv_path)
    scores = score_pairs(pair_csv_path, image_size=image_size, similarity_metric=config.similarity_metric)
    pairs_df = pairs_df.copy()
    pairs_df["score"] = scores

    selection_df = _select_split_rows(pairs_df, config.threshold_selection_split, pair_csv_path)
    final_df = _select_split_rows(pairs_df, config.final_evaluation_split, pair_csv_path)
    thresholds = _build_thresholds(config.threshold_sweep)

    threshold_results = evaluate_thresholds(
        selection_df["score"].tolist(),
        selection_df["label"].tolist(),
        thresholds=thresholds,
    )
    best_threshold = select_best_threshold(
        selection_df["score"].tolist(),
        selection_df["label"].tolist(),
        thresholds=thresholds,
    )
    final_metrics = _compute_metric_bundle(
        final_df["score"].tolist(),
        final_df["label"].tolist(),
        threshold=float(best_threshold["threshold"]),
    )

    scores_json = output_path / "scores.json"
    metrics_json = output_path / "metrics.json"
    threshold_sweep_json = output_path / "threshold_sweep.json"
    roc_png = output_path / "roc.png"
    confusion_matrix_png = output_path / "confusion_matrix.png"

    _write_json(
        scores_json,
        {
            "experiment_name": config.experiment_name,
            "pair_csv": str(pair_csv_path),
            "similarity_metric": config.similarity_metric,
            "feature_extractor": config.feature_extractor,
            "scores": pairs_df.to_dict(orient="records"),
        },
    )
    _write_json(
        metrics_json,
        {
            "experiment_name": config.experiment_name,
            "threshold_selection_rule": config.threshold_selection_rule,
            "threshold_selection_split": config.threshold_selection_split,
            "final_evaluation_split": config.final_evaluation_split,
            "selected_threshold": float(best_threshold["threshold"]),
            "selection_metrics": _json_ready(best_threshold),
            "final_metrics": final_metrics,
        },
    )
    _write_json(
        threshold_sweep_json,
        {
            "experiment_name": config.experiment_name,
            "split": config.threshold_selection_split,
            "threshold_sweep": [dict(result) for result in threshold_results],
        },
    )

    plot_roc_curve(
        final_df["score"].tolist(),
        final_df["label"].tolist(),
        roc_png,
    )
    plot_confusion_matrix(
        final_df["score"].tolist(),
        final_df["label"].tolist(),
        float(best_threshold["threshold"]),
        confusion_matrix_png,
    )

    return {
        "pair_csv": pair_csv_path,
        "output_dir": output_path,
        "selected_threshold": float(best_threshold["threshold"]),
        "selection_metrics": _json_ready(best_threshold),
        "final_metrics": final_metrics,
        "threshold_sweep": [dict(result) for result in threshold_results],
        "scores_json": scores_json,
        "metrics_json": metrics_json,
        "threshold_sweep_json": threshold_sweep_json,
        "roc_png": roc_png,
        "confusion_matrix_png": confusion_matrix_png,
    }


def _resolve_output_dir(config: EvalConfig, cli_output_dir: str | None) -> Path:
    if cli_output_dir is not None:
        return Path(cli_output_dir)
    return config.tracked_run_dir / config.experiment_name


def _validate_supported_config(config: EvalConfig) -> None:
    if config.feature_extractor != "grayscale_flatten_l2":
        raise ValueError(
            "Only the 'grayscale_flatten_l2' feature extractor is currently supported by evaluate_pairs.py."
        )
    if config.similarity_metric != "cosine":
        raise ValueError("Only cosine similarity is currently supported by evaluate_pairs.py.")
    if config.threshold_selection_rule != "maximize_balanced_accuracy":
        raise ValueError(
            "Only the 'maximize_balanced_accuracy' threshold selection rule is currently supported."
        )


def _select_split_rows(pairs_df: pd.DataFrame, split_name: str, pair_csv_path: Path) -> pd.DataFrame:
    split_df = pairs_df.loc[pairs_df["split"] == split_name].reset_index(drop=True)
    if split_df.empty:
        raise ValueError(f"Pair CSV '{pair_csv_path}' does not contain any rows for split '{split_name}'.")
    return split_df


def _build_thresholds(sweep: ThresholdSweep) -> list[float]:
    start = Decimal(str(sweep.start))
    stop = Decimal(str(sweep.stop))
    step = Decimal(str(sweep.step))
    values: list[float] = []
    current = start

    while current <= stop:
        values.append(float(current))
        current += step

    if not values:
        raise ValueError("Threshold sweep produced no threshold values.")
    return values


def _compute_metric_bundle(scores: list[float], labels: list[int], threshold: float) -> dict[str, float | int]:
    cm = confusion_matrix(scores, labels, threshold)
    return {
        "threshold": float(threshold),
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    main()
