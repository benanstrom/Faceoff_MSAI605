from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_pairs_cli_writes_json_and_plot_artifacts(tmp_path: Path) -> None:
    positive_pattern = np.array(
        [
            [255, 255, 0, 0],
            [255, 255, 0, 0],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )
    negative_left_pattern = np.array(
        [
            [255, 255, 255, 255],
            [255, 255, 255, 255],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    negative_right_pattern = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )

    _make_grayscale_image(tmp_path / "val_pos_left.png", positive_pattern)
    _make_grayscale_image(tmp_path / "val_pos_right.png", positive_pattern)
    _make_grayscale_image(tmp_path / "val_neg_left.png", negative_left_pattern)
    _make_grayscale_image(tmp_path / "val_neg_right.png", negative_right_pattern)
    _make_grayscale_image(tmp_path / "test_pos_left.png", positive_pattern)
    _make_grayscale_image(tmp_path / "test_pos_right.png", positive_pattern)
    _make_grayscale_image(tmp_path / "test_neg_left.png", negative_left_pattern)
    _make_grayscale_image(tmp_path / "test_neg_right.png", negative_right_pattern)

    pair_csv = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {"left_path": "val_pos_left.png", "right_path": "val_pos_right.png", "label": 1, "split": "val"},
            {"left_path": "val_neg_left.png", "right_path": "val_neg_right.png", "label": 0, "split": "val"},
            {"left_path": "test_pos_left.png", "right_path": "test_pos_right.png", "label": 1, "split": "test"},
            {"left_path": "test_neg_left.png", "right_path": "test_neg_right.png", "label": 0, "split": "test"},
        ]
    ).to_csv(pair_csv, index=False)

    config_path = tmp_path / "m2_eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: script_eval",
                "feature_extractor: grayscale_flatten_l2",
                "similarity_metric: cosine",
                "threshold_sweep:",
                "  start: 0.0",
                "  stop: 1.0",
                "  step: 0.5",
                "threshold_selection_rule: maximize_balanced_accuracy",
                "threshold_selection_split: val",
                "final_evaluation_split: test",
                "tracked_run_dir: artifacts/runs",
                "notes: test config",
                "",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "eval_outputs"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_pairs.py"),
            "--pairs",
            str(pair_csv),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--image-size",
            "4",
            "4",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Wrote scores:" in result.stdout

    scores_payload = json.loads((output_dir / "scores.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    sweep_payload = json.loads((output_dir / "threshold_sweep.json").read_text(encoding="utf-8"))

    assert len(scores_payload["scores"]) == 4
    assert metrics_payload["selected_threshold"] == 0.5
    assert metrics_payload["final_metrics"]["balanced_accuracy"] == 1.0
    assert len(sweep_payload["threshold_sweep"]) == 3
    assert (output_dir / "roc.png").exists()
    assert (output_dir / "confusion_matrix.png").exists()


def _make_grayscale_image(path: Path, pixels: int | np.ndarray) -> Path:
    if isinstance(pixels, np.ndarray):
        Image.fromarray(pixels, mode="L").save(path)
        return path

    Image.new("L", (4, 4), color=pixels).save(path)
    return path
