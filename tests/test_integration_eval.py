from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tracked_evaluation_produces_artifacts_and_is_deterministic(tmp_path: Path) -> None:
    pair_csv = _build_tiny_fixture_dataset(tmp_path)
    config_path = _write_eval_config(tmp_path)

    first_run = _run_tracked_eval(pair_csv, config_path)
    second_run = _run_tracked_eval(pair_csv, config_path)

    first_run_dir = _extract_path_from_stdout(first_run.stdout, "Run directory:")
    second_run_dir = _extract_path_from_stdout(second_run.stdout, "Run directory:")
    assert first_run_dir != second_run_dir

    _assert_run_artifacts_exist(first_run_dir)
    _assert_run_artifacts_exist(second_run_dir)

    first_scores = json.loads((first_run_dir / "scores.json").read_text(encoding="utf-8"))
    first_metrics = json.loads((first_run_dir / "metrics.json").read_text(encoding="utf-8"))
    first_sweep = json.loads((first_run_dir / "threshold_sweep.json").read_text(encoding="utf-8"))
    first_manifest = json.loads((first_run_dir / "run.json").read_text(encoding="utf-8"))

    second_scores = json.loads((second_run_dir / "scores.json").read_text(encoding="utf-8"))
    second_metrics = json.loads((second_run_dir / "metrics.json").read_text(encoding="utf-8"))
    second_sweep = json.loads((second_run_dir / "threshold_sweep.json").read_text(encoding="utf-8"))

    assert first_scores == second_scores
    assert first_metrics == second_metrics
    assert first_sweep == second_sweep
    assert first_metrics["selected_threshold"] == 0.5
    assert first_metrics["final_metrics"]["balanced_accuracy"] == 1.0

    assert first_manifest["run_id"] == first_run_dir.name
    assert first_manifest["code_version"]["git_commit_hash"] is not None
    assert first_manifest["config"]["experiment_name"] == "tracked_integration_eval"
    assert first_manifest["data_version"]["pair_row_count"] == 4
    assert first_manifest["data_version"]["split_counts"] == {"test": 2, "val": 2}
    assert first_manifest["threshold"] == 0.5
    assert first_manifest["metrics"]["final_metrics"]["balanced_accuracy"] == 1.0
    assert first_manifest["notes"] == "integration test config"


def _run_tracked_eval(pair_csv: Path, config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_tracked_eval.py"),
            "--pairs",
            str(pair_csv),
            "--config",
            str(config_path),
            "--image-size",
            "4",
            "4",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def _build_tiny_fixture_dataset(tmp_path: Path) -> Path:
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
    return pair_csv


def _write_eval_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "m2_eval.yaml"
    tracked_root = tmp_path / "artifacts" / "runs"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: tracked_integration_eval",
                "feature_extractor: grayscale_flatten_l2",
                "similarity_metric: cosine",
                "threshold_sweep:",
                "  start: 0.0",
                "  stop: 1.0",
                "  step: 0.5",
                "threshold_selection_rule: maximize_balanced_accuracy",
                "threshold_selection_split: val",
                "final_evaluation_split: test",
                f"tracked_run_dir: {tracked_root.as_posix()}",
                "notes: integration test config",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _assert_run_artifacts_exist(run_dir: Path) -> None:
    assert run_dir.exists()
    assert (run_dir / "run.json").exists()
    assert (run_dir / "scores.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "threshold_sweep.json").exists()
    assert (run_dir / "roc.png").exists()
    assert (run_dir / "confusion_matrix.png").exists()


def _extract_path_from_stdout(stdout: str, prefix: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return Path(line.split(":", 1)[1].strip())
    raise AssertionError(f"Could not find '{prefix}' in subprocess output:\n{stdout}")


def _make_grayscale_image(path: Path, pixels: int | np.ndarray) -> Path:
    if isinstance(pixels, np.ndarray):
        Image.fromarray(pixels, mode="L").save(path)
        return path

    Image.new("L", (4, 4), color=pixels).save(path)
    return path
