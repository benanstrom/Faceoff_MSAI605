from __future__ import annotations

from pathlib import Path

import pytest

from lfw_verif.eval_config import EvalConfigError, load_eval_config


def test_load_eval_config_reads_baseline_config() -> None:
    cfg = load_eval_config("configs/m2_baseline.yaml")

    assert cfg.experiment_name == "m2_baseline"
    assert cfg.feature_extractor == "embeddings_from_manifest"
    assert cfg.similarity_metric == "cosine"
    assert cfg.threshold_sweep.start == 0.0
    assert cfg.threshold_sweep.stop == 1.0
    assert cfg.threshold_sweep.step == 0.01
    assert cfg.threshold_selection_rule == "max_validation_accuracy"
    assert cfg.threshold_selection_split == "val"
    assert cfg.final_evaluation_split == "test"
    assert cfg.tracked_run_dir == Path("outputs/runs/m2_baseline")


def test_load_eval_config_rejects_missing_required_field(tmp_path: Path) -> None:
    cfg_path = tmp_path / "missing.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "experiment_name: test_eval",
                "feature_extractor: embeddings_from_manifest",
                "similarity_metric: cosine",
                "threshold_sweep:",
                "  start: 0.0",
                "  stop: 1.0",
                "  step: 0.1",
                "threshold_selection_rule: max_validation_accuracy",
                "threshold_selection_split: val",
                "tracked_run_dir: outputs/runs/test_eval",
                "notes: missing final_evaluation_split for test",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvalConfigError, match="missing required field\\(s\\): final_evaluation_split"):
        load_eval_config(cfg_path)


def test_load_eval_config_rejects_non_positive_threshold_step(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad_step.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "experiment_name: test_eval",
                "feature_extractor: embeddings_from_manifest",
                "similarity_metric: cosine",
                "threshold_sweep:",
                "  start: 0.0",
                "  stop: 1.0",
                "  step: 0.0",
                "threshold_selection_rule: max_validation_accuracy",
                "threshold_selection_split: val",
                "final_evaluation_split: test",
                "tracked_run_dir: outputs/runs/test_eval",
                "notes: invalid threshold step",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvalConfigError, match="threshold_sweep.step"):
        load_eval_config(cfg_path)


def test_load_eval_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "not_mapping.yaml"
    cfg_path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(EvalConfigError, match="top-level mapping"):
        load_eval_config(cfg_path)
