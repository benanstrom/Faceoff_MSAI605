from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from lfw_verif.scoring import score_pairs


def test_score_pairs_returns_one_score_per_csv_row(tmp_path: Path) -> None:
    left_a = _make_grayscale_image(tmp_path / "left_a.png", 0)
    right_a = _make_grayscale_image(tmp_path / "right_a.png", 255)
    left_b = _make_grayscale_image(tmp_path / "left_b.png", 128)
    right_b = _make_grayscale_image(tmp_path / "right_b.png", 128)

    pair_csv = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {"left_path": left_a.name, "right_path": right_a.name, "label": 0, "split": "val"},
            {"left_path": left_b.name, "right_path": right_b.name, "label": 1, "split": "val"},
        ]
    ).to_csv(pair_csv, index=False)

    scores = score_pairs(pair_csv, image_size=(4, 4))

    assert isinstance(scores, list)
    assert len(scores) == 2


def test_score_pairs_is_deterministic_for_same_input_csv(tmp_path: Path) -> None:
    left = _make_grayscale_image(tmp_path / "left.png", 200)
    right = _make_grayscale_image(tmp_path / "right.png", 200)

    pair_csv = tmp_path / "pairs.csv"
    pd.DataFrame([{"left_path": left.name, "right_path": right.name}]).to_csv(pair_csv, index=False)

    scores_a = score_pairs(pair_csv, image_size=(4, 4))
    scores_b = score_pairs(pair_csv, image_size=(4, 4))

    assert scores_a == scores_b


def test_score_pairs_supports_euclidean_metric(tmp_path: Path) -> None:
    left = _make_grayscale_image(tmp_path / "left.png", 0)
    right = _make_grayscale_image(tmp_path / "right.png", 255)

    pair_csv = tmp_path / "pairs.csv"
    pd.DataFrame([{"left_path": left.name, "right_path": right.name}]).to_csv(pair_csv, index=False)

    scores = score_pairs(pair_csv, image_size=(4, 4), similarity_metric="euclidean")

    assert len(scores) == 1
    assert scores[0] > 0.0


def test_score_pairs_rejects_missing_required_columns(tmp_path: Path) -> None:
    pair_csv = tmp_path / "pairs.csv"
    pd.DataFrame([{"left_path": "a.png"}]).to_csv(pair_csv, index=False)

    with pytest.raises(ValueError, match="missing required column\\(s\\): right_path"):
        score_pairs(pair_csv)


def test_score_pairs_returns_high_cosine_for_identical_images(tmp_path: Path) -> None:
    left = _make_grayscale_image(tmp_path / "left.png", 150)
    right = _make_grayscale_image(tmp_path / "right.png", 150)

    pair_csv = tmp_path / "pairs.csv"
    pd.DataFrame([{"left_path": left.name, "right_path": right.name}]).to_csv(pair_csv, index=False)

    scores = score_pairs(pair_csv, image_size=(4, 4), similarity_metric="cosine")

    assert np.isclose(scores[0], 1.0)


def _make_grayscale_image(path: Path, pixel_value: int) -> Path:
    Image.new("L", (4, 4), color=pixel_value).save(path)
    return path
