from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from lfw_verif.features import extract_baseline_features


def test_extract_baseline_features_returns_fixed_length_vector(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(image_path)

    features = extract_baseline_features(image_path, size=(8, 6))

    assert isinstance(features, np.ndarray)
    assert features.shape == (48,)
    assert features.dtype == np.float64


def test_extract_baseline_features_normalizes_vector(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(image_path)

    features = extract_baseline_features(image_path, size=(2, 2))

    assert np.isclose(np.linalg.norm(features), 1.0)


def test_extract_baseline_features_is_deterministic(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (5, 5), color=(123, 45, 67)).save(image_path)

    features_a = extract_baseline_features(image_path, size=(7, 7))
    features_b = extract_baseline_features(image_path, size=(7, 7))

    assert np.array_equal(features_a, features_b)


def test_extract_baseline_features_returns_zero_vector_for_zero_norm_input(tmp_path: Path) -> None:
    image_path = tmp_path / "black.png"
    Image.new("L", (3, 3), color=0).save(image_path)

    features = extract_baseline_features(image_path, size=(3, 3))

    assert np.array_equal(features, np.zeros((9,), dtype=np.float64))
