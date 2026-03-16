from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lfw_verif.images import load_grayscale_image


def test_load_grayscale_image_returns_resized_array(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    image = Image.new("RGB", (4, 3), color=(10, 20, 30))
    image.save(image_path)

    loaded = load_grayscale_image(image_path, size=(8, 6))

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (6, 8)
    assert loaded.dtype == np.uint8


def test_load_grayscale_image_converts_rgb_to_grayscale_values(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    image = Image.new("RGB", (1, 1), color=(255, 0, 0))
    image.save(image_path)

    loaded = load_grayscale_image(image_path, size=(1, 1))

    assert loaded.shape == (1, 1)
    assert int(loaded[0, 0]) == 76


def test_load_grayscale_image_is_deterministic(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    image = Image.new("RGB", (5, 5), color=(123, 45, 67))
    image.save(image_path)

    loaded_a = load_grayscale_image(image_path, size=(7, 7))
    loaded_b = load_grayscale_image(image_path, size=(7, 7))

    assert np.array_equal(loaded_a, loaded_b)


def test_load_grayscale_image_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match="Image path does not exist"):
        load_grayscale_image(missing_path, size=(8, 8))


def test_load_grayscale_image_rejects_invalid_size(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(image_path)

    with pytest.raises(ValueError, match="greater than 0"):
        load_grayscale_image(image_path, size=(0, 8))
