from __future__ import annotations

from pathlib import Path

import numpy as np

from lfw_verif.images import load_grayscale_image


def extract_baseline_features(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    image = load_grayscale_image(path, size=size)
    vector = image.astype(np.float64, copy=False).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        return np.zeros_like(vector)
    return vector / norm
