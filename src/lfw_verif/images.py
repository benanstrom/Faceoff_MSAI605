from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale_image(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: '{image_path}'.")

    width, height = _validate_size(size)
    with Image.open(image_path) as image:
        # Convert before resizing so every caller gets the same single-channel preprocessing path.
        processed = image.convert("L").resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(processed, dtype=np.uint8)


def _validate_size(size: tuple[int, int]) -> tuple[int, int]:
    if len(size) != 2:
        raise ValueError("Image size must contain exactly two integers: (width, height).")

    width, height = size
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (width, height)):
        raise ValueError("Image size values must be integers.")
    if width <= 0 or height <= 0:
        raise ValueError("Image size values must be greater than 0.")
    return width, height
