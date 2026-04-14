from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image


def preprocess_image(image_path: str | Path, image_size: int = 160) -> np.ndarray:
    """Load and preprocess a face image deterministically."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    # Normalize to [-1, 1]
    arr = (arr - 127.5) / 128.0
    return arr


def get_embedder(model_name: str = "facenet"):
    """Return an embedder callable. Currently supports 'facenet'."""
    if model_name == "facenet":
        return _FaceNetEmbedder()
    raise ValueError(f"Unknown model: {model_name}")


class _FaceNetEmbedder:
    """
    FaceNet-style embedder using facenet-pytorch InceptionResnetV1
    pretrained on VGGFace2.

    Embedding dimensionality: 512
    Input: RGB image resized to 160x160, normalized to [-1, 1]
    """

    def __init__(self):
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1

            self._torch = torch
            self._model = InceptionResnetV1(pretrained="vggface2").eval()
        except ImportError as e:
            raise ImportError(
                "facenet-pytorch and torch are required. "
                "Run: pip install facenet-pytorch torch"
            ) from e

    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        """
        Generate embedding from a preprocessed image array.

        Args:
            image_array: HxWx3 float32 array normalized to [-1, 1]

        Returns:
            1-D float32 numpy array of shape (512,)
        """
        import torch

        # HWC -> CHW -> add batch dim
        tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            embedding = self._model(tensor)
        return embedding.squeeze(0).numpy()


def embed_image(
    image_path: str | Path,
    embedder,
    image_size: int = 160,
) -> tuple[np.ndarray, float]:
    """
    Full preprocessing + embedding for one image.

    Returns:
        (embedding_vector, elapsed_seconds)
    """
    t0 = time.perf_counter()
    arr = preprocess_image(image_path, image_size=image_size)
    emb = embedder(arr)
    elapsed = time.perf_counter() - t0
    return emb, elapsed