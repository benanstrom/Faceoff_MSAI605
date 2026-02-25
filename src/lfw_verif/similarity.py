from __future__ import annotations

import numpy as np


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Cosine similarity for corresponding rows of a and b: (N,D) -> (N,)"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / (den + eps)


def euclidean_distance_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance for corresponding rows of a and b: (N,D) -> (N,)"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    return np.sqrt(np.sum(diff * diff, axis=1))


def cosine_similarity_rows_loop(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    out = np.empty((a.shape[0],), dtype=np.float64)
    for i in range(a.shape[0]):
        num = 0.0
        a_sq = 0.0
        b_sq = 0.0
        for j in range(a.shape[1]):
            ai = float(a[i, j])
            bi = float(b[i, j])
            num += ai * bi
            a_sq += ai * ai
            b_sq += bi * bi
        den = float(np.sqrt(a_sq) * np.sqrt(b_sq))
        out[i] = num / (den + eps)
    return out


def euclidean_distance_rows_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    out = np.empty((a.shape[0],), dtype=np.float64)
    for i in range(a.shape[0]):
        sq = 0.0
        for j in range(a.shape[1]):
            diff = float(a[i, j]) - float(b[i, j])
            sq += diff * diff
        out[i] = float(np.sqrt(sq))
    return out
