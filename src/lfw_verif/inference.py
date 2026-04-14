from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from lfw_verif.embeddings import embed_image, get_embedder
from lfw_verif.confidence import compute_confidence


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def run_inference(
    image_path_a: str | Path,
    image_path_b: str | Path,
    threshold: float,
    embedder=None,
    model_name: str = "facenet",
    image_size: int = 160,
) -> dict:
    """
    Full pair-level inference pipeline.

    Stages (separated for profiling in Milestone 4):
        1. Preprocess + Embed image A
        2. Preprocess + Embed image B
        3. Compute cosine similarity score
        4. Apply threshold → binary decision
        5. Compute calibrated confidence
        6. Measure total latency

    Returns a dict with keys:
        image_a, image_b, score, threshold, decision, confidence, latency_seconds
    """
    if embedder is None:
        embedder = get_embedder(model_name)

    t_total_start = time.perf_counter()

    # Stage 1 & 2: preprocess + embed
    emb_a, lat_a = embed_image(image_path_a, embedder, image_size=image_size)
    emb_b, lat_b = embed_image(image_path_b, embedder, image_size=image_size)

    # Stage 3: similarity scoring
    t_score_start = time.perf_counter()
    score = cosine_similarity(emb_a, emb_b)
    t_score_end = time.perf_counter()

    # Stage 4: threshold decision
    decision = bool(score >= threshold)

    # Stage 5: calibrated confidence
    confidence = compute_confidence(score, threshold)

    total_latency = time.perf_counter() - t_total_start

    return {
        "image_a": str(image_path_a),
        "image_b": str(image_path_b),
        "score": round(score, 6),
        "threshold": threshold,
        "decision": decision,
        "decision_label": "SAME" if decision else "DIFFERENT",
        "confidence": round(confidence, 4),
        "latency_seconds": round(total_latency, 4),
        "embed_latency_a": round(lat_a, 4),
        "embed_latency_b": round(lat_b, 4),
        "score_latency": round(t_score_end - t_score_start, 6),
    }