from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lfw_verif.confidence import compute_confidence
from lfw_verif.inference import cosine_similarity, run_inference


# ── Smoke test: full inference pipeline with synthetic data ──────────────────

class _DummyEmbedder:
    """Deterministic dummy embedder for smoke testing without real images."""

    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal(512).astype(np.float32)


def test_inference_pipeline_runs(tmp_path):
    """
    Smoke test: confirm the full inference pipeline completes without error
    using synthetic (dummy) images and a dummy embedder.
    """
    from PIL import Image
    import tempfile, os

    # Create two dummy RGB images
    tmp_dir = Path(tempfile.mkdtemp())
    img_a = tmp_dir / "face_a.jpg"
    img_b = tmp_dir / "face_b.jpg"
    Image.fromarray(
        np.full((160, 160, 3), 128, dtype=np.uint8)
    ).save(img_a)
    Image.fromarray(
        np.full((160, 160, 3), 200, dtype=np.uint8)
    ).save(img_b)

    threshold = 0.6
    embedder = _DummyEmbedder()

    result = run_inference(
        img_a,
        img_b,
        threshold=threshold,
        embedder=embedder,
        image_size=160,
    )

    # Check all required output keys are present
    required_keys = [
        "image_a", "image_b", "score", "threshold",
        "decision", "decision_label", "confidence", "latency_seconds",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    # Check types and ranges
    assert isinstance(result["score"], float)
    assert -1.0 <= result["score"] <= 1.0
    assert result["threshold"] == threshold
    assert isinstance(result["decision"], bool)
    assert result["decision_label"] in ("SAME", "DIFFERENT")
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["latency_seconds"] >= 0.0


def test_confidence_and_decision_consistent():
    """
    Smoke test: confidence above 0.5 should match SAME decision
    and confidence below 0.5 should match DIFFERENT decision.
    """
    threshold = 0.6

    score_same = 0.8
    conf_same = compute_confidence(score_same, threshold)
    assert conf_same > 0.5
    assert score_same >= threshold

    score_diff = 0.3
    conf_diff = compute_confidence(score_diff, threshold)
    assert conf_diff < 0.5
    assert score_diff < threshold


def test_cli_help_runs():
    """
    Smoke test: confirm the CLI script runs and returns help without error.
    """
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts" / "infer_pairs.py"
    result = subprocess.run(
        [sys.executable, str(scripts_dir), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "inference" in result.stdout.lower() or "image" in result.stdout.lower()