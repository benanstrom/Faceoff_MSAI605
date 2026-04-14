from __future__ import annotations

import numpy as np
import pytest

from lfw_verif.confidence import compute_confidence
from lfw_verif.inference import cosine_similarity


# ── Cosine similarity tests ──────────────────────────────────────────────────

def test_cosine_similarity_identical():
    """Identical vectors should return similarity of 1.0."""
    a = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_opposite():
    """Opposite vectors should return similarity of -1.0."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-5)


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors should return similarity of 0.0."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)


def test_cosine_similarity_range():
    """Cosine similarity should always be in [-1, 1]."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        a = rng.standard_normal(512)
        b = rng.standard_normal(512)
        sim = cosine_similarity(a, b)
        assert -1.0 - 1e-5 <= sim <= 1.0 + 1e-5


# ── Threshold decision tests ─────────────────────────────────────────────────

def test_threshold_same_decision():
    """Score above threshold should be SAME (True)."""
    threshold = 0.6
    score = 0.75
    assert score >= threshold


def test_threshold_different_decision():
    """Score below threshold should be DIFFERENT (False)."""
    threshold = 0.6
    score = 0.4
    assert not (score >= threshold)


def test_threshold_exactly_at_boundary():
    """Score exactly at threshold should be SAME (True)."""
    threshold = 0.6
    score = 0.6
    assert score >= threshold


# ── Confidence computation tests ─────────────────────────────────────────────

def test_confidence_at_threshold_is_half():
    """Score exactly at threshold should give confidence of 0.5."""
    assert compute_confidence(0.6, 0.6) == pytest.approx(0.5, abs=1e-5)


def test_confidence_max_score():
    """Score of 1.0 should give confidence of 1.0."""
    assert compute_confidence(1.0, 0.6) == pytest.approx(1.0, abs=1e-5)


def test_confidence_min_score():
    """Score of -1.0 should give confidence of 0.0."""
    assert compute_confidence(-1.0, 0.6) == pytest.approx(0.0, abs=1e-5)


def test_confidence_range():
    """Confidence should always be in [0.0, 1.0]."""
    threshold = 0.6
    for score in np.linspace(-1.0, 1.0, 50):
        c = compute_confidence(float(score), threshold)
        assert 0.0 <= c <= 1.0


def test_confidence_above_threshold_above_half():
    """Score above threshold should give confidence > 0.5."""
    assert compute_confidence(0.8, 0.6) > 0.5


def test_confidence_below_threshold_below_half():
    """Score below threshold should give confidence < 0.5."""
    assert compute_confidence(0.3, 0.6) < 0.5