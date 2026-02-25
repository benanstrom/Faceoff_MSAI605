import numpy as np

from lfw_verif.similarity import (
    cosine_similarity_rows,
    cosine_similarity_rows_loop,
    euclidean_distance_rows,
    euclidean_distance_rows_loop,
)


def test_cosine_matches_loop():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1000, 128))
    b = rng.standard_normal((1000, 128))
    v = cosine_similarity_rows(a, b)
    l = cosine_similarity_rows_loop(a, b)
    assert np.max(np.abs(v - l)) < 1e-10


def test_euclidean_matches_loop():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((1000, 64))
    b = rng.standard_normal((1000, 64))
    v = euclidean_distance_rows(a, b)
    l = euclidean_distance_rows_loop(a, b)
    assert np.max(np.abs(v - l)) < 1e-10
