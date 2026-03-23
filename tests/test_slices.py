from __future__ import annotations

import pytest

from lfw_verif.slices import build_error_slices


def test_build_error_slices_returns_counts_and_ordered_examples() -> None:
    slices = build_error_slices(
        [
            {"left_path": "a1.png", "right_path": "a2.png", "label": 1, "split": "test", "score": 0.20},
            {"left_path": "b1.png", "right_path": "b2.png", "label": 1, "split": "test", "score": 0.40},
            {"left_path": "c1.png", "right_path": "c2.png", "label": 0, "split": "test", "score": 0.70},
            {"left_path": "d1.png", "right_path": "d2.png", "label": 0, "split": "test", "score": 0.90},
            {"left_path": "e1.png", "right_path": "e2.png", "label": 1, "split": "val", "score": 0.95},
        ],
        threshold=0.5,
        split="test",
        max_examples=2,
    )

    assert slices["same_identity_errors"]["count"] == 2
    assert slices["different_identity_errors"]["count"] == 2
    assert [example["score"] for example in slices["low_similarity_false_negatives"]["examples"]] == [0.2, 0.4]
    assert [example["score"] for example in slices["high_similarity_false_positives"]["examples"]] == [0.9, 0.7]


def test_build_error_slices_returns_empty_payload_for_empty_filtered_split() -> None:
    slices = build_error_slices(
        [
            {"left_path": "a1.png", "right_path": "a2.png", "label": 1, "split": "val", "score": 0.2},
        ],
        threshold=0.5,
        split="test",
        max_examples=1,
    )

    assert slices["same_identity_errors"]["count"] == 0
    assert slices["different_identity_errors"]["examples"] == []


def test_build_error_slices_validates_max_examples() -> None:
    with pytest.raises(ValueError, match="max_examples"):
        build_error_slices(
            [
                {"left_path": "a1.png", "right_path": "a2.png", "label": 1, "split": "test", "score": 0.2},
            ],
            threshold=0.5,
            max_examples=0,
        )
