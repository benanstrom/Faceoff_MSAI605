from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


_REQUIRED_SCORE_COLUMNS = ("left_path", "right_path", "label", "split", "score")


def build_error_slices(
    scores: pd.DataFrame | list[dict[str, Any]],
    *,
    threshold: float,
    split: str | None = None,
    max_examples: int = 5,
) -> dict[str, dict[str, Any]]:
    scores_df = _coerce_scores_df(scores)
    if split is not None:
        scores_df = scores_df.loc[scores_df["split"] == split].reset_index(drop=True)

    if scores_df.empty:
        return _empty_slice_payload(max_examples=max_examples)

    _validate_max_examples(max_examples)

    scored = scores_df.assign(predicted_label=(scores_df["score"] >= float(threshold)).astype(int))

    false_negatives = scored.loc[(scored["label"] == 1) & (scored["predicted_label"] == 0)].copy()
    false_positives = scored.loc[(scored["label"] == 0) & (scored["predicted_label"] == 1)].copy()

    return {
        "same_identity_errors": _slice_payload(false_negatives, max_examples=max_examples, ascending=True),
        "different_identity_errors": _slice_payload(false_positives, max_examples=max_examples, ascending=False),
        "low_similarity_false_negatives": _slice_payload(false_negatives, max_examples=max_examples, ascending=True),
        "high_similarity_false_positives": _slice_payload(false_positives, max_examples=max_examples, ascending=False),
    }


def resolve_example_image_path(raw_path: str, *, pair_csv_path: str | Path | None = None) -> Path:
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path
    if image_path.exists():
        return image_path
    if pair_csv_path is not None:
        candidate = Path(pair_csv_path).parent / image_path
        if candidate.exists():
            return candidate
    return image_path


def _coerce_scores_df(scores: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    scores_df = scores.copy() if isinstance(scores, pd.DataFrame) else pd.DataFrame(scores)
    missing = [column for column in _REQUIRED_SCORE_COLUMNS if column not in scores_df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Scored pairs are missing required column(s): {missing_str}")

    coerced = scores_df.loc[:, list(_REQUIRED_SCORE_COLUMNS)].copy()
    coerced = coerced.assign(
        label=coerced["label"].astype(int),
        score=coerced["score"].astype(float),
    )
    return coerced


def _validate_max_examples(max_examples: int) -> None:
    if isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples <= 0:
        raise ValueError("max_examples must be a positive integer.")


def _slice_payload(slice_df: pd.DataFrame, *, max_examples: int, ascending: bool) -> dict[str, Any]:
    ordered = slice_df.sort_values(by=["score", "left_path", "right_path"], ascending=[ascending, True, True])
    examples = []
    for row in ordered.head(max_examples).to_dict(orient="records"):
        examples.append(
            {
                "left_path": row["left_path"],
                "right_path": row["right_path"],
                "label": int(row["label"]),
                "split": row["split"],
                "score": float(row["score"]),
                "predicted_label": int(row["predicted_label"]),
            }
        )
    return {
        "count": int(len(slice_df)),
        "examples": examples,
    }


def _empty_slice_payload(*, max_examples: int) -> dict[str, dict[str, Any]]:
    _validate_max_examples(max_examples)
    empty = {"count": 0, "examples": []}
    return {
        "same_identity_errors": dict(empty),
        "different_identity_errors": dict(empty),
        "low_similarity_false_negatives": dict(empty),
        "high_similarity_false_positives": dict(empty),
    }
