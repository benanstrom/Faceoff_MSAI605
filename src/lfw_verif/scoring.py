from __future__ import annotations

from pathlib import Path

import pandas as pd

from lfw_verif.features import extract_baseline_features
from lfw_verif.similarity import cosine_similarity_rows, euclidean_distance_rows


_REQUIRED_PAIR_COLUMNS = ("left_path", "right_path")


def score_pairs(
    pair_csv: str | Path,
    image_size: tuple[int, int] = (32, 32),
    similarity_metric: str = "cosine",
) -> list[float]:
    pair_csv_path = Path(pair_csv)
    pairs_df = pd.read_csv(pair_csv_path)
    _validate_pair_columns(pairs_df.columns, pair_csv_path)

    if pairs_df.empty:
        return []

    left_features = []
    right_features = []
    for row in pairs_df.itertuples(index=False):
        left_path = _resolve_image_path(row.left_path, pair_csv_path)
        right_path = _resolve_image_path(row.right_path, pair_csv_path)
        left_features.append(extract_baseline_features(left_path, size=image_size))
        right_features.append(extract_baseline_features(right_path, size=image_size))

    if similarity_metric == "cosine":
        scores = cosine_similarity_rows(left_features, right_features)
    elif similarity_metric == "euclidean":
        scores = euclidean_distance_rows(left_features, right_features)
    else:
        raise ValueError(
            f"Unsupported similarity metric '{similarity_metric}'. Expected 'cosine' or 'euclidean'."
        )

    return [float(score) for score in scores.tolist()]


def _validate_pair_columns(columns: pd.Index, pair_csv_path: Path) -> None:
    missing = [column for column in _REQUIRED_PAIR_COLUMNS if column not in columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Pair CSV '{pair_csv_path}' is missing required column(s): {missing_str}")


def _resolve_image_path(raw_path: str, pair_csv_path: Path) -> Path:
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path

    if image_path.exists():
        return image_path

    return pair_csv_path.parent / image_path
