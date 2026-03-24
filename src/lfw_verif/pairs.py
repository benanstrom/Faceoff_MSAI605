from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import read_json, write_json


@dataclass(frozen=True)
class PairConfig:
    seed: int
    pairs_per_split: Dict[str, int]
    positive_fraction: float = 0.5
    min_images_per_identity: int = 2
    positive_pair_strategy: str = "sample_with_replacement"


def _files_df(manifest: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(manifest["files"])
    # Pair sampling depends on row order, so normalize it once up front.
    return df.sort_values(["person", "relpath"]).reset_index(drop=True)


def _split_df(df: pd.DataFrame, split_people: List[str]) -> pd.DataFrame:
    return df[df["person"].isin(split_people)].reset_index(drop=True)


def generate_pairs_for_split(
    df_split: pd.DataFrame,
    n_pairs: int,
    positive_fraction: float,
    min_images_per_identity: int,
    positive_pair_strategy: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = df_split.groupby("person")["relpath"].count()
    eligible_people = sorted(counts[counts >= min_images_per_identity].index.to_list())
    if len(eligible_people) < 2:
        raise RuntimeError("Not enough identities for pair generation in this split.")

    # Precompute sorted per-person image lists so positive and negative sampling stay deterministic.
    person_to_paths = {
        p: sorted(df_split[df_split["person"] == p]["relpath"].to_list()) for p in eligible_people
    }

    n_pos = int(round(n_pairs * positive_fraction))
    n_neg = n_pairs - n_pos

    positive_pairs = _sample_positive_pairs(
        person_to_paths=person_to_paths,
        n_pos=n_pos,
        strategy=positive_pair_strategy,
        rng=rng,
    )
    a_list = [left_path for left_path, _ in positive_pairs]
    b_list = [right_path for _, right_path in positive_pairs]
    y_list = [1] * len(positive_pairs)

    eligible_people_arr = np.array(eligible_people, dtype=object)
    for _ in range(n_neg):
        p1, p2 = rng.choice(eligible_people_arr, size=2, replace=False)
        a_list.append(rng.choice(person_to_paths[p1]))
        b_list.append(rng.choice(person_to_paths[p2]))
        y_list.append(0)

    # Shuffle after class construction so CSV rows are mixed without changing the class balance.
    idx = np.arange(n_pairs)
    rng.shuffle(idx)
    a = np.array(a_list, dtype=object)[idx]
    b = np.array(b_list, dtype=object)[idx]
    y = np.array(y_list, dtype=np.int8)[idx]
    return a, b, y


def generate_and_save_pairs(
    manifest_path: str | Path,
    splits_path: str | Path,
    out_dir: str | Path,
    cfg: PairConfig,
) -> Dict[str, str]:
    manifest = read_json(manifest_path)
    splits_obj = read_json(splits_path)
    splits = splits_obj["splits"]

    df = _files_df(manifest)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    saved: Dict[str, str] = {}
    for split_name in ["train", "val", "test"]:
        n_pairs = int(cfg.pairs_per_split.get(split_name, 0))
        if n_pairs <= 0:
            continue
        people = splits[split_name]
        df_split = _split_df(df, people)
        a, b, y = generate_pairs_for_split(
            df_split=df_split,
            n_pairs=n_pairs,
            positive_fraction=float(cfg.positive_fraction),
            min_images_per_identity=int(cfg.min_images_per_identity),
            positive_pair_strategy=str(cfg.positive_pair_strategy),
            rng=rng,
        )
        out_path = pairs_dir / f"{split_name}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["left_path", "right_path", "label", "split"])
            writer.writeheader()
            for left_path, right_path, label in zip(a.tolist(), b.tolist(), y.tolist()):
                writer.writerow(
                    {
                        "left_path": left_path,
                        "right_path": right_path,
                        "label": int(label),
                        "split": split_name,
                    }
                )
        saved[split_name] = str(out_path)

    write_json(
        out_dir / "pairs_manifest.json",
        {
            "seed": int(cfg.seed),
            "pair_policy": {
                "pairs_per_split": cfg.pairs_per_split,
                "positive_fraction": float(cfg.positive_fraction),
                "min_images_per_identity": int(cfg.min_images_per_identity),
                "positive_pair_strategy": str(cfg.positive_pair_strategy),
            },
            "files": saved,
        },
    )

    return saved


def _sample_positive_pairs(
    *,
    person_to_paths: Dict[str, List[str]],
    n_pos: int,
    strategy: str,
    rng: np.random.Generator,
) -> List[Tuple[str, str]]:
    if n_pos == 0:
        return []
    if strategy == "sample_with_replacement":
        return _sample_positive_pairs_with_replacement(person_to_paths=person_to_paths, n_pos=n_pos, rng=rng)
    if strategy == "prefer_unique":
        return _sample_positive_pairs_prefer_unique(person_to_paths=person_to_paths, n_pos=n_pos, rng=rng)
    raise ValueError(
        f"Unsupported positive_pair_strategy '{strategy}'. Expected 'sample_with_replacement' or 'prefer_unique'."
    )


def _sample_positive_pairs_with_replacement(
    *,
    person_to_paths: Dict[str, List[str]],
    n_pos: int,
    rng: np.random.Generator,
) -> List[Tuple[str, str]]:
    eligible_people = sorted(person_to_paths)
    sampled: List[Tuple[str, str]] = []
    for _ in range(n_pos):
        person = str(rng.choice(eligible_people))
        paths = person_to_paths[person]
        pair_indices = rng.choice(len(paths), size=2, replace=False)
        left_path, right_path = sorted((paths[int(pair_indices[0])], paths[int(pair_indices[1])]))
        sampled.append((left_path, right_path))
    return sampled


def _sample_positive_pairs_prefer_unique(
    *,
    person_to_paths: Dict[str, List[str]],
    n_pos: int,
    rng: np.random.Generator,
) -> List[Tuple[str, str]]:
    all_unique_pairs = _all_unique_positive_pairs(person_to_paths)
    if not all_unique_pairs:
        return []

    if n_pos <= len(all_unique_pairs):
        sampled_indices = rng.choice(len(all_unique_pairs), size=n_pos, replace=False)
        return [all_unique_pairs[int(index)] for index in sampled_indices.tolist()]

    sampled = list(all_unique_pairs)
    remaining = n_pos - len(all_unique_pairs)
    extra_indices = rng.choice(len(all_unique_pairs), size=remaining, replace=True)
    sampled.extend(all_unique_pairs[int(index)] for index in extra_indices.tolist())
    return sampled


def _all_unique_positive_pairs(person_to_paths: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for person in sorted(person_to_paths):
        paths = sorted(person_to_paths[person])
        for left_path, right_path in combinations(paths, 2):
            pairs.append((left_path, right_path))
    return pairs
