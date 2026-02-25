from __future__ import annotations

import csv
from dataclasses import dataclass
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


def _files_df(manifest: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(manifest["files"])
    return df.sort_values(["person", "relpath"]).reset_index(drop=True)


def _split_df(df: pd.DataFrame, split_people: List[str]) -> pd.DataFrame:
    return df[df["person"].isin(split_people)].reset_index(drop=True)


def generate_pairs_for_split(
    df_split: pd.DataFrame,
    n_pairs: int,
    positive_fraction: float,
    min_images_per_identity: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = df_split.groupby("person")["relpath"].count()
    eligible_people = sorted(counts[counts >= min_images_per_identity].index.to_list())
    if len(eligible_people) < 2:
        raise RuntimeError("Not enough identities for pair generation in this split.")

    person_to_paths = {
        p: sorted(df_split[df_split["person"] == p]["relpath"].to_list()) for p in eligible_people
    }

    n_pos = int(round(n_pairs * positive_fraction))
    n_neg = n_pairs - n_pos

    a_list: List[str] = []
    b_list: List[str] = []
    y_list: List[int] = []

    for _ in range(n_pos):
        p = rng.choice(eligible_people)
        paths = person_to_paths[p]
        i, j = rng.choice(len(paths), size=2, replace=False)
        a_list.append(paths[i])
        b_list.append(paths[j])
        y_list.append(1)

    eligible_people_arr = np.array(eligible_people, dtype=object)
    for _ in range(n_neg):
        p1, p2 = rng.choice(eligible_people_arr, size=2, replace=False)
        a_list.append(rng.choice(person_to_paths[p1]))
        b_list.append(rng.choice(person_to_paths[p2]))
        y_list.append(0)

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
            },
            "files": saved,
        },
    )

    return saved
