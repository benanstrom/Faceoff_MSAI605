from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .utils import write_json, stable_hash_to_unit_interval


@dataclass(frozen=True)
class SplitPolicy:
    type: str
    train_frac: float
    val_frac: float
    test_frac: float
    hash: str = "sha256"


def _iter_lfw_images(lfw_root: Path) -> List[Dict[str, Any]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    records: List[Dict[str, Any]] = []
    for person_dir in sorted([p for p in lfw_root.iterdir() if p.is_dir()]):
        person = person_dir.name
        for img in sorted(person_dir.iterdir()):
            if img.suffix.lower() in exts and img.is_file():
                records.append(
                    {
                        "person": person,
                        "relpath": str(img.relative_to(lfw_root)).replace("\\", "/"),
                        "filename": img.name,
                    }
                )
    return records


def build_manifest(lfw_root: str | Path, seed: int, split_policy: SplitPolicy) -> Dict[str, Any]:
    lfw_root = Path(lfw_root).expanduser().resolve()
    if not lfw_root.exists():
        raise FileNotFoundError(f"lfw_root does not exist: {lfw_root}")

    records = _iter_lfw_images(lfw_root)
    if len(records) == 0:
        raise RuntimeError(f"No images found under: {lfw_root}")

    df = pd.DataFrame(records)
    counts = df.groupby("person")["relpath"].count().sort_values(ascending=False)

    return {
        "seed": int(seed),
        "split_policy": (
            f"{split_policy.type} with fractions "
            f"train={split_policy.train_frac:.2f}, val={split_policy.val_frac:.2f}, "
            f"test={split_policy.test_frac:.2f} using {split_policy.hash}"
        ),
        "split_policy_detail": {
            "type": split_policy.type,
            "train_frac": float(split_policy.train_frac),
            "val_frac": float(split_policy.val_frac),
            "test_frac": float(split_policy.test_frac),
            "hash": split_policy.hash,
        },
        "data_source": {
            "dataset": "LFW (local directory)",
            "lfw_root": str(lfw_root),
            "cache_note": "Using existing local files; no downloader used.",
        },
        "ordering_policy": "Sorted by person name, then sorted by image filename.",
        "counts": {"total": {"n_identities": int(counts.shape[0]), "n_images": int(df.shape[0])}},
        "people_counts": counts.to_dict(),
        "files": df.sort_values(["person", "relpath"]).to_dict(orient="records"),
    }


def make_person_splits(manifest: Dict[str, Any]) -> Dict[str, List[str]]:
    sp = manifest["split_policy_detail"]
    seed = int(manifest["seed"])
    people = sorted(manifest["people_counts"].keys())

    train_frac = float(sp["train_frac"])
    val_frac = float(sp["val_frac"])
    test_frac = float(sp["test_frac"])
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("Split fracs must sum to 1.0")

    algo = sp.get("hash", "sha256")

    train: List[str] = []
    val: List[str] = []
    test: List[str] = []

    for p in people:
        u = stable_hash_to_unit_interval(p, seed=seed, algo=algo)
        if u < train_frac:
            train.append(p)
        elif u < train_frac + val_frac:
            val.append(p)
        else:
            test.append(p)

    return {"train": train, "val": val, "test": test}


def write_manifest_and_splits(
    lfw_root: str | Path,
    out_dir: str | Path,
    seed: int,
    split_policy: SplitPolicy,
) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(lfw_root=lfw_root, seed=seed, split_policy=split_policy)
    splits = make_person_splits(manifest)
    df = pd.DataFrame(manifest["files"])

    split_counts: Dict[str, Dict[str, int]] = {}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["person"].isin(splits[split_name])]
        split_counts[split_name] = {
            "n_identities": int(split_df["person"].nunique()),
            "n_images": int(split_df.shape[0]),
        }
    manifest["counts"].update(split_counts)

    manifest_path = out_dir / "lfw_manifest.json"
    splits_path = out_dir / "splits.json"

    write_json(manifest_path, manifest)
    write_json(
        splits_path,
        {
            "seed": seed,
            "split_policy": manifest["split_policy"],
            "split_policy_detail": manifest["split_policy_detail"],
            "splits": splits,
            "counts": split_counts,
        },
    )
    return manifest_path, splits_path
