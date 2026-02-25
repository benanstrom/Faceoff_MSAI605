from __future__ import annotations

import json
from pathlib import Path

from lfw_verif.dataset import SplitPolicy, write_manifest_and_splits
from lfw_verif.pairs import PairConfig, generate_and_save_pairs


def _make_fake_lfw(root: Path) -> None:
    for person_idx in range(1, 7):
        person_dir = root / f"person_{person_idx:02d}"
        person_dir.mkdir(parents=True, exist_ok=True)
        for img_idx in range(1, 4):
            (person_dir / f"{person_idx:02d}_{img_idx:02d}.jpg").write_bytes(b"jpg")


def test_ingestion_is_deterministic(tmp_path: Path) -> None:
    lfw_root = tmp_path / "lfw"
    _make_fake_lfw(lfw_root)

    policy = SplitPolicy(type="person_level_hash", train_frac=0.7, val_frac=0.15, test_frac=0.15)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    manifest_a, splits_a = write_manifest_and_splits(lfw_root, out_a, seed=1337, split_policy=policy)
    manifest_b, splits_b = write_manifest_and_splits(lfw_root, out_b, seed=1337, split_policy=policy)

    assert manifest_a.read_text(encoding="utf-8") == manifest_b.read_text(encoding="utf-8")
    assert splits_a.read_text(encoding="utf-8") == splits_b.read_text(encoding="utf-8")


def test_pair_generation_is_deterministic(tmp_path: Path) -> None:
    manifest = {
        "seed": 1337,
        "split_policy": "manual split for test",
        "split_policy_detail": {"type": "manual", "train_frac": 0.34, "val_frac": 0.33, "test_frac": 0.33},
        "counts": {},
        "data_source": {"dataset": "unit-test"},
        "ordering_policy": "sorted",
        "people_counts": {},
        "files": [],
    }
    splits = {"splits": {"train": [], "val": [], "test": []}}

    for split_name, ids in {"train": [1, 2], "val": [3, 4], "test": [5, 6]}.items():
        for pid in ids:
            person = f"person_{pid:02d}"
            splits["splits"][split_name].append(person)
            for img_idx in range(1, 3):
                manifest["files"].append(
                    {"person": person, "relpath": f"{person}/{person}_{img_idx:02d}.jpg", "filename": f"{person}_{img_idx:02d}.jpg"}
                )

    manifest_path = tmp_path / "manifest.json"
    splits_path = tmp_path / "splits.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    splits_path.write_text(json.dumps(splits, sort_keys=True, indent=2), encoding="utf-8")

    cfg = PairConfig(
        seed=1337,
        pairs_per_split={"train": 8, "val": 8, "test": 8},
        positive_fraction=0.5,
        min_images_per_identity=2,
    )

    out_a = tmp_path / "pairs_a"
    out_b = tmp_path / "pairs_b"
    saved_a = generate_and_save_pairs(manifest_path, splits_path, out_a, cfg)
    saved_b = generate_and_save_pairs(manifest_path, splits_path, out_b, cfg)

    for split_name in ["train", "val", "test"]:
        assert Path(saved_a[split_name]).read_text(encoding="utf-8") == Path(saved_b[split_name]).read_text(
            encoding="utf-8"
        )
