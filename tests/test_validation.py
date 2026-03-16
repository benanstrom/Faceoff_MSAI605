from __future__ import annotations

from pathlib import Path

import pytest

from lfw_verif.validation import ValidationError, load_and_validate_config, validate_pair_csv


def test_validate_pair_csv_accepts_valid_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "\n".join(
            [
                "left_path,right_path,label,split",
                "Alice/0001.jpg,Alice/0002.jpg,1,train",
                "Alice/0001.jpg,Bob/0001.jpg,0,train",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    validated = validate_pair_csv(csv_path, expected_split="train")

    assert validated == csv_path


def test_validate_pair_csv_rejects_invalid_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "\n".join(
            [
                "left_path,right_path,split,label",
                "Alice/0001.jpg,Alice/0002.jpg,train,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="expected \\['left_path', 'right_path', 'label', 'split'\\] in that order"):
        validate_pair_csv(csv_path)


def test_validate_pair_csv_rejects_invalid_label(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "\n".join(
            [
                "left_path,right_path,label,split",
                "Alice/0001.jpg,Bob/0001.jpg,2,train",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="invalid label '2' on line 2"):
        validate_pair_csv(csv_path)


def test_validate_pair_csv_rejects_invalid_split_name(tmp_path: Path) -> None:
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "\n".join(
            [
                "left_path,right_path,label,split",
                "Alice/0001.jpg,Bob/0001.jpg,0,dev",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="must be one of \\['train', 'val', 'test'\\]; got 'dev'"):
        validate_pair_csv(csv_path)


def test_load_and_validate_config_accepts_m1_config() -> None:
    cfg = load_and_validate_config("configs/m1.yaml")

    assert cfg.seed == 1337
    assert cfg.split_policy.type == "person_level_hash"
    assert cfg.pairs.pairs_per_split == {"train": 20000, "val": 5000, "test": 5000}


def test_load_and_validate_config_rejects_bad_split_fractions(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "seed: 1337",
                "ingest:",
                "  split_policy:",
                "    type: person_level_hash",
                "    train_frac: 0.7",
                "    val_frac: 0.2",
                "    test_frac: 0.2",
                "    hash: sha256",
                "pairs:",
                "  pairs_per_split:",
                "    train: 10",
                "    val: 5",
                "    test: 5",
                "  positive_fraction: 0.5",
                "  min_images_per_identity: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="must sum to 1.0"):
        load_and_validate_config(cfg_path)


def test_load_and_validate_config_rejects_bad_pairs_split_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "seed: 1337",
                "ingest:",
                "  split_policy:",
                "    type: person_level_hash",
                "    train_frac: 0.8",
                "    val_frac: 0.1",
                "    test_frac: 0.1",
                "    hash: sha256",
                "pairs:",
                "  pairs_per_split:",
                "    train: 10",
                "    dev: 5",
                "    test: 5",
                "  positive_fraction: 0.5",
                "  min_images_per_identity: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="must contain exactly the split keys \\['train', 'val', 'test'\\]"):
        load_and_validate_config(cfg_path)
