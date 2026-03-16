from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_SPLIT_NAMES = ("train", "val", "test")
PAIR_CSV_COLUMNS = ("left_path", "right_path", "label", "split")


class ValidationError(ValueError):
    """Raised when a pairs file or config file fails validation."""


@dataclass(frozen=True)
class ValidationSplitPolicy:
    type: str
    train_frac: float
    val_frac: float
    test_frac: float
    hash: str


@dataclass(frozen=True)
class ValidationPairConfig:
    seed: int
    pairs_per_split: dict[str, int]
    positive_fraction: float
    min_images_per_identity: int


@dataclass(frozen=True)
class ValidationConfig:
    seed: int
    split_policy: ValidationSplitPolicy
    pairs: ValidationPairConfig


def validate_pair_csv(path: str | Path, *, expected_split: str | None = None) -> Path:
    csv_path = Path(path)
    if not csv_path.exists():
        raise ValidationError(f"Pair CSV does not exist: '{csv_path}'.")

    if expected_split is not None:
        _validate_split_name(expected_split, field_name="expected_split")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValidationError(f"Pair CSV '{csv_path}' is empty or missing a header row.")
        if tuple(fieldnames) != PAIR_CSV_COLUMNS:
            raise ValidationError(
                f"Pair CSV '{csv_path}' has invalid columns {fieldnames}; "
                f"expected {list(PAIR_CSV_COLUMNS)} in that order."
            )

        for line_number, row in enumerate(reader, start=2):
            _require_non_empty_row_value(row, "left_path", csv_path, line_number)
            _require_non_empty_row_value(row, "right_path", csv_path, line_number)

            label_raw = _require_non_empty_row_value(row, "label", csv_path, line_number)
            if label_raw not in {"0", "1"}:
                raise ValidationError(
                    f"Pair CSV '{csv_path}' has invalid label '{label_raw}' on line {line_number}; "
                    "expected one of: 0, 1."
                )

            split_name = _require_non_empty_row_value(row, "split", csv_path, line_number)
            _validate_split_name(split_name, field_name=f"split on line {line_number}", source=csv_path)
            if expected_split is not None and split_name != expected_split:
                raise ValidationError(
                    f"Pair CSV '{csv_path}' has split '{split_name}' on line {line_number}; "
                    f"expected split '{expected_split}'."
                )

    return csv_path


def load_and_validate_config(path: str | Path) -> ValidationConfig:
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValidationError(f"Config '{cfg_path}' must contain a top-level mapping.")

    seed = _require_int(data, "seed", source=cfg_path)
    ingest = _require_mapping(data, "ingest", source=cfg_path)
    split_policy_data = _require_mapping(ingest, "split_policy", source=cfg_path)
    pairs_data = _require_mapping(data, "pairs", source=cfg_path)

    split_policy = ValidationSplitPolicy(
        type=_require_non_empty_string(split_policy_data, "type", source=cfg_path),
        train_frac=_require_float(split_policy_data, "train_frac", source=cfg_path),
        val_frac=_require_float(split_policy_data, "val_frac", source=cfg_path),
        test_frac=_require_float(split_policy_data, "test_frac", source=cfg_path),
        hash=_require_non_empty_string(split_policy_data, "hash", source=cfg_path),
    )
    _validate_split_fractions(split_policy, cfg_path)

    pairs_per_split = _validate_pairs_per_split(
        _require_mapping(pairs_data, "pairs_per_split", source=cfg_path),
        source=cfg_path,
    )
    pair_seed = pairs_data.get("seed", seed)
    if "seed" in pairs_data:
        pair_seed = _require_int(pairs_data, "seed", source=cfg_path)

    positive_fraction = _require_float(pairs_data, "positive_fraction", source=cfg_path)
    if not 0.0 <= positive_fraction <= 1.0:
        raise ValidationError(
            f"Field 'pairs.positive_fraction' in '{cfg_path}' must be between 0.0 and 1.0 inclusive."
        )

    min_images_per_identity = _require_int(pairs_data, "min_images_per_identity", source=cfg_path)
    if min_images_per_identity < 2:
        raise ValidationError(
            f"Field 'pairs.min_images_per_identity' in '{cfg_path}' must be at least 2."
        )

    return ValidationConfig(
        seed=seed,
        split_policy=split_policy,
        pairs=ValidationPairConfig(
            seed=pair_seed,
            pairs_per_split=pairs_per_split,
            positive_fraction=positive_fraction,
            min_images_per_identity=min_images_per_identity,
        ),
    )


def _validate_pairs_per_split(value: dict[str, Any], *, source: Path) -> dict[str, int]:
    actual_keys = tuple(value.keys())
    if actual_keys != VALID_SPLIT_NAMES:
        raise ValidationError(
            f"Field 'pairs.pairs_per_split' in '{source}' must contain exactly the split keys "
            f"{list(VALID_SPLIT_NAMES)} in that order; got {list(actual_keys)}."
        )

    validated: dict[str, int] = {}
    for split_name in VALID_SPLIT_NAMES:
        split_value = value[split_name]
        if isinstance(split_value, bool) or not isinstance(split_value, int):
            raise ValidationError(
                f"Field 'pairs.pairs_per_split.{split_name}' in '{source}' must be an integer."
            )
        if split_value < 0:
            raise ValidationError(
                f"Field 'pairs.pairs_per_split.{split_name}' in '{source}' must be non-negative."
            )
        validated[split_name] = int(split_value)
    return validated


def _validate_split_fractions(split_policy: ValidationSplitPolicy, source: Path) -> None:
    total = split_policy.train_frac + split_policy.val_frac + split_policy.test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValidationError(
            f"Split fractions in '{source}' must sum to 1.0; got {total:.6f}."
        )


def _require_non_empty_row_value(
    row: dict[str, str | None], field_name: str, source: Path, line_number: int
) -> str:
    value = row.get(field_name)
    if value is None or not value.strip():
        raise ValidationError(
            f"Pair CSV '{source}' is missing a non-empty '{field_name}' value on line {line_number}."
        )
    return value.strip()


def _validate_split_name(value: str, *, field_name: str, source: Path | None = None) -> str:
    if value not in VALID_SPLIT_NAMES:
        prefix = f"Field '{field_name}'" if source is None else f"{field_name.capitalize()}"
        suffix = "" if source is None else f" in '{source}'"
        raise ValidationError(
            f"{prefix}{suffix} must be one of {list(VALID_SPLIT_NAMES)}; got '{value}'."
        )
    return value


def _require_mapping(data: dict[str, Any], field_name: str, *, source: Path) -> dict[str, Any]:
    value = data.get(field_name)
    if not isinstance(value, dict):
        raise ValidationError(f"Field '{field_name}' in '{source}' must be a mapping.")
    return value


def _require_non_empty_string(data: dict[str, Any], field_name: str, *, source: Path) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"Field '{field_name}' in '{source}' must be a non-empty string.")
    return value.strip()


def _require_int(data: dict[str, Any], field_name: str, *, source: Path) -> int:
    value = data.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"Field '{field_name}' in '{source}' must be an integer.")
    return int(value)


def _require_float(data: dict[str, Any], field_name: str, *, source: Path) -> float:
    value = data.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"Field '{field_name}' in '{source}' must be numeric.")
    return float(value)
