from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lfw_verif.eval_config import EvalConfig


def make_run_id(experiment_name: str, timestamp: datetime | None = None) -> str:
    run_timestamp = _normalize_timestamp(timestamp)
    return f"{_slugify(experiment_name)}_{run_timestamp.strftime('%Y%m%dT%H%M%S%fZ')}"


def prepare_run_dir(tracked_run_dir: str | Path, run_id: str) -> Path:
    run_dir = Path(tracked_run_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_manifest(
    run_dir: str | Path,
    *,
    run_id: str,
    timestamp: datetime,
    config: EvalConfig,
    pair_csv: str | Path,
    threshold: float,
    metrics: dict[str, Any],
    artifact_paths: dict[str, str | Path],
    notes: str,
) -> Path:
    manifest_path = Path(run_dir) / "run.json"
    payload = {
        "run_id": run_id,
        "timestamp": _normalize_timestamp(timestamp).isoformat().replace("+00:00", "Z"),
        "code_version": _build_code_version(),
        "config": _json_ready(config),
        "data_version": _build_data_version(pair_csv),
        "threshold": float(threshold),
        "metrics": _json_ready(metrics),
        "artifact_paths": _json_ready(artifact_paths),
        "notes": notes,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def run_tracked_evaluation(
    *,
    pair_csv: str | Path,
    config: EvalConfig,
    image_size: tuple[int, int],
    evaluation_runner: Any,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    run_timestamp = _normalize_timestamp(timestamp)
    run_id = make_run_id(config.experiment_name, run_timestamp)
    run_dir = prepare_run_dir(config.tracked_run_dir, run_id)
    evaluation = evaluation_runner(
        pair_csv=pair_csv,
        config=config,
        output_dir=run_dir,
        image_size=image_size,
    )
    manifest_path = write_run_manifest(
        run_dir,
        run_id=run_id,
        timestamp=run_timestamp,
        config=config,
        pair_csv=pair_csv,
        threshold=float(evaluation["selected_threshold"]),
        metrics={
            "selection_metrics": evaluation["selection_metrics"],
            "final_metrics": evaluation["final_metrics"],
        },
        artifact_paths={
            "scores_json": evaluation["scores_json"],
            "metrics_json": evaluation["metrics_json"],
            "threshold_sweep_json": evaluation["threshold_sweep_json"],
            "roc_png": evaluation["roc_png"],
            "confusion_matrix_png": evaluation["confusion_matrix_png"],
        },
        notes=config.notes,
    )
    return {
        "run_id": run_id,
        "timestamp": run_timestamp,
        "run_dir": run_dir,
        "run_manifest": manifest_path,
        **evaluation,
    }


def _normalize_timestamp(timestamp: datetime | None) -> datetime:
    if timestamp is None:
        return datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _slugify(value: str) -> str:
    collapsed = "_".join(value.strip().lower().split())
    sanitized = "".join(character if character.isalnum() or character == "_" else "_" for character in collapsed)
    slug = sanitized.strip("_")
    return slug or "run"


def _build_code_version() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    commit_hash = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        commit_hash = result.stdout.strip()
    except Exception:
        commit_hash = None
    return {
        "git_commit_hash": commit_hash,
    }


def _build_data_version(pair_csv: str | Path) -> dict[str, Any]:
    pair_csv_path = Path(pair_csv)
    split_counts: dict[str, int] = {}
    row_count = 0
    with pair_csv_path.open("rb") as handle:
        pair_csv_sha256 = hashlib.sha256(handle.read()).hexdigest()

    referenced_image_paths: list[Path] = []
    with pair_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = ",".join(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            split_name = str(row.get("split", "")).strip()
            if split_name:
                split_counts[split_name] = split_counts.get(split_name, 0) + 1
            for key in ("left_path", "right_path"):
                raw_path = str(row.get(key, "")).strip()
                if not raw_path:
                    continue
                raw_image_path = Path(raw_path)
                resolved_image_path = raw_image_path if raw_image_path.is_absolute() else pair_csv_path.parent / raw_image_path
                referenced_image_paths.append(resolved_image_path)

    dataset_hasher = hashlib.sha256()
    dataset_hasher.update(pair_csv_sha256.encode("utf-8"))
    for image_path in sorted({path.resolve() for path in referenced_image_paths}):
        dataset_hasher.update(str(image_path).encode("utf-8"))
        with image_path.open("rb") as handle:
            dataset_hasher.update(handle.read())
    return {
        "pair_csv_path": str(pair_csv_path),
        "pair_csv_sha256": pair_csv_sha256,
        "referenced_image_count": len({str(path.resolve()) for path in referenced_image_paths}),
        "referenced_dataset_sha256": dataset_hasher.hexdigest(),
        "pair_row_count": row_count,
        "pair_csv_header": header,
        "split_counts": split_counts,
    }


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value
