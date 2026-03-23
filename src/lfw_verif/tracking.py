from __future__ import annotations

import json
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
    threshold: float,
    metrics: dict[str, Any],
    artifact_paths: dict[str, str | Path],
    notes: str,
) -> Path:
    manifest_path = Path(run_dir) / "run.json"
    payload = {
        "run_id": run_id,
        "timestamp": _normalize_timestamp(timestamp).isoformat().replace("+00:00", "Z"),
        "config": _json_ready(config),
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
