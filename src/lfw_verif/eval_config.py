from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class EvalConfigError(ValueError):
    """Raised when a Milestone 2 evaluation config is invalid."""


@dataclass(frozen=True)
class ThresholdSweep:
    start: float
    stop: float
    step: float


@dataclass(frozen=True)
class EvalConfig:
    experiment_name: str
    feature_extractor: str
    similarity_metric: str
    threshold_sweep: ThresholdSweep
    threshold_selection_rule: str
    threshold_selection_split: str
    final_evaluation_split: str
    tracked_run_dir: Path
    notes: str


_REQUIRED_FIELDS = (
    "experiment_name",
    "feature_extractor",
    "similarity_metric",
    "threshold_sweep",
    "threshold_selection_rule",
    "threshold_selection_split",
    "final_evaluation_split",
    "tracked_run_dir",
    "notes",
)


def load_eval_config(path: str | Path) -> EvalConfig:
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise EvalConfigError(f"Eval config at '{cfg_path}' must contain a top-level mapping.")

    missing = [field for field in _REQUIRED_FIELDS if field not in data]
    if missing:
        missing_str = ", ".join(missing)
        raise EvalConfigError(f"Eval config at '{cfg_path}' is missing required field(s): {missing_str}")

    return EvalConfig(
        experiment_name=_require_non_empty_string(data, "experiment_name"),
        feature_extractor=_require_non_empty_string(data, "feature_extractor"),
        similarity_metric=_require_non_empty_string(data, "similarity_metric"),
        threshold_sweep=_parse_threshold_sweep(data["threshold_sweep"], cfg_path),
        threshold_selection_rule=_require_non_empty_string(data, "threshold_selection_rule"),
        threshold_selection_split=_require_non_empty_string(data, "threshold_selection_split"),
        final_evaluation_split=_require_non_empty_string(data, "final_evaluation_split"),
        tracked_run_dir=Path(_require_non_empty_string(data, "tracked_run_dir")),
        notes=_require_non_empty_string(data, "notes"),
    )


def _require_non_empty_string(data: dict[str, Any], field_name: str) -> str:
    value = data[field_name]
    if not isinstance(value, str) or not value.strip():
        raise EvalConfigError(f"Field '{field_name}' must be a non-empty string.")
    return value.strip()


def _parse_threshold_sweep(value: Any, cfg_path: Path) -> ThresholdSweep:
    if not isinstance(value, dict):
        raise EvalConfigError(f"Field 'threshold_sweep' in '{cfg_path}' must be a mapping with start, stop, and step.")

    required = ("start", "stop", "step")
    missing = [field for field in required if field not in value]
    if missing:
        missing_str = ", ".join(missing)
        raise EvalConfigError(f"Field 'threshold_sweep' is missing required key(s): {missing_str}")

    start = _coerce_float(value["start"], "threshold_sweep.start")
    stop = _coerce_float(value["stop"], "threshold_sweep.stop")
    step = _coerce_float(value["step"], "threshold_sweep.step")

    if step <= 0.0:
        raise EvalConfigError("Field 'threshold_sweep.step' must be greater than 0.")
    if stop < start:
        raise EvalConfigError("Field 'threshold_sweep.stop' must be greater than or equal to 'threshold_sweep.start'.")

    return ThresholdSweep(start=start, stop=stop, step=step)


def _coerce_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvalConfigError(f"Field '{field_name}' must be numeric.")
    return float(value)
