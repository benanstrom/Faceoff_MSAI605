from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lfw_verif.slices import build_error_slices, resolve_example_image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract error slices from evaluation artifacts.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Tracked run directory containing scores.json and metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for slice metadata and copied examples. Defaults to <run-dir>/error_slices.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split filter for slice extraction.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="Maximum number of example pairs to copy for each slice.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else run_dir / "error_slices"
    artifacts = extract_error_slices(
        run_dir=run_dir,
        output_dir=output_dir,
        split=args.split,
        max_examples=args.max_examples,
    )

    print(f"Wrote slice summary: {artifacts['slices_json']}")
    print(f"Wrote slice examples under: {artifacts['output_dir']}")


def extract_error_slices(
    *,
    run_dir: str | Path,
    output_dir: str | Path,
    split: str | None = None,
    max_examples: int = 3,
) -> dict[str, Path]:
    run_path = Path(run_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scores_payload = json.loads((run_path / "scores.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))

    threshold = float(metrics_payload["selected_threshold"])
    pair_csv_path = Path(scores_payload["pair_csv"])
    slices = build_error_slices(
        scores_payload["scores"],
        threshold=threshold,
        split=split,
        max_examples=max_examples,
    )
    copied_slices = _copy_example_images(
        slices,
        output_dir=output_path,
        pair_csv_path=pair_csv_path,
    )

    slices_json = output_path / "slices.json"
    slices_json.write_text(
        json.dumps(
            {
                "run_dir": str(run_path),
                "threshold": threshold,
                "split": split,
                "slices": copied_slices,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": output_path,
        "slices_json": slices_json,
    }


def _copy_example_images(
    slices: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    pair_csv_path: Path,
) -> dict[str, dict[str, Any]]:
    copied: dict[str, dict[str, Any]] = {}
    for slice_name, payload in slices.items():
        slice_dir = output_dir / slice_name
        slice_dir.mkdir(parents=True, exist_ok=True)

        examples = []
        for index, example in enumerate(payload["examples"], start=1):
            left_source = resolve_example_image_path(example["left_path"], pair_csv_path=pair_csv_path)
            right_source = resolve_example_image_path(example["right_path"], pair_csv_path=pair_csv_path)
            left_target = slice_dir / f"{index:02d}_left{left_source.suffix}"
            right_target = slice_dir / f"{index:02d}_right{right_source.suffix}"
            _copy_if_exists(left_source, left_target)
            _copy_if_exists(right_source, right_target)

            examples.append(
                {
                    **example,
                    "left_example_path": str(left_target),
                    "right_example_path": str(right_target),
                }
            )

        copied[slice_name] = {
            "count": int(payload["count"]),
            "examples": examples,
        }
    return copied


def _copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Could not copy missing example image: '{source}'.")
    shutil.copy2(source, target)


if __name__ == "__main__":
    main()
