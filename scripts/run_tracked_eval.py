from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lfw_verif.eval_config import load_eval_config
from lfw_verif.tracking import run_tracked_evaluation

from evaluate_pairs import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tracked face verification evaluation.")
    parser.add_argument("--pairs", required=True, help="Path to pair CSV with left_path, right_path, label, split.")
    parser.add_argument("--config", default="configs/m2_baseline.yaml", help="Milestone 2 evaluation config.")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(32, 32),
        help="Image size for baseline grayscale feature extraction.",
    )
    args = parser.parse_args()

    config = load_eval_config(args.config)
    tracked = run_tracked_evaluation(
        pair_csv=args.pairs,
        config=config,
        image_size=tuple(args.image_size),
        evaluation_runner=run_evaluation,
    )

    print(f"Run ID: {tracked['run_id']}")
    print(f"Run directory: {tracked['run_dir']}")
    print(f"Run manifest: {tracked['run_manifest']}")
    print(f"Wrote scores: {tracked['scores_json']}")
    print(f"Wrote metrics: {tracked['metrics_json']}")
    print(f"Wrote threshold sweep: {tracked['threshold_sweep_json']}")
    print(f"Wrote ROC plot: {tracked['roc_png']}")
    print(f"Wrote confusion matrix plot: {tracked['confusion_matrix_png']}")


if __name__ == "__main__":
    main()
