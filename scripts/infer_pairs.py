from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

# Allow running as script without installing package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfw_verif.embeddings import get_embedder
from lfw_verif.inference import run_inference


def print_result(result: dict) -> None:
    """Pretty-print a single pair inference result."""
    print("-" * 60)
    print(f"Image A    : {result['image_a']}")
    print(f"Image B    : {result['image_b']}")
    print(f"Score      : {result['score']:.6f}")
    print(f"Threshold  : {result['threshold']}")
    print(f"Decision   : {result['decision_label']}")
    print(f"Confidence : {result['confidence']:.4f}")
    print(f"Latency    : {result['latency_seconds']:.4f}s")
    print("-" * 60)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Milestone 3 CLI: face verification pair inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/m3_inference.yaml",
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--image-a",
        type=str,
        default=None,
        help="Path to first image (single pair mode)",
    )
    parser.add_argument(
        "--image-b",
        type=str,
        default=None,
        help="Path to second image (single pair mode)",
    )
    parser.add_argument(
        "--pairs-csv",
        type=str,
        default=None,
        help="Path to CSV file with columns: image_a, image_b (batch mode)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save results as JSON",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    threshold = config["inference"]["threshold"]
    model_name = config["embedding"]["model_name"]
    image_size = config["embedding"]["image_size"]

    print(f"Loading embedder: {model_name}")
    embedder = get_embedder(model_name)

    results = []

    if args.image_a and args.image_b:
        # Single pair mode
        result = run_inference(
            args.image_a,
            args.image_b,
            threshold=threshold,
            embedder=embedder,
            image_size=image_size,
        )
        print_result(result)
        results.append(result)

    elif args.pairs_csv:
        # Batch mode
        with open(args.pairs_csv, newline="") as f:
            reader = csv.DictReader(f)
            pairs = list(reader)

        print(f"Running inference on {len(pairs)} pairs...")
        for i, row in enumerate(pairs, 1):
            result = run_inference(
                row["image_a"],
                row["image_b"],
                threshold=threshold,
                embedder=embedder,
                image_size=image_size,
            )
            print(f"[{i}/{len(pairs)}]", end=" ")
            print_result(result)
            results.append(result)

    else:
        parser.error("Provide either --image-a and --image-b, or --pairs-csv")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()