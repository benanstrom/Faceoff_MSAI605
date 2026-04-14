from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfw_verif.inference import run_inference


def run_single(args_tuple):
    """Worker function for a single inference request."""
    image_a, image_b, threshold, model_name, image_size = args_tuple
    try:
        result = run_inference(
            image_a,
            image_b,
            threshold=threshold,
            model_name=model_name,
            image_size=image_size,
        )
        return {"status": "ok", "latency": result["latency_seconds"]}
    except Exception as e:
        return {"status": "error", "latency": None, "error": str(e)}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Milestone 3 load test: concurrent face verification inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/m3_inference.yaml",
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--pairs-csv",
        type=str,
        default=None,
        help="Path to pairs CSV (overrides config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of concurrent workers (overrides config)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Total number of requests to run (overrides config)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/load_test_results.json",
        help="Path to save load test results JSON",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    threshold = config["inference"]["threshold"]
    model_name = config["embedding"]["model_name"]
    image_size = config["embedding"]["image_size"]
    pairs_file = args.pairs_csv or config["load_test"]["pairs_file"]
    num_workers = args.num_workers or config["load_test"]["num_workers"]
    num_requests = args.num_requests or config["load_test"]["num_requests"]

    # Load pairs deterministically
    with open(pairs_file, newline="") as f:
        reader = csv.DictReader(f)
        all_pairs = list(reader)

    # Repeat/cycle pairs to fill num_requests
    pairs = [all_pairs[i % len(all_pairs)] for i in range(num_requests)]

    task_args = [
        (row["image_a"], row["image_b"], threshold, model_name, image_size)
        for row in pairs
    ]

    print(f"Load test: {num_requests} requests | {num_workers} workers")
    print(f"Pairs file: {pairs_file}")
    print(f"Model: {model_name} | Threshold: {threshold}")
    print("-" * 60)

    latencies = []
    failures = 0

    wall_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_single, a): i for i, a in enumerate(task_args)}
        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "ok":
                latencies.append(res["latency"])
            else:
                failures += 1
                print(f"  [FAIL] {res.get('error', 'unknown error')}")

    wall_time = time.perf_counter() - wall_start

    latencies_arr = np.array(latencies)
    throughput = len(latencies) / wall_time if wall_time > 0 else 0.0

    summary = {
        "total_requests": num_requests,
        "successful": len(latencies),
        "failures": failures,
        "wall_time_seconds": round(wall_time, 4),
        "throughput_rps": round(throughput, 4),
        "latency_mean": round(float(np.mean(latencies_arr)), 4) if len(latencies) else None,
        "latency_median": round(float(np.median(latencies_arr)), 4) if len(latencies) else None,
        "latency_p95": round(float(np.percentile(latencies_arr, 95)), 4) if len(latencies) else None,
        "latency_min": round(float(np.min(latencies_arr)), 4) if len(latencies) else None,
        "latency_max": round(float(np.max(latencies_arr)), 4) if len(latencies) else None,
    }

    print("\n=== Load Test Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()