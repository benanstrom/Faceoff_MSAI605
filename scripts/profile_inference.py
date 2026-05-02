from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfw_verif.embeddings import get_embedder, preprocess_image
from lfw_verif.inference import cosine_similarity


def time_stage(fn, *args, repeats=10):
    """Run fn(*args) `repeats` times and return mean, std, all timings."""
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        timings.append(time.perf_counter() - t0)
    return result, np.mean(timings), np.std(timings), timings


def profile_single_pair(image_a, image_b, embedder, image_size=160, repeats=10):
    """Profile preprocessing, embedding, and scoring stages separately."""

    # Stage 1: Preprocessing
    _, pre_mean, pre_std, pre_times = time_stage(
        preprocess_image, image_a, image_size, repeats=repeats
    )
    arr_a = preprocess_image(image_a, image_size)
    arr_b = preprocess_image(image_b, image_size)

    # Stage 2: Embedding
    _, emb_mean, emb_std, emb_times = time_stage(
        embedder, arr_a, repeats=repeats
    )
    emb_a = embedder(arr_a)
    emb_b = embedder(arr_b)

    # Stage 3: Scoring
    _, score_mean, score_std, score_times = time_stage(
        cosine_similarity, emb_a, emb_b, repeats=repeats
    )

    end_to_end = pre_mean + emb_mean * 2 + score_mean

    return {
        "repeats": repeats,
        "preprocessing_mean_s": round(float(pre_mean), 6),
        "preprocessing_std_s": round(float(pre_std), 6),
        "embedding_mean_s": round(float(emb_mean), 6),
        "embedding_std_s": round(float(emb_std), 6),
        "scoring_mean_s": round(float(score_mean), 6),
        "scoring_std_s": round(float(score_std), 6),
        "end_to_end_estimate_s": round(float(end_to_end), 6),
        "preprocessing_pct": round(float(pre_mean / end_to_end * 100), 2),
        "embedding_pct": round(float(emb_mean * 2 / end_to_end * 100), 2),
        "scoring_pct": round(float(score_mean / end_to_end * 100), 2),
    }


def profile_batch_sizes(pairs, embedder, image_size=160, batch_sizes=None):
    """Profile latency and throughput across different batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]

    results = []
    for bs in batch_sizes:
        batch = pairs[:bs]
        if len(batch) < bs:
            batch = [pairs[i % len(pairs)] for i in range(bs)]

        t0 = time.perf_counter()
        for row in batch:
            arr_a = preprocess_image(row["image_a"], image_size)
            arr_b = preprocess_image(row["image_b"], image_size)
            emb_a = embedder(arr_a)
            emb_b = embedder(arr_b)
            cosine_similarity(emb_a, emb_b)
        elapsed = time.perf_counter() - t0

        results.append({
            "batch_size": bs,
            "total_time_s": round(elapsed, 4),
            "avg_latency_s": round(elapsed / bs, 4),
            "throughput_rps": round(bs / elapsed, 4),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Milestone 4 profiling: stage-wise latency and batch sensitivity"
    )
    parser.add_argument(
        "--pairs-csv",
        type=str,
        default="artifacts/real_eval/m3_sample_pairs.csv",
    )
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/profiling_results.json",
    )
    args = parser.parse_args()

    import platform
    hw_info = {
        "processor": platform.processor(),
        "system": platform.system(),
        "version": platform.version(),
        "python": platform.python_version(),
    }

    print("Loading embedder...")
    embedder = get_embedder("facenet")

    with open(args.pairs_csv, newline="") as f:
        pairs = list(csv.DictReader(f))

    # Use first pair for stage profiling
    first = pairs[0]
    print(f"Profiling stages on: {Path(first['image_a']).name} vs {Path(first['image_b']).name}")
    print(f"Repeats per stage: {args.repeats}")

    stage_profile = profile_single_pair(
        first["image_a"],
        first["image_b"],
        embedder,
        image_size=args.image_size,
        repeats=args.repeats,
    )

    print("\n=== Stage Latency ===")
    print(f"  Preprocessing : {stage_profile['preprocessing_mean_s']*1000:.3f}ms ({stage_profile['preprocessing_pct']}%)")
    print(f"  Embedding (x2): {stage_profile['embedding_mean_s']*1000:.3f}ms ({stage_profile['embedding_pct']}%)")
    print(f"  Scoring       : {stage_profile['scoring_mean_s']*1000:.6f}ms ({stage_profile['scoring_pct']}%)")
    print(f"  End-to-end est: {stage_profile['end_to_end_estimate_s']*1000:.3f}ms")

    print("\nProfiling batch sizes...")
    batch_results = profile_batch_sizes(
        pairs,
        embedder,
        image_size=args.image_size,
        batch_sizes=[1, 2, 4, 8, 16],
    )

    print("\n=== Batch Size Sensitivity ===")
    print(f"  {'Batch':>6}  {'Avg Latency':>12}  {'Throughput':>12}")
    for r in batch_results:
        print(f"  {r['batch_size']:>6}  {r['avg_latency_s']*1000:>10.1f}ms  {r['throughput_rps']:>10.2f} rps")

    summary = {
        "hardware": hw_info,
        "stage_profile": stage_profile,
        "batch_sensitivity": batch_results,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()