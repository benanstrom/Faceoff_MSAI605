from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from lfw_verif.similarity import (
    cosine_similarity_rows,
    cosine_similarity_rows_loop,
    euclidean_distance_rows,
    euclidean_distance_rows_loop,
)


def _timeit(fn, *args, repeat: int = 3) -> float:
    fn(*args)  # warmup
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(min(times))


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Python loop vs NumPy vectorization for similarity scoring.")
    ap.add_argument("--out_dir", default="outputs", help="Output directory for benchmark.json")
    ap.add_argument("--config", default="configs/benchmark.yaml", help="Benchmark YAML config.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bench_cfg = cfg.get("benchmark", cfg)
    seed = int(cfg.get("seed", bench_cfg["seed"]))
    n_pairs = int(bench_cfg["n_pairs"])
    d = int(bench_cfg["embedding_dim"])
    tolerance = float(bench_cfg.get("tolerance", 1e-10))

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_pairs, d), dtype=np.float64)
    b = rng.standard_normal((n_pairs, d), dtype=np.float64)

    cos_vec = cosine_similarity_rows(a, b)
    cos_loop = cosine_similarity_rows_loop(a, b)
    euc_vec = euclidean_distance_rows(a, b)
    euc_loop = euclidean_distance_rows_loop(a, b)

    cosine_max_abs_error = float(np.max(np.abs(cos_vec - cos_loop)))
    euclidean_max_abs_error = float(np.max(np.abs(euc_vec - euc_loop)))
    out = {
        "seed": seed,
        "n_pairs": n_pairs,
        "embedding_dim": d,
        "tolerance": tolerance,
        "cosine": {
            "time_loop_s": _timeit(cosine_similarity_rows_loop, a, b),
            "time_vectorized_s": _timeit(cosine_similarity_rows, a, b),
            "max_abs_error_vs_loop": cosine_max_abs_error,
            "within_tolerance": bool(cosine_max_abs_error <= tolerance),
        },
        "euclidean": {
            "time_loop_s": _timeit(euclidean_distance_rows_loop, a, b),
            "time_vectorized_s": _timeit(euclidean_distance_rows, a, b),
            "max_abs_error_vs_loop": euclidean_max_abs_error,
            "within_tolerance": bool(euclidean_max_abs_error <= tolerance),
        },
    }
    out["cosine"]["speedup"] = out["cosine"]["time_loop_s"] / out["cosine"]["time_vectorized_s"]
    out["euclidean"]["speedup"] = out["euclidean"]["time_loop_s"] / out["euclidean"]["time_vectorized_s"]

    out_dir = Path(args.out_dir) / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote benchmark: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
