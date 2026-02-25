# LFW Verification Milestone 1 (v1)

This project implements the Milestone 1 foundation pipeline for face verification:

- deterministic LFW ingestion from a local dataset directory
- deterministic identity-level train/val/test splitting
- deterministic positive/negative pair generation saved to disk
- vectorized cosine similarity and Euclidean distance for batched vectors
- loop vs NumPy benchmark with correctness checks

The focus is reproducibility and performance-aware plumbing for later milestones.

## Repository Layout

- `src/lfw_verif/`: importable package code (dataset, pairs, similarity, utils)
- `scripts/`: CLI entrypoints for ingestion, pair generation, and benchmarking
- `configs/`: pinned Milestone 1 settings (`m1.yaml`)
- `tests/`: unit and determinism tests
- `notebooks/`: optional scratch notebooks
- `outputs/`: generated artifacts
- `data/`: local dataset cache 
## Environment Setup

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## How To Run

Set `--lfw_root` to your local LFW directory.

1. Ingest LFW and create deterministic splits + manifest

```bash
python scripts/ingest_lfw.py --lfw_root /path/to/lfw --out_dir outputs --config configs/m1.yaml
```

2. Generate deterministic verification pairs

```bash
python scripts/make_pairs.py --manifest outputs/lfw_manifest.json --splits outputs/splits.json --out_dir outputs --config configs/m1.yaml
```

3. Run loop-vs-vectorized similarity benchmark

```bash
python scripts/bench_similarity.py --out_dir outputs --config configs/m1.yaml
```

4. Run tests

```bash
pytest -q
```

## Generated Artifacts

- `outputs/lfw_manifest.json`
  - Includes: `seed`, `split_policy`, `data_source`, and per-split counts for `train`, `val`, `test`
- `outputs/splits.json`
  - Deterministic identity split assignments and split counts
- `outputs/pairs/train.csv`, `outputs/pairs/val.csv`, `outputs/pairs/test.csv`
  - Columns: `left_path,right_path,label,split`
- `outputs/pairs_manifest.json`
  - Seed and pair policy summary
- `outputs/bench/benchmark.json`
  - Loop vs vectorized timings, speedup, max absolute error, and tolerance checks

## Determinism Notes

- Seed is fixed in `configs/m1.yaml` (`seed: 1337`).
- Ingestion ordering is deterministic: sorted identity names, then sorted filenames.
- Splits use deterministic hash assignment by identity.
- Pair generation sorts candidates before seeded sampling.

To confirm deterministic outputs, rerun ingestion and pair generation with the same config and compare hashes:

```powershell
Get-FileHash outputs\lfw_manifest.json
Get-FileHash outputs\splits.json
Get-FileHash outputs\pairs\train.csv
Get-FileHash outputs\pairs\val.csv
Get-FileHash outputs\pairs\test.csv
```
