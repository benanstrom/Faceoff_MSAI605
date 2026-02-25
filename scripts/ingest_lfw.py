from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lfw_verif.dataset import SplitPolicy, write_manifest_and_splits


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic LFW ingestion + manifest + person-level splits.")
    ap.add_argument("--lfw_root", required=True, help="Path to LFW root folder (person subdirs).")
    ap.add_argument("--out_dir", default="outputs", help="Output directory for manifest/splits.")
    ap.add_argument("--config", default="configs/lfw_ingest.yaml", help="YAML config with seed + split_policy.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ingest_cfg = cfg.get("ingest", cfg)
    seed = int(cfg.get("seed", ingest_cfg["seed"]))
    sp_cfg = ingest_cfg["split_policy"]
    sp = SplitPolicy(
        type=str(sp_cfg["type"]),
        train_frac=float(sp_cfg["train_frac"]),
        val_frac=float(sp_cfg["val_frac"]),
        test_frac=float(sp_cfg["test_frac"]),
        hash=str(sp_cfg.get("hash", "sha256")),
    )

    manifest_path, splits_path = write_manifest_and_splits(
        lfw_root=args.lfw_root, out_dir=args.out_dir, seed=seed, split_policy=sp
    )

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote splits:   {splits_path}")


if __name__ == "__main__":
    main()
