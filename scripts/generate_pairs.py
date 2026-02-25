from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lfw_verif.pairs import PairConfig, generate_and_save_pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic verification pair generation (saved CSV files).")
    ap.add_argument("--manifest", required=True, help="Path to outputs/lfw_manifest.json")
    ap.add_argument("--splits", required=True, help="Path to outputs/splits.json")
    ap.add_argument("--out_dir", default="outputs", help="Output directory for pair files.")
    ap.add_argument("--config", default="configs/pairs.yaml", help="YAML config for pair generation.")
    args = ap.parse_args()

    cfg_y = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pair_cfg = cfg_y.get("pairs", cfg_y)
    cfg = PairConfig(
        seed=int(cfg_y.get("seed", pair_cfg["seed"])),
        pairs_per_split={k: int(v) for k, v in pair_cfg["pairs_per_split"].items()},
        positive_fraction=float(pair_cfg["positive_fraction"]),
        min_images_per_identity=int(pair_cfg.get("min_images_per_identity", 2)),
    )

    saved = generate_and_save_pairs(
        manifest_path=args.manifest, splits_path=args.splits, out_dir=args.out_dir, cfg=cfg
    )

    for split, path in saved.items():
        print(f"Saved {split} pairs -> {path}")


if __name__ == "__main__":
    main()
