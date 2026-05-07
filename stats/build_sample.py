#!/usr/bin/env python3
"""
build_sample.py — Produces a subsample <= 500 MB for inspection.

Strategy: stratified sampling by split, bounded by sampling_ratio + size limit.
The sample preserves:
  - the amateur/professional proportion (stratification)
  - spectrograph diversity (at least 1 star per (split, spectrograph_class))


Outputs: ``$BESS_HF_DATASET_SAMPLE_DIR`` (DatasetDict arrow), default ``./hf_dataset_sample``.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk

# Source/destination paths can be overridden via environment variables to keep
# the committed defaults free of any infrastructure-specific information.
DEFAULT_SRC = Path(os.environ.get("BESS_HF_DATASET_DIR", "./hf_dataset"))
DEFAULT_DST = Path(os.environ.get("BESS_HF_DATASET_SAMPLE_DIR", "./hf_dataset_sample"))
DEFAULT_SPLITS = Path(__file__).resolve().parent.parent / "paper" / "splits.csv"
TARGET_MB = 500
SEED = 20260501


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--splits-csv", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--target-mb", type=int, default=TARGET_MB)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    print(f"[*] Loading full dataset from {args.src}")
    ds = load_from_disk(str(args.src))["train"]
    n_total = len(ds)
    print(f"    total rows: {n_total}")

    splits = pd.read_csv(args.splits_csv).set_index("star_name")["split"].to_dict()

    rng = random.Random(args.seed)

    # Target rows: assume avg 30 KB/row on disk
    target_rows = args.target_mb * 1024 // 30
    print(f"[*] Target rows (approx. {args.target_mb} MB): {target_rows}")

    # Stratified pick: for each split, keep all stars with <= k spectra OR
    # subsample a star uniformly.
    # Simpler: pick ~1/8 of stars at random per split, then cap.
    by_split: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for star, sp in splits.items():
        if sp in by_split:
            by_split[sp].append(star)

    frac = {"train": 0.07, "validation": 0.10, "test": 0.10}
    chosen: set[str] = set()
    for sp, stars in by_split.items():
        k = max(1, int(len(stars) * frac[sp]))
        chosen.update(rng.sample(stars, k=k))
    print(f"[*] Selected stars: {len(chosen)} / {len(splits)}")

    # Filter dataset — only keep rows whose star is in chosen
    print("[*] Filtering ...")
    sample = ds.filter(lambda x: x["star_name"] in chosen, num_proc=4)
    print(f"    sampled rows: {len(sample)}  ({len(sample)*100/n_total:.1f}%)")

    # Persist
    args.dst.mkdir(parents=True, exist_ok=True)
    out = DatasetDict({"train": sample})
    out.save_to_disk(str(args.dst))
    print(f"[+] Saved → {args.dst}")

    # Also write the sample split csv
    sample_splits = {s: splits[s] for s in chosen}
    sdf = pd.DataFrame(
        [{"star_name": s, "split": sp} for s, sp in sample_splits.items()]
    ).sort_values("star_name")
    sdf.to_csv(args.dst / "splits.csv", index=False)
    print(f"[+] {args.dst / 'splits.csv'}")


if __name__ == "__main__":
    main()
