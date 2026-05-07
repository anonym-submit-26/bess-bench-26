#!/usr/bin/env python3
"""
generate_star_list.py — Produces star_list.csv + splits.csv from the HF dataset on disk.

Input: ``$BESS_HF_DATASET_DIR`` (DatasetDict, single 'train' split), default ``./hf_dataset``.
Outputs:
  - dataset/star_list.csv        : one row per star (1468) with aggregates
  - dataset/splits.csv           : star_name → split (train/val/test) reproducible

The split is a deterministic MD5 hash of star_name:
  hash % 100 < 70   → train
  70 <= h < 85     → validation
  85 <= h < 100    → test

Rationale: split by star (not by spectrum) to avoid any temporal leakage
in downstream temporal tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

DEFAULT_SRC = Path(os.environ.get("BESS_HF_DATASET_DIR", "./hf_dataset"))
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "paper"


def assign_split(star_name: str, seed: str = "bess_bench_v1") -> str:
    h = hashlib.md5(f"{seed}::{star_name}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading {args.src} ...")
    ds = load_from_disk(str(args.src))["train"]

    cols = [
        "star_name", "star_name_raw", "ra", "dec", "mjd",
        "snr", "snr_quality", "observer_type",
        "is_echelle_order", "lambda_min", "lambda_max",
    ]
    print(f"[*] Collecting {len(ds)} rows on {len(cols)} columns ...")
    df = ds.select_columns(cols).to_pandas()

    # Aggregations per star
    print("[*] Aggregating per star ...")
    agg = df.groupby("star_name").agg(
        n_spectra=("mjd", "size"),
        n_single=("is_echelle_order", lambda s: int((~s).sum())),
        n_echelle_orders=("is_echelle_order", "sum"),
        ra_deg=("ra", "first"),
        dec_deg=("dec", "first"),
        mjd_min=("mjd", "min"),
        mjd_max=("mjd", "max"),
        median_snr=("snr", "median"),
        amateur_frac=(
            "observer_type",
            lambda s: float((s == "amateur").mean()),
        ),
    ).reset_index()

    agg["time_span_days"] = (agg["mjd_max"] - agg["mjd_min"]).round(1)
    agg["split"] = agg["star_name"].apply(assign_split)

    # Write star_list.csv
    star_list_path = args.out / "star_list.csv"
    agg.sort_values("star_name").to_csv(star_list_path, index=False)
    print(f"[+] {star_list_path}  ({len(agg)} stars)")

    # Write splits.csv (minimal)
    splits_path = args.out / "splits.csv"
    agg[["star_name", "split"]].sort_values("star_name").to_csv(
        splits_path, index=False
    )
    print(f"[+] {splits_path}")

    # Summary
    print("\n[=] Split distribution")
    split_summary = agg.groupby("split").agg(
        n_stars=("star_name", "size"),
        n_spectra=("n_spectra", "sum"),
    )
    print(split_summary.to_string())

    n_total = len(agg)
    n_spectra_total = int(agg["n_spectra"].sum())
    print(f"\n[=] Totals: {n_total} stars, {n_spectra_total} spectra")


if __name__ == "__main__":
    main()
