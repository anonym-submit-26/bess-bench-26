"""Extract per-spectrum metadata for paper figures and cache as parquet.

Reads the local snapshot at ``$BESS_HF_DATASET_DIR`` (default ``./hf_dataset_one``).
(which has the full 34-column schema of the HF release) and writes a
single parquet with the slim subset used by Figures 1 and 2 of the
paper:

  spectrum_id, star_name, mjd, snr, snr_continuum, spectral_resolution,
  observer_type, observation_date, lambda_min, lambda_max

Output:
  paper/tables/metadata_slim.parquet  (~30 MB)

Runtime: ~1 min (scans the three splits, drops array columns).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parent.parent
SRC = os.environ.get("BESS_HF_DATASET_DIR", "./hf_dataset_one")
OUT = ROOT / "paper" / "tables" / "metadata_slim.parquet"

COLS = [
    "spectrum_id", "star_name", "mjd", "snr", "snr_continuum",
    "spectral_resolution", "observer_type", "observation_date",
    "lambda_min", "lambda_max", "is_echelle_order",
]


def main() -> None:
    ds = load_from_disk(SRC)
    parts = []
    for split_name, split in ds.items():
        print(f"[{split_name}] {len(split)} rows")
        d = split.remove_columns([c for c in split.column_names if c not in COLS])
        df = d.to_pandas()
        df["split"] = split_name
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUT, index=False)
    print(f"[+] {OUT} ({len(full)} rows, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
