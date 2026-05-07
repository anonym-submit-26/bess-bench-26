#!/usr/bin/env python3
"""
02_build_dataset.py — Assembly of the HuggingFace Dataset

Combines the DuckDB metadata (cleaned) with the spectral arrays (Parquet)
to produce a complete HuggingFace Dataset.

No train/test split at this stage: everything goes into the 'train' split,
so end users can choose their own split.

Inputs:
  - harvest/bess_data.duckdb (83 columns, 339k rows)
  - data_pipeline/03_export/parquet_spectra/*.parquet (spectral arrays)

Output:
  - data_pipeline/03_export/hf_dataset/ (HuggingFace DatasetDict saved on disk)

Usage:
    python 02_build_dataset.py [--db PATH] [--parquet-dir PATH] [--output-dir PATH]
"""

import os
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, Features, Sequence, Value

# ─── Default paths ──────────────────────────────────────────────────────────
DEFAULT_DB = Path(__file__).parent.parent.parent / "harvest" / "bess_data.duckdb"
DEFAULT_PARQUET_DIR = Path(os.environ.get("BESS_PARQUET_DIR", "./parquet_spectra"))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("BESS_HF_DATASET_DIR", "./hf_dataset"))

# ─── Constants ───────────────────────────────────────────────────────────────
M_TO_A = 1e10  # meters → Angstroms

# ─── DuckDB columns to extract, with mapping to export names ──
# (duckdb_col, export_col, type)
METADATA_COLUMNS = [
    # Identity
    ("id",                   "spectrum_id",         "string"),
    ("star_name_clean",      "star_name",           "string"),
    ("fits_objname",         "star_name_raw",       "string"),
    ("ra",                   "ra",                  "float64"),
    ("dec",                  "dec",                 "float64"),
    # Spectral
    ("spectral_start",       "lambda_min_m",        "float64"),   # converted to Å
    ("spectral_stop",        "lambda_max_m",        "float64"),   # converted to Å
    ("num_points",           "n_pixels",            "int32"),
    ("fits_bss_itrp",        "spectral_resolution", "float64"),
    ("fits_bss_esrp",        "spectral_resolution_measured", "float64"),
    # Quality
    ("snr_estimated",        "snr",                 "float64"),
    ("snr_continuum",        "snr_continuum",       "float64"),
    # Temporel
    ("observation_date",     "observation_date",    "string"),    # ISO-8601
    ("time_location_mjd",    "mjd",                 "float64"),
    ("fits_exptime",         "exposure_time",       "float64"),
    ("temporal_quality",     "temporal_quality",     "string"),
    # Instrument
    ("spectrograph_clean",   "spectrograph",        "string"),
    ("telescope_clean",      "telescope",           "string"),
    ("detector_clean",       "detector",            "string"),
    ("instrument_setup",     "instrument_setup",    "string"),
    ("instrument_confidence","instrument_confidence","string"),
    ("observer_type",        "observer_type",       "string"),
    # Geographie
    ("site_lat_clean",       "site_latitude",       "float64"),
    ("site_lon_clean",       "site_longitude",      "float64"),
    ("site_elev_clean",      "site_elevation",      "float64"),
    # Corrections
    ("fits_bss_vhel",        "helio_velocity",      "float64"),
    ("fits_bss_rqvh",        "bss_rqvh",            "float64"),  # km/s, residual heliocentric velocity NOT applied in the FITS
    ("fits_bss_tell",        "telluric_corrected",  "string"),
    ("fits_bss_norm",        "normalized",          "string"),
    # Pipeline flags
    ("is_echelle_order",     "is_echelle_order",    "bool"),
]


def load_metadata(db_path: Path) -> pd.DataFrame:
    """Load metadata from DuckDB and rename columns."""
    print("--- Loading DuckDB metadata ---")

    duck_cols = [col[0] for col in METADATA_COLUMNS]
    export_cols = [col[1] for col in METADATA_COLUMNS]

    col_list = ", ".join(duck_cols)
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(f"SELECT {col_list} FROM spectra_votable").fetchdf()
    con.close()

    # Rename
    rename_map = dict(zip(duck_cols, export_cols))
    df.rename(columns=rename_map, inplace=True)

    # Convert wavelengths m → Å
    df["lambda_min"] = df["lambda_min_m"] * M_TO_A
    df["lambda_max"] = df["lambda_max_m"] * M_TO_A
    df.drop(columns=["lambda_min_m", "lambda_max_m"], inplace=True)

    # Convert observation_date to string
    df["observation_date"] = df["observation_date"].astype(str)

    # ── Privacy: rounding coordinates and elevation ──────────────────────
    # BeSS FITS files contain exact coordinates (down to ~10 cm) of the
    # observation site and its altitude to the metre. For an amateur
    # observer (one site = one dome = one house), this precision allows
    # trivial re-identification via Google Maps. Before public export we
    # therefore apply:
    #   - lat/lon rounded to 0.1° (~11 km at the equator). Sufficient for
    #     all statistical analyses (hemispheric bias, time zone, climate)
    #     and for heliocentric corrections used in astronomy:
    #     the induced error on the barycentric velocity is
    #     < 1 m/s, i.e. three orders of magnitude below the typical
    #     BeSS calibration noise (~1-5 km/s).
    #   - elevation rounded to 100 m. Consistent with the lat/lon grid: a
    #     site within a 0.1° cell can span 1000+ m of elevation;
    #     a 100 m grain thus preserves the "lowland vs mountain site"
    #     signature (useful for airmass / extinction analyses)
    #     without revealing the exact valley. Atmospheric pressure varies
    #     by ~1.2% per 100 m, negligible for the anticipated ML uses.
    #
    # Note: the scalar `helio_velocity` (BSS_VHEL) is left unchanged as it
    # was computed by each observer using their exact coordinates before
    # submission — it is a velocity, not a geographic coordinate, and
    # does not allow any re-identification.
    df["site_latitude"] = df["site_latitude"].round(1)
    df["site_longitude"] = df["site_longitude"].round(1)
    df["site_elevation"] = (
        np.round(df["site_elevation"].astype(float) / 100.0) * 100.0
    )

    # Compute SNR quality label
    def snr_quality(val):
        if pd.isna(val): return "unknown"
        if val >= 200: return "excellent"
        if val >= 100: return "good"
        if val >= 50:  return "medium"
        if val >= 10:  return "low"
        return "very_low"

    df["snr_quality"] = df["snr"].apply(snr_quality)

    print(f"  Rows loaded: {len(df):,}")
    print(f"  Columns    : {len(df.columns)}")

    return df


def load_spectral_arrays(parquet_dir: Path) -> pd.DataFrame:
    """Load spectral arrays from the Parquet shards."""
    print("\n--- Loading spectral arrays (Parquet) ---")

    parquet_files = sorted(parquet_dir.glob("spectra_*.parquet"))
    if not parquet_files:
        print(f"  ✗ No Parquet file found in {parquet_dir}")
        sys.exit(1)

    print(f"  Shards found: {len(parquet_files)}")

    tables = []
    for pf in parquet_files:
        tables.append(pq.read_table(pf))

    combined = pa.concat_tables(tables)
    df = combined.to_pandas()

    print(f"  Spectra loaded: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    return df


def create_splits(df: pd.DataFrame, train_ratio: float, val_ratio: float,
                  seed: int) -> dict:
    """
    Split by star (not by spectrum) to avoid data leakage.

    Stars with the most spectra go to train (stability),
    smaller stars are distributed at random.
    """
    print(f"\n--- Creating splits (per star, seed={seed}) ---")
    print(f"  Ratios: train={train_ratio:.0%}, val={val_ratio:.0%}, "
          f"test={1-train_ratio-val_ratio:.0%}")

    rng = np.random.RandomState(seed)

    # Count spectra per star
    star_counts = df["star_name"].value_counts()
    stars = star_counts.index.tolist()
    n_stars = len(stars)

    # Shuffle
    rng.shuffle(stars)

    # Split
    n_train = int(n_stars * train_ratio)
    n_val = int(n_stars * val_ratio)

    train_stars = set(stars[:n_train])
    val_stars = set(stars[n_train:n_train + n_val])
    test_stars = set(stars[n_train + n_val:])

    # Assign
    def get_split(star):
        if star in train_stars: return "train"
        if star in val_stars: return "validation"
        return "test"

    df["split"] = df["star_name"].apply(get_split)

    split_stats = df["split"].value_counts()
    star_splits = df.groupby("split")["star_name"].nunique()

    print(f"\n  {'Split':<12} {'Spectra':>10} {'%':>6} {'Stars':>8}")
    print(f"  {'-'*40}")
    for split in ["train", "validation", "test"]:
        n = split_stats.get(split, 0)
        ns = star_splits.get(split, 0)
        print(f"  {split:<12} {n:>10,} {100*n/len(df):>5.1f}% {ns:>8,}")

    return df


def build_hf_features() -> Features:
    """Defines the schema HuggingFace Features."""
    return Features({
        # Spectral arrays
        "wavelength": Sequence(Value("float32")),
        "flux": Sequence(Value("float32")),
        "flux_error": Sequence(Value("float32")),
        # Identity
        "spectrum_id": Value("string"),
        "star_name": Value("string"),
        "star_name_raw": Value("string"),
        "ra": Value("float64"),
        "dec": Value("float64"),
        # Spectral metadata
        "lambda_min": Value("float64"),
        "lambda_max": Value("float64"),
        "n_pixels": Value("int32"),
        "spectral_resolution": Value("float64"),
        "spectral_resolution_measured": Value("float64"),
        # Quality
        "snr": Value("float64"),
        "snr_continuum": Value("float64"),
        "snr_quality": Value("string"),
        # Temporal
        "observation_date": Value("string"),
        "mjd": Value("float64"),
        "exposure_time": Value("float64"),
        "temporal_quality": Value("string"),
        # Instrument
        "spectrograph": Value("string"),
        "telescope": Value("string"),
        "detector": Value("string"),
        "instrument_setup": Value("string"),
        "instrument_confidence": Value("string"),
        "observer_type": Value("string"),
        # Geography
        "site_latitude": Value("float64"),
        "site_longitude": Value("float64"),
        "site_elevation": Value("float64"),
        # Corrections
        "helio_velocity": Value("float64"),
        "bss_rqvh": Value("float64"),
        "telluric_corrected": Value("string"),
        "normalized": Value("string"),
        # Pipeline flags
        "is_echelle_order": Value("bool"),
        "fits_format": Value("string"),
    })


def main():
    parser = argparse.ArgumentParser(
        description="Assembly of the HuggingFace Dataset")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    # parser.add_argument("--train-ratio", type=float, default=0.70)
    # parser.add_argument("--val-ratio", type=float, default=0.15)
    # parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("HUGGINGFACE DATASET ASSEMBLY")
    print("=" * 70)
    print(f"  Date : {datetime.now().isoformat()}")

    t0 = time.time()

    # ── 1. Load metadata ──
    meta_df = load_metadata(args.db)

    # ── 2. Load spectral arrays ──
    spec_df = load_spectral_arrays(args.parquet_dir)

    # ── 3. Join on 'id' ──
    print(f"\n--- Join metadata + spectral arrays ---")
    merged = meta_df.merge(spec_df, left_on="spectrum_id", right_on="id",
                           how="inner", suffixes=("", "_spec"))
    # Remove the duplicate id column from the parquet
    if "id" in merged.columns:
        merged.drop(columns=["id"], inplace=True)

    n_meta = len(meta_df)
    n_spec = len(spec_df)
    n_merged = len(merged)
    print(f"  Metadata : {n_meta:,}")
    print(f"  Spectral : {n_spec:,}")
    print(f"  Joined   : {n_merged:,}")
    if n_merged < n_spec:
        print(f"  ⚠ {n_spec - n_merged:,} spectra without metadata")
    if n_merged < n_meta:
        print(f"  ⚠ {n_meta - n_merged:,} metadata without extracted spectrum")

    # Use n_pixels from the Parquet if available
    if "n_pixels_extracted" in merged.columns:
        merged["n_pixels"] = merged["n_pixels_extracted"].fillna(merged["n_pixels"])
        merged.drop(columns=["n_pixels_extracted"], inplace=True)

    # ── 4. No split (full Dataset) ──
    print(f"\n--- No split (full Dataset) ---")
    merged["split"] = "train"  # Everything goes to 'train' by default for HuggingFace
    
    # ── 5. Build the DatasetDict ──
    print(f"\n--- Building the DatasetDict ---")
    features = build_hf_features()

    # Final columns in feature order
    final_cols = list(features.keys()) + ["split"]

    # Make sure all columns exist
    for col in final_cols:
        if col not in merged.columns:
            if col == "split":
                continue
            print(f"  ⚠ Missing column: {col} → filled with None")
            merged[col] = None

    # Replace NaN by None for string columns
    for col in merged.select_dtypes(include=["object"]).columns:
        merged[col] = merged[col].where(merged[col].notna(), None)

    ds_dict = {}
    # Put everything in a single split named 'train'
    split_name = "train"
    
    # Keep only the feature columns
    feature_cols = [c for c in features.keys() if c in merged.columns]
    final_df = merged[feature_cols].reset_index(drop=True)

    ds = Dataset.from_pandas(final_df, features=Features({k:features[k] for k in feature_cols}), preserve_index=False)
    ds_dict[split_name] = ds
    print(f"  {split_name:<12} : {len(ds):,} spectra")

    dataset = DatasetDict(ds_dict)

    # ── 6. Save to disk ──
    print(f"\n--- Saving to disk ---")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))

    elapsed = time.time() - t0

    # ── 7. Report ──
    total_size = sum(f.stat().st_size for f in args.output_dir.rglob("*") if f.is_file())

    print(f"\n{'=' * 70}")
    print(f"HUGGINGFACE DATASET BUILT")
    print(f"{'=' * 70}")
    print(f"  Output  : {args.output_dir}")
    print(f"  Size    : {total_size / 1e9:.2f} GB")
    print(f"  Duration: {elapsed / 60:.1f} min")
    print(f"  Splits  : {', '.join(f'{k}: {len(v):,}' for k, v in dataset.items())}")
    print(f"\n  ✓ Ready for push_to_hub (03_push_to_hub.py)")


if __name__ == "__main__":
    main()
