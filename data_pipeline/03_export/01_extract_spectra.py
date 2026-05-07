#!/usr/bin/env python3
"""
01_extract_spectra.py — Extract spectral data from FITS files.

Reads each BeSS FITS file and extracts the spectral arrays:
  - wavelength (Å)
  - flux
  - flux_error (if available, otherwise None)

Results are saved as Parquet shards for efficient consumption
by the assembly script (02_build_dataset.py).

Supported FITS formats:
  1. BeSS standard: BinTable extension "SPECTRUM" with columns WAVE, FLUX, ERROR
  2. Generic fallback: first BinTable extension with columns WAVE, FLUX
  3. WCS header: flux in HDU[0].data, calibration via CRVAL1/CDELT1/CRPIX1

Usage:
    python 01_extract_spectra.py [--db PATH] [--fits-dir PATH] [--output-dir PATH]
                                 [--workers N] [--shard-size N] [--limit N]

Approx. duration: ~30-60 min for 339k spectra (depends on workers and I/O).
"""

import os
import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits

# ─── Default paths ──────────────────────────────────────────────────────────
DEFAULT_DB = Path(__file__).parent.parent.parent / "harvest" / "bess_data.duckdb"
DEFAULT_FITS_DIR = Path(os.environ.get("BESS_FITS_DIR", "./spectra_data"))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("BESS_PARQUET_DIR", "./parquet_spectra"))

# ─── Constants ───────────────────────────────────────────────────────────────
M_TO_A = 1e10  # meters → Angstroms (DuckDB metadata is stored in meters)


def extract_one_spectrum(fits_path: str) -> dict:
    """
    Extract spectral arrays from a single FITS file.

    Returns a dict with:
      - id: spectrum identifier
      - wavelength: float32 wavelength array (Å)
      - flux: float32 flux array
      - flux_error: error array (or None)
      - n_pixels: number of points
      - format: detected FITS format ('bintable_spectrum', 'bintable_generic', 'wcs_header')
      - error: error message (or None on success)
    """
    fits_path = Path(fits_path)
    spectrum_id = fits_path.stem.split("_")[0]  # hash MD5

    result = {
        "id": spectrum_id,
        "wavelength": None,
        "flux": None,
        "flux_error": None,
        "n_pixels": 0,
        "format": None,
        "error": None,
    }

    try:
        with fits.open(fits_path, memmap=True) as hdul:
            wave = None
            flux = None
            flux_err = None

            # ── Format 1 : BeSS standard (BinTable "SPECTRUM") ──
            ext_names = [h.name for h in hdul]
            if "SPECTRUM" in ext_names:
                spectrum = hdul["SPECTRUM"].data
                wave = np.array(spectrum["WAVE"], dtype=np.float64)
                flux = np.array(spectrum["FLUX"], dtype=np.float64)
                if "ERROR" in spectrum.names:
                    flux_err = np.array(spectrum["ERROR"], dtype=np.float64)
                result["format"] = "bintable_spectrum"

            # ── Format 2: generic BinTable ──
            elif len(hdul) > 1 and hasattr(hdul[1], "data") and hdul[1].data is not None:
                data = hdul[1].data
                if hasattr(data, "names"):
                    if "WAVE" in data.names and "FLUX" in data.names:
                        wave = np.array(data["WAVE"], dtype=np.float64)
                        flux = np.array(data["FLUX"], dtype=np.float64)
                        if "ERROR" in data.names:
                            flux_err = np.array(data["ERROR"], dtype=np.float64)
                        result["format"] = "bintable_generic"

            # ── Format 3 : WCS header ──
            if wave is None and hdul[0].data is not None:
                header = hdul[0].header
                flux = np.array(hdul[0].data, dtype=np.float64)
                if flux.ndim > 1:
                    flux = flux.flatten()

                crval1 = header.get("CRVAL1", 0)
                cdelt1 = header.get("CDELT1", header.get("CD1_1", 1))
                crpix1 = header.get("CRPIX1", 1)
                wave = crval1 + (np.arange(len(flux)) - crpix1 + 1) * cdelt1
                result["format"] = "wcs_header"

            # ── Validation ──
            if wave is None or flux is None:
                result["error"] = "no_spectral_data"
                return result

            if len(wave) == 0 or len(flux) == 0:
                result["error"] = "empty_arrays"
                return result

            if len(wave) != len(flux):
                result["error"] = f"shape_mismatch: wave={len(wave)}, flux={len(flux)}"
                return result

            # Clean up NaN/Inf
            valid = np.isfinite(wave) & np.isfinite(flux)
            if valid.sum() < 10:
                result["error"] = f"too_few_valid_pixels: {valid.sum()}"
                return result

            wave = wave[valid]
            flux = flux[valid]
            if flux_err is not None:
                flux_err = flux_err[valid] if len(flux_err) == len(valid) else None

            # Convert to float32 to save space
            result["wavelength"] = wave.astype(np.float32).tolist()
            result["flux"] = flux.astype(np.float32).tolist()
            if flux_err is not None and np.any(np.isfinite(flux_err)):
                result["flux_error"] = flux_err.astype(np.float32).tolist()
            result["n_pixels"] = len(wave)

    except Exception as e:
        result["error"] = f"exception: {str(e)[:200]}"

    return result


def process_batch(batch_args: tuple) -> list:
    """
    Process a batch of FITS files.
    Called inside a ProcessPoolExecutor.
    """
    fits_paths, fits_dir = batch_args
    results = []
    for fname in fits_paths:
        fpath = Path(fits_dir) / fname
        results.append(extract_one_spectrum(str(fpath)))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extraction of FITS spectra → Parquet")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="Path to the DuckDB database")
    parser.add_argument("--fits-dir", type=Path, default=DEFAULT_FITS_DIR,
                        help="Directory containing the FITS files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for the Parquet shards")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--shard-size", type=int, default=10000,
                        help="Number of spectra per Parquet shard")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N spectra (0 = all)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Spectra per batch worker")
    args = parser.parse_args()

    print("=" * 70)
    print("SPECTRAL EXTRACTION — FITS → Parquet")
    print("=" * 70)
    print(f"  DuckDB         : {args.db}")
    print(f"  FITS directory : {args.fits_dir}")
    print(f"  Output       : {args.output_dir}")
    print(f"  Workers      : {args.workers}")
    print(f"  Shard size   : {args.shard_size}")
    print(f"  Date         : {datetime.now().isoformat()}")

    # ── Retrieve the file list from DuckDB ──
    con = duckdb.connect(str(args.db), read_only=True)
    query = "SELECT local_filename FROM spectra_votable WHERE local_filename IS NOT NULL"
    if args.limit > 0:
        query += f" LIMIT {args.limit}"
    filenames = [row[0] for row in con.execute(query).fetchall()]
    con.close()

    n_total = len(filenames)
    print(f"\n  Spectra to process: {n_total:,}")

    # Check that the FITS directory exists
    if not args.fits_dir.exists():
        print(f"\n  ✗ FITS directory not found : {args.fits_dir}")
        sys.exit(1)

    # Create the output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Prepare the batches ──
    batches = []
    for i in range(0, n_total, args.batch_size):
        batch_fnames = filenames[i : i + args.batch_size]
        batches.append((batch_fnames, str(args.fits_dir)))

    print(f"  Batches      : {len(batches)}")

    # ── Parallel extraction ──
    t0 = time.time()
    all_results = []
    n_done = 0
    n_errors = 0

    shard_idx = 0
    shard_buffer = []

    print(f"\n--- Extraction in progress ---")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_batch, batch): i
                   for i, batch in enumerate(batches)}

        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)

            for r in batch_results:
                if r["error"]:
                    n_errors += 1
                else:
                    shard_buffer.append(r)

                n_done += 1

            # Write a shard when the buffer is full
            while len(shard_buffer) >= args.shard_size:
                _write_shard(shard_buffer[:args.shard_size],
                             args.output_dir, shard_idx)
                shard_buffer = shard_buffer[args.shard_size:]
                shard_idx += 1

            # Progress
            if n_done % 10000 < args.batch_size:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_total - n_done) / rate if rate > 0 else 0
                print(f"  {n_done:>8,} / {n_total:,} "
                      f"({100 * n_done / n_total:.1f}%) "
                      f"| {rate:.0f} spectra/s "
                      f"| ETA {eta / 60:.1f} min "
                      f"| errors {n_errors:,}")

    # Write the last shard
    if shard_buffer:
        _write_shard(shard_buffer, args.output_dir, shard_idx)
        shard_idx += 1

    elapsed = time.time() - t0

    # ── Error report ──
    errors = [r for r in all_results if r["error"]]
    error_types = {}
    for e in errors:
        etype = e["error"].split(":")[0]
        error_types[etype] = error_types.get(etype, 0) + 1

    # Save the report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_spectra": n_total,
        "extracted_ok": n_total - n_errors,
        "errors": n_errors,
        "error_types": error_types,
        "n_shards": shard_idx,
        "elapsed_seconds": round(elapsed, 1),
        "rate_per_second": round(n_total / elapsed, 1) if elapsed > 0 else 0,
        "output_dir": str(args.output_dir),
    }

    report_path = args.output_dir / "extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Format distributions
    format_counts = {}
    for r in all_results:
        fmt = r.get("format") or "error"
        format_counts[fmt] = format_counts.get(fmt, 0) + 1

    print(f"\n{'=' * 70}")
    print(f"EXTRACTION DONE")
    print(f"{'=' * 70}")
    print(f"  Spectra extracted: {n_total - n_errors:,} / {n_total:,}")
    print(f"  Errors            : {n_errors:,}")
    print(f"  Parquet shards    : {shard_idx}")
    print(f"  Duration          : {elapsed / 60:.1f} min ({report['rate_per_second']} spectra/s)")
    print(f"  Report            : {report_path}")
    print(f"\n  Detected FITS formats :")
    for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
        print(f"    {fmt:<25} : {count:>8,}")
    if error_types:
        print(f"\n  Error types :")
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"    {etype:<25} : {count:>8,}")


def _write_shard(records: list, output_dir: Path, shard_idx: int):
    """Write a Parquet shard containing the extracted spectra."""
    ids = [r["id"] for r in records]
    wavelengths = [r["wavelength"] for r in records]
    fluxes = [r["flux"] for r in records]
    flux_errors = [r["flux_error"] for r in records]
    n_pixels = [r["n_pixels"] for r in records]
    formats = [r["format"] for r in records]

    table = pa.table({
        "id": pa.array(ids, type=pa.string()),
        "wavelength": pa.array(wavelengths, type=pa.list_(pa.float32())),
        "flux": pa.array(fluxes, type=pa.list_(pa.float32())),
        "flux_error": pa.array(flux_errors, type=pa.list_(pa.float32())),
        "n_pixels_extracted": pa.array(n_pixels, type=pa.int32()),
        "fits_format": pa.array(formats, type=pa.string()),
    })

    shard_path = output_dir / f"spectra_{shard_idx:04d}.parquet"
    pq.write_table(table, shard_path, compression="zstd")

    print(f"    → Shard {shard_idx:04d} written : {len(records):,} spectra "
          f"({shard_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
