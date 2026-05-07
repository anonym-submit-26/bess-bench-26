#!/usr/bin/env python3
"""
Module 1 — Automated SNR Estimation of BeSS Spectra.

Primary Method: DER_SNR (Stoehr et al. 2008)
    Standard ESO algorithm, independent of continuum window selection.
    Exploits second-order finite differences to estimate pixel-by-pixel noise:

        noise = 1.482602 × median(|2·f_i − f_{i−2} − f_{i+2}|)
        SNR   = median(f) / noise

    Reference: Stoehr, F. et al. (2008), ST-ECF Newsletter 42, 4
    https://www.stecf.org/software/ASTROsoft/DER_SNR/

Secondary Method (cross-validation): Continuum Windows
    Spectral zones adapted to Be stars, avoiding He I, Hα, telluric lines.
    Used to check consistency with DER_SNR.

Columns written to DuckDB:
    - snr_estimated     : SNR DER_SNR (primary method)
    - snr_continuum     : SNR from continuum windows (cross-validation)

Usage:
    python compute_snr.py [--batch-size N] [--sample N] [--dry-run]
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

import duckdb
import numpy as np
from astropy.io import fits

# Configuration
DB_PATH = Path(__file__).resolve().parent.parent.parent / "harvest" / "bess_data.duckdb"
FITS_DIR = Path(os.environ.get("BESS_FITS_DIR", "./spectra_data"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Continuum zones adapted to Be stars (in Angstroms)
# Windows chosen to AVOID characteristic Be lines:
#   - He I 4471, 4713, 5876, 6678, 7065
#   - Hα 6563, Hβ 4861, Hγ 4340
#   - Na D 5890/5896
#   - Fe II 6347/6371
#   - Fe II multiplets
#
# IMPORTANT: ~25% of BeSS spectra are HR centered on Hα (start > 6200 Å).
# Windows 1-5 cover only wide-band spectra.
# Windows 6-7 cover the Hα zone for HR spectra.
#
CONTINUUM_WINDOWS = [
    # --- Wide-band windows (spectra spanning < 6200 Å) ---
    (4000, 4100),  # Blue zone, before Hδ 4101 — clean continuum
    (4550, 4650),  # Between Hγ 4340 and Hβ 4861, far from He I 4471/4713
    (5400, 5500),  # Before Na D 5890, far from He I 5876
    (5800, 5870),  # Before He I 5876, after Fe II 5780 — narrow but clean window
    (6100, 6200),  # Between Fe II 6347 and Hα, reduced vs v1
    # --- Hα zone windows (HR spectra centered on Hα) ---
    (6380, 6470),  # Before Hα: after Fe II 6371, well before Hα 6563
    (6730, 6800),  # After He I 6678: far from the line, before telluric B band ~6860
]

# Minimum number of points per window for a valid computation
MIN_POINTS_PER_WINDOW = 20

# Minimum number of pixels for DER_SNR
MIN_PIXELS_DER_SNR = 50


def load_spectrum(fits_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load a spectrum from a FITS file.
    
    Returns:
        (wavelength_in_angstrom, flux) or None if error
    """
    try:
        with fits.open(fits_path) as hdul:
            # SPECTRUM extension (BeSS standard format)
            ext_names = [h.name for h in hdul]
            
            if 'SPECTRUM' in ext_names:
                data = hdul['SPECTRUM'].data
                wave = np.array(data['WAVE'], dtype=np.float64)
                flux = np.array(data['FLUX'], dtype=np.float64)
            elif len(hdul) > 1 and hasattr(hdul[1], 'data') and hdul[1].data is not None:
                data = hdul[1].data
                if 'WAVE' in data.names and 'FLUX' in data.names:
                    wave = np.array(data['WAVE'], dtype=np.float64)
                    flux = np.array(data['FLUX'], dtype=np.float64)
                else:
                    return None
            else:
                # Simple image format (1D)
                header = hdul[0].header
                flux = np.array(hdul[0].data, dtype=np.float64)
                if flux is None or flux.ndim == 0:
                    return None
                
                crval1 = header.get('CRVAL1', 0)
                cdelt1 = header.get('CDELT1', header.get('CD1_1', 1))
                crpix1 = header.get('CRPIX1', 1)
                wave = crval1 + (np.arange(len(flux)) - crpix1 + 1) * cdelt1
                
                # Convert nm → Å if necessary
                cunit = header.get('CUNIT1', 'Angstroms')
                if cunit and 'nm' in cunit.lower():
                    wave = wave * 10.0
            
            # Consistency check
            if len(wave) != len(flux):
                return None
            if len(wave) < 50:
                return None
            
            # Convert nm → Å if wavelengths appear to be in nm
            if wave.max() < 1000:
                wave = wave * 10.0
            
            return wave, flux
            
    except Exception as e:
        logger.debug(f"Load error {fits_path.name}: {e}")
        return None


def der_snr(flux: np.ndarray) -> Optional[float]:
    """
    Calculate SNR using the DER_SNR method (Stoehr et al. 2008).
    
    Standard ESO algorithm based on second-order finite differences.
    Independent of window choice, robust to emission/absorption lines.
    
        noise = 1.482602 × median(|2·f_i − f_{i−2} − f_{i+2}|)
        SNR   = median(f) / noise
    
    The factor 1.482602 = 1/Φ⁻¹(3/4) converts MAD to standard deviation
    for a Gaussian distribution.
    
    Returns:
        Estimated SNR or None if impossible
    """
    # Filter NaN/Inf
    valid = np.isfinite(flux)
    f = flux[valid]
    
    if len(f) < MIN_PIXELS_DER_SNR:
        return None
    
    # Signal: median of flux
    signal = np.median(f)
    if signal <= 0:
        return None
    
    # Noise: second-order finite differences
    # Δ_i = 2·f_i − f_{i−2} − f_{i+2}
    # For white Gaussian noise, Var(Δ) = 4·σ² + σ² + σ² = 6·σ²  (incorrect)
    # Actually: Var(2f_i - f_{i-2} - f_{i+2}) = 4·σ² + σ² + σ² = 6·σ²
    # Thus σ_Δ = σ·√6, and MAD(Δ) = 1.482602·σ·√6...
    # The original DER_SNR implementation uses directly:
    #   noise = 1.482602 / √6 × median(|Δ|) ... NO
    #
    # Exact implementation by Stoehr et al.:
    n = len(f)
    delta = 2.0 * f[2:n-2] - f[0:n-4] - f[4:n]
    noise = 1.482602 * np.median(np.abs(delta)) / np.sqrt(6.0)
    
    if noise <= 0:
        return None
    
    return float(signal / noise)


def estimate_snr_continuum(wave: np.ndarray, flux: np.ndarray) -> Optional[float]:
    """
    Estimate SNR using continuum windows (secondary method, cross-validation).
    
    Windows adapted to Be stars:
        - Avoid He I 4471, 5876, 6678, 7065
        - Avoid Hα/Hβ/Hγ
        - Avoid Na D, Fe II main lines
    
    Algorithm:
        1. For each continuum window within spectral coverage
        2. Extract flux in this zone
        3. Apply sigma-clipping (3σ, 2 iterations)
        4. Calculate SNR = μ / σ
        5. Return median of SNR values from valid windows
    
    Returns:
        Estimated SNR or None if impossible
    """
    snr_values = []
    
    for wmin, wmax in CONTINUUM_WINDOWS:
        # Select the zone
        mask = (wave >= wmin) & (wave <= wmax)
        
        if mask.sum() < MIN_POINTS_PER_WINDOW:
            continue
        
        flux_window = flux[mask]
        
        # Remove NaN/Inf
        valid = np.isfinite(flux_window)
        if valid.sum() < MIN_POINTS_PER_WINDOW:
            continue
        flux_window = flux_window[valid]
        
        # Sigma-clipping (2 iterations, 3σ)
        for _ in range(2):
            mu = np.median(flux_window)
            sigma = np.std(flux_window)
            if sigma == 0:
                break
            clip_mask = np.abs(flux_window - mu) < 3 * sigma
            if clip_mask.sum() < MIN_POINTS_PER_WINDOW:
                break
            flux_window = flux_window[clip_mask]
        
        # Compute SNR
        mu = np.mean(flux_window)
        sigma = np.std(flux_window)
        
        if sigma > 0 and mu > 0:
            snr = mu / sigma
            snr_values.append(snr)
    
    if not snr_values:
        return None
    
    # Median of SNR zones for robustness
    return float(np.median(snr_values))


def add_snr_columns(conn: duckdb.DuckDBPyConnection):
    """Add snr_estimated and snr_continuum columns if they don't exist."""
    existing = set(r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'spectra_votable'"
    ).fetchall())
    
    if 'snr_estimated' not in existing:
        conn.execute("ALTER TABLE spectra_votable ADD COLUMN snr_estimated DOUBLE")
        logger.info("➕ Column snr_estimated added (DER_SNR)")
    
    if 'snr_continuum' not in existing:
        conn.execute("ALTER TABLE spectra_votable ADD COLUMN snr_continuum DOUBLE")
        logger.info("➕ Column snr_continuum added (continuum windows)")
    
    logger.info("✅ SNR columns ready")


def compute_snr_batch(
    batch_size: int = 1000,
    sample: int = None,
    dry_run: bool = False
):
    """
    Calculate SNR for all spectra.
    
    Args:
        batch_size: Batch size for database commits
        sample: If defined, only process a random sample
        dry_run: Test mode without writes
    """
    print("=" * 70)
    print("MODULE 1 — SNR ESTIMATION OF SPECTRA")
    print("  Primary method: DER_SNR (Stoehr et al. 2008)")
    print("  Secondary method: continuum windows (cross-validation)")
    print("=" * 70)
    print(f"\n📁 Database: {DB_PATH}")
    print(f"📂 FITS: {FITS_DIR}")
    print(f"📦 Batch: {batch_size}")
    if sample:
        print(f"🎲 Sample: {sample}")
    print(f"🔍 Mode: {'DRY RUN' if dry_run else 'EXECUTION'}\n")
    
    conn = duckdb.connect(str(DB_PATH), read_only=dry_run)
    
    # Add the columns
    if not dry_run:
        add_snr_columns(conn)
    
    # Retrieve spectra to process
    query = """
        SELECT id, local_filename
        FROM spectra_votable
        WHERE local_filename IS NOT NULL
          AND snr_estimated IS NULL
    """
    if sample:
        query += f" ORDER BY RANDOM() LIMIT {sample}"
    else:
        query += " ORDER BY id"
    
    try:
        rows = conn.execute(query).fetchall()
    except Exception as e:
        logger.error(f"Query error (missing column?): {e}")
        conn.close()
        return
    
    total = len(rows)
    print(f"📊 {total:,} spectra to process\n")
    
    if total == 0:
        print("✅ All spectra already have a computed SNR!")
        conn.close()
        return
    
    # Processing
    start_time = datetime.now()
    processed = 0
    updated = 0
    errors = 0
    snr_der_all = []
    snr_cont_all = []
    
    for batch_start in range(0, total, batch_size):
        batch = rows[batch_start:batch_start + batch_size]
        
        for spectrum_id, local_filename in batch:
            fits_path = FITS_DIR / local_filename
            
            result = load_spectrum(fits_path)
            if result is None:
                errors += 1
                processed += 1
                continue
            
            wave, flux = result
            
            # Primary method: DER_SNR
            snr_primary = der_snr(flux)
            
            # Secondary method: continuum windows
            snr_secondary = estimate_snr_continuum(wave, flux)
            
            if snr_primary is not None:
                if not dry_run:
                    conn.execute(
                        "UPDATE spectra_votable SET snr_estimated = ?, snr_continuum = ? WHERE id = ?",
                        [snr_primary, snr_secondary, spectrum_id]
                    )
                updated += 1
                snr_der_all.append(snr_primary)
                if snr_secondary is not None:
                    snr_cont_all.append(snr_secondary)
            else:
                errors += 1
            
            processed += 1
        
        if not dry_run:
            conn.commit()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0
        
        print(f"\r  Progress: {processed:,}/{total:,} ({processed/total*100:.1f}%) | "
              f"{rate:.0f}/s | ETA: {eta/60:.1f}min | "
              f"OK: {updated:,} | Err: {errors}", end="", flush=True)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Statistics
    print(f"\n\n{'='*70}")
    print("RESULT")
    print("=" * 70)
    print(f"✅ Processed: {processed:,}")
    print(f"✅ SNR computed: {updated:,}")
    print(f"❌ Errors/Impossible: {errors:,}")
    print(f"⏱️ Duration: {elapsed/60:.1f} minutes ({processed/elapsed:.0f}/s)")
    
    if snr_der_all:
        arr = np.array(snr_der_all)
        print(f"\n📊 SNR Distribution — DER_SNR (primary, n={len(arr):,}):")
        print(f"  Min: {arr.min():.1f}")
        print(f"  Q1:  {np.percentile(arr, 25):.1f}")
        print(f"  Median: {np.median(arr):.1f}")
        print(f"  Q3:  {np.percentile(arr, 75):.1f}")
        print(f"  Max: {arr.max():.1f}")
        print(f"  Mean: {arr.mean():.1f} ± {arr.std():.1f}")
        
        # Quality breakdown
        print(f"\n📊 Quality breakdown (DER_SNR):")
        print(f"  SNR < 10  (low):       {(arr < 10).sum():,} ({(arr < 10).mean()*100:.1f}%)")
        print(f"  SNR 10-50 (medium):    {((arr >= 10) & (arr < 50)).sum():,} ({((arr >= 10) & (arr < 50)).mean()*100:.1f}%)")
        print(f"  SNR 50-100 (good):     {((arr >= 50) & (arr < 100)).sum():,} ({((arr >= 50) & (arr < 100)).mean()*100:.1f}%)")
        print(f"  SNR 100-200 (very good):{((arr >= 100) & (arr < 200)).sum():,} ({((arr >= 100) & (arr < 200)).mean()*100:.1f}%)")
        print(f"  SNR >= 200 (excellent):{(arr >= 200).sum():,} ({(arr >= 200).mean()*100:.1f}%)")
    
    if snr_cont_all:
        arr2 = np.array(snr_cont_all)
        print(f"\n📊 SNR Distribution — Continuum (secondary, n={len(arr2):,}):")
        print(f"  Median: {np.median(arr2):.1f} | Mean: {arr2.mean():.1f} ± {arr2.std():.1f}")
    
    # Correlation between the two methods (on spectra having both values)
    if snr_der_all and snr_cont_all:
        # Build pairs (der, cont) for spectra with both values
        min_len = min(len(snr_der_all), len(snr_cont_all))
        if min_len >= 10:
            # The lists are not matched directly — just log the stats
            print(f"\n📊 Spectra with DER_SNR: {len(snr_der_all):,} | with Continuum: {len(snr_cont_all):,}")
    
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNR estimation of BeSS spectra")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    parser.add_argument("--sample", type=int, default=None, help="Process a random sample")
    parser.add_argument("--dry-run", action="store_true", help="Test mode without writes")
    args = parser.parse_args()
    
    compute_snr_batch(
        batch_size=args.batch_size,
        sample=args.sample,
        dry_run=args.dry_run
    )
