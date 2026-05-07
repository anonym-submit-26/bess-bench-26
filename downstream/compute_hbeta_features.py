"""
compute_hbeta_features.py — Extract Hβ-window flux for T2 (cross-line probing).

For each BeSS spectrum covering the Hβ ± 50 Å window
(≈ 4811–4911 Å), computes and stores:
  - flux_norm       : flux interpolated on a 128-bin grid + normalized
                      (same format as compute_spectral_features.py for Hα).
  - ew_hbeta        : EW measured locally on the Hβ window (diagnostic,
                      not used as target).
  - mjds, star_name : alignment keys with spectral_features.pt (Hα).

Target labels for T2 (EW(Hα), FWHM(Hα), ...) are reused from
reference_results/features/spectral_features.pt, aligned by (star_name, mjd).

Usage:
    python compute_hbeta_features.py
    python compute_hbeta_features.py --max_spectra 1000
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# ── Hβ constants ───────────────────────────────────────────────────────────────

HBETA_CENTER = 4861.3        # Å (Hβ rest-frame air)
HBETA_HALF_WINDOW = 50.0     # Å
N_BINS = 128
LAMBDA_MIN = HBETA_CENTER - HBETA_HALF_WINDOW   # 4811.3
LAMBDA_MAX = HBETA_CENTER + HBETA_HALF_WINDOW   # 4911.3
TARGET_GRID = np.linspace(LAMBDA_MIN, LAMBDA_MAX, N_BINS)
DELTA_LAMBDA = TARGET_GRID[1] - TARGET_GRID[0]   # ~0.78 Å

# Speed of light (km/s) for the heliocentric correction
C_KMS = 299792.458


# ════════════════════════════════════════════════════════════════════════════
# CROP + NORMALISATION (mirrors compute_spectral_features.py)
# ══════════════════════════════════════════════════════════════════════════════

def covers_hbeta_window(wl_min: float, wl_max: float) -> bool:
    """True if [wl_min, wl_max] fully covers the Hβ ± 50 Å window."""
    return wl_min <= LAMBDA_MIN and wl_max >= LAMBDA_MAX


def crop_and_interpolate(flux, wavelength, target_grid=TARGET_GRID,
                        rqvh_kms=None):
    """Crop + interpolation + heliocentric correction (BSS_RQVH, km/s)."""
    if rqvh_kms is not None and np.isfinite(rqvh_kms) and rqvh_kms != 0.0:
        wavelength = wavelength * (1.0 - rqvh_kms / C_KMS)
    margin = 2.0
    mask = (wavelength >= LAMBDA_MIN - margin) & (wavelength <= LAMBDA_MAX + margin)
    if mask.sum() < 5:
        return None
    wl_crop = wavelength[mask]
    fl_crop = flux[mask]
    sort_idx = np.argsort(wl_crop)
    wl_crop = wl_crop[sort_idx]
    fl_crop = fl_crop[sort_idx]
    return np.interp(target_grid, wl_crop, fl_crop).astype(np.float32)


def normalize_local_continuum(flux, n_edge=10):
    if len(flux) < 2 * n_edge:
        n_edge = max(1, len(flux) // 4)
    left = flux[:n_edge]
    right = flux[-n_edge:]
    continuum = np.mean(np.concatenate([left, right]))
    if continuum <= 0 or not np.isfinite(continuum):
        continuum = 1.0
    normalized = flux / continuum
    normalized = np.clip(normalized, -5.0, 10.0)
    normalized = np.nan_to_num(normalized, nan=1.0, posinf=10.0, neginf=-5.0)
    return normalized


def compute_ew_local(flux_norm, delta_lambda=DELTA_LAMBDA):
    """
    EW integrated over the central window ±30 Å around Hβ.
    EW > 0 = absorption (typical for Hβ on Be stars),
    EW < 0 = emission.
    """
    center_idx = len(flux_norm) // 2
    half_window = int(30.0 / delta_lambda)
    i_start = max(0, center_idx - half_window)
    i_end = min(len(flux_norm), center_idx + half_window)
    flux_window = flux_norm[i_start:i_end]
    return float(np.sum((1.0 - flux_window)) * delta_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract Hβ-window flux_norm for T2 cross-line probing"
    )
    parser.add_argument("--max_spectra", type=int, default=None,
                        help="Max spectra to process (for testing)")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "reference_results" / "features" / "spectral_features_hbeta.pt"))
    parser.add_argument("--batch_log", type=int, default=5000)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the HF dataset
    print("Loading BeSS dataset from HuggingFace...")
    from datasets import load_dataset  # noqa

    ds = load_dataset("anonym-submit-26/bess-bench-26", split="train")
    print(f"  Total raw: {len(ds):,} spectra")

    data_by_star = defaultdict(lambda: {
        "mjds": [],
        "ew_hbeta": [],
        "flux_norm": [],
    })

    n_processed = 0
    n_no_coverage = 0
    n_bad = 0

    for i in range(len(ds)):
        if args.max_spectra and n_processed >= args.max_spectra:
            break

        row = ds[i]
        flux_raw = np.array(row["flux"], dtype=np.float32)
        wavelength = np.array(row["wavelength"], dtype=np.float32)
        star_name = row.get("star_name", "")
        mjd = float(row.get("mjd", 0.0))

        # BSS_RQVH (residual heliocentric velocity, km/s)
        rqvh_raw = row.get("bss_rqvh", None)
        rqvh_kms = float(rqvh_raw) if rqvh_raw is not None else 0.0
        if not np.isfinite(rqvh_kms):
            rqvh_kms = 0.0

        if not star_name or mjd <= 0:
            n_bad += 1
            continue

        wl_min = wavelength.min() if len(wavelength) > 0 else 0
        wl_max = wavelength.max() if len(wavelength) > 0 else 0
        if not covers_hbeta_window(wl_min, wl_max):
            n_no_coverage += 1
            continue

        flux_interp = crop_and_interpolate(flux_raw, wavelength, rqvh_kms=rqvh_kms)
        if flux_interp is None:
            n_bad += 1
            continue

        flux_norm = normalize_local_continuum(flux_interp)
        ew_h = compute_ew_local(flux_norm)

        sd = data_by_star[star_name]
        sd["mjds"].append(mjd)
        sd["ew_hbeta"].append(ew_h)
        sd["flux_norm"].append(flux_norm)

        n_processed += 1

        if n_processed % args.batch_log == 0:
            print(f"  {n_processed:,} Hβ spectra kept, "
                  f"{len(data_by_star)} stars, "
                  f"{n_no_coverage:,} no coverage, {n_bad:,} bad")

    total_raw = min(len(ds), args.max_spectra) if args.max_spectra else len(ds)
    coverage_pct = 100.0 * n_processed / max(1, total_raw)
    print(
        f"\nDone: {n_processed:,}/{total_raw:,} "
        f"spectra cover Hβ ({coverage_pct:.1f}%), "
        f"{len(data_by_star)} stars, "
        f"{n_no_coverage:,} without coverage, {n_bad:,} bad"
    )

    # ── Convert to tensors, sort by MJD ──
    print("\nConverting to tensors and sorting by MJD...")
    final_data = {}
    for star, data in data_by_star.items():
        indices = np.argsort(data["mjds"])
        final_data[star] = {
            "mjds":      torch.tensor([data["mjds"][i] for i in indices], dtype=torch.float32),
            "ew_hbeta":  torch.tensor([data["ew_hbeta"][i] for i in indices], dtype=torch.float32),
            "flux_norm": torch.tensor(
                np.stack([data["flux_norm"][i] for i in indices]),
                dtype=torch.float32,
            ),  # [n_obs, 128]
        }

    # ── Save ──
    print(f"Saving to {output_path}...")
    torch.save(final_data, output_path)

    # ── Stats and manifest ──
    manifest = {
        "total_spectra_raw": int(total_raw),
        "n_with_hbeta_coverage": int(n_processed),
        "hbeta_coverage_pct": round(coverage_pct, 2),
        "n_stars": len(final_data),
        "n_without_coverage": int(n_no_coverage),
        "n_bad": int(n_bad),
        "window_center_AA": HBETA_CENTER,
        "window_half_width_AA": HBETA_HALF_WINDOW,
        "n_bins": N_BINS,
        "output_path": output_path.name,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")

    print("\n── Summary ──")
    for k, v in manifest.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
