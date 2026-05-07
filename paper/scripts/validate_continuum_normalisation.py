"""
validate_continuum_normalisation.py

Empirical validation of the BESS-Bench edge-mean continuum normalisation
against polynomial alternatives (poly-2 on ±50 Å, poly-3 on ±100 Å).

For each spectrum we compute EW(Hα) on the same ±30 Å integration window
using three different continuum estimators, all evaluated on the same
128-bin interpolated grid centred on Hα ±50 Å.  We report the relative
EW disagreement between the edge-mean reference and each polynomial
variant on a high-SNR subsample.

Usage:
    .venv/bin/python3 paper/scripts/validate_continuum_normalisation.py \
        --n 200 --out paper/scripts/continuum_validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset, load_from_disk

# Constants (kept local; do not import config to avoid pulling torch deps)
HALPHA_CENTER = 6562.8  # Å (air)
HALF_WINDOW = 50.0       # Å, defines the ±50 Å fiducial crop
WIDE_HALF_WINDOW = 100.0  # Å, the ±100 Å crop used for poly-3
N_BINS = 128
N_EDGE = 10              # number of edge pixels used as continuum side-bands
EW_HALF = 30.0           # ±30 Å integration window for EW
C_KMS = 299792.458


def heliocentric_correct(wl: np.ndarray, rqvh: float | None) -> np.ndarray:
    if rqvh is None or not np.isfinite(rqvh) or rqvh == 0.0:
        return wl
    return wl * (1.0 - rqvh / C_KMS)


def crop_and_interp(wl: np.ndarray, fl: np.ndarray, half_window: float, n_bins: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Crop around Hα ± half_window and interpolate to n_bins. Returns (grid, flux) or None."""
    lo, hi = HALPHA_CENTER - half_window, HALPHA_CENTER + half_window
    margin = 2.0
    mask = (wl >= lo - margin) & (wl <= hi + margin)
    if mask.sum() < 5:
        return None
    wl_c, fl_c = wl[mask], fl[mask]
    order = np.argsort(wl_c)
    wl_c, fl_c = wl_c[order], fl_c[order]
    grid = np.linspace(lo, hi, n_bins)
    flux_interp = np.interp(grid, wl_c, fl_c).astype(np.float64)
    if not np.all(np.isfinite(flux_interp)):
        return None
    return grid, flux_interp


def normalise_edge_mean(grid: np.ndarray, flux: np.ndarray, n_edge: int = N_EDGE) -> np.ndarray:
    """Reference normalisation: divide by the mean of the n_edge first and last pixels."""
    cont = float(np.mean(np.concatenate([flux[:n_edge], flux[-n_edge:]])))
    if not np.isfinite(cont) or cont <= 0:
        return flux.copy()
    return flux / cont


def normalise_poly_sidebands(grid: np.ndarray, flux: np.ndarray, n_edge: int, degree: int) -> np.ndarray:
    """Fit a polynomial of given degree on the n_edge first and last pixels (only),
    then divide the full grid by the evaluated polynomial."""
    x = np.concatenate([grid[:n_edge], grid[-n_edge:]])
    y = np.concatenate([flux[:n_edge], flux[-n_edge:]])
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return flux.copy()
    # Centre x for numerical stability
    x_c = x - HALPHA_CENTER
    grid_c = grid - HALPHA_CENTER
    coeffs = np.polyfit(x_c, y, deg=degree)
    cont = np.polyval(coeffs, grid_c)
    cont = np.where(np.isfinite(cont) & (cont > 0), cont, 1.0)
    return flux / cont


def normalise_wide_poly3(wl_full: np.ndarray, fl_full: np.ndarray) -> np.ndarray | None:
    """
    Wide poly-3 estimator:
      - crop ±100 Å
      - take 20-bin side-bands at the FAR edges (i.e. the ±90..±100 Å regions),
        which sit further from any Hα emission wing than the ±40..±50 Å bands
      - fit poly-3 on those side-bands only
      - evaluate the polynomial on the fiducial 128-bin ±50 Å grid
      - return flux_50A / cont_50A
    """
    out_wide = crop_and_interp(wl_full, fl_full, WIDE_HALF_WINDOW, 256)
    if out_wide is None:
        return None
    grid_w, flux_w = out_wide
    n_far = 20
    x = np.concatenate([grid_w[:n_far], grid_w[-n_far:]]) - HALPHA_CENTER
    y = np.concatenate([flux_w[:n_far], flux_w[-n_far:]])
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None
    coeffs = np.polyfit(x, y, deg=3)

    out_50 = crop_and_interp(wl_full, fl_full, HALF_WINDOW, N_BINS)
    if out_50 is None:
        return None
    grid_50, flux_50 = out_50
    cont = np.polyval(coeffs, grid_50 - HALPHA_CENTER)
    cont = np.where(np.isfinite(cont) & (cont > 0), cont, 1.0)
    return flux_50 / cont


def compute_ew(flux_norm: np.ndarray) -> float:
    """EW = ∫(1 - F) dλ over ±30 Å around Hα; ±50 Å crop, 128 bins."""
    delta_lambda = (2 * HALF_WINDOW) / (N_BINS - 1)
    centre = N_BINS // 2
    half = int(EW_HALF / delta_lambda)
    i0 = max(0, centre - half)
    i1 = min(N_BINS, centre + half)
    return float(np.sum(1.0 - flux_norm[i0:i1]) * delta_lambda)


def covers_window(lambda_min: float, lambda_max: float, half: float) -> bool:
    return (lambda_min <= HALPHA_CENTER - half) and (lambda_max >= HALPHA_CENTER + half)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ds_path", default="anonym-submit-26/bess-bench-26",
                   help="HuggingFace Hub dataset id (default) or path to a local dataset on disk.")
    p.add_argument("--n", type=int, default=200, help="Number of high-SNR spectra to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    print(f"Loading dataset from {args.ds_path} ...")
    if Path(args.ds_path).exists():
        ds = load_from_disk(args.ds_path)["train"]
    else:
        ds = load_dataset(args.ds_path, split="train")
    print(f"Total rows: {len(ds):,}")

    rng = np.random.default_rng(args.seed)

    # Pre-filter on metadata (fast: do not load wavelength/flux yet)
    print("Selecting candidate indices (snr_quality=excellent + ±100 Å coverage)...")
    cols = ds.select_columns(["snr_quality", "lambda_min", "lambda_max", "is_echelle_order"])
    candidate_idx: list[int] = []
    for i, row in enumerate(cols):
        if row["snr_quality"] != "excellent":
            continue
        if row["is_echelle_order"]:
            continue  # echelle orders may not span ±100 Å
        if not covers_window(row["lambda_min"], row["lambda_max"], WIDE_HALF_WINDOW):
            continue
        candidate_idx.append(i)
    print(f"Candidates with SNR≥200 + ±100 Å coverage + non-echelle: {len(candidate_idx):,}")
    if len(candidate_idx) < args.n:
        print(f"WARNING: only {len(candidate_idx)} candidates, requested {args.n}")
        sample_idx = candidate_idx
    else:
        sample_idx = list(rng.choice(candidate_idx, size=args.n, replace=False))

    results = []
    skipped = 0
    for k, i in enumerate(sample_idx):
        row = ds[int(i)]
        wl = np.asarray(row["wavelength"], dtype=np.float64)
        fl = np.asarray(row["flux"], dtype=np.float64)
        rqvh = row.get("bss_rqvh")
        wl = heliocentric_correct(wl, rqvh)

        out_50 = crop_and_interp(wl, fl, HALF_WINDOW, N_BINS)
        if out_50 is None:
            skipped += 1
            continue
        grid_50, flux_50 = out_50

        # Method A: edge-mean (reference, the current pipeline)
        f_em = normalise_edge_mean(grid_50, flux_50)
        # Method B: poly-2 on the same 10-bin side-bands at ±50 Å edges
        f_p2 = normalise_poly_sidebands(grid_50, flux_50, n_edge=N_EDGE, degree=2)
        # Method C: poly-3 on far side-bands of ±100 Å crop
        f_p3 = normalise_wide_poly3(wl, fl)
        if f_p3 is None:
            skipped += 1
            continue

        ew_em = compute_ew(f_em)
        ew_p2 = compute_ew(f_p2)
        ew_p3 = compute_ew(f_p3)

        results.append({
            "idx": int(i),
            "spectrum_id": row["spectrum_id"],
            "snr": float(row["snr"]),
            "ew_edge_mean": ew_em,
            "ew_poly2_50A": ew_p2,
            "ew_poly3_100A": ew_p3,
        })

        if (k + 1) % 25 == 0:
            print(f"  processed {k+1}/{len(sample_idx)}")

    print(f"Done. {len(results)} usable, {skipped} skipped.")

    # Aggregate
    ew_em = np.array([r["ew_edge_mean"] for r in results])
    ew_p2 = np.array([r["ew_poly2_50A"] for r in results])
    ew_p3 = np.array([r["ew_poly3_100A"] for r in results])

    def summarise(label: str, alt: np.ndarray) -> dict:
        # Relative absolute EW disagreement, normalising by |EW_em| with a 0.5 Å floor
        denom = np.maximum(np.abs(ew_em), 0.5)
        rel = np.abs(alt - ew_em) / denom
        # Absolute Å difference
        abs_diff = np.abs(alt - ew_em)
        d = {
            "label": label,
            "n": int(len(alt)),
            "rel_abs_pct_p50": float(np.median(rel) * 100.0),
            "rel_abs_pct_p95": float(np.percentile(rel, 95) * 100.0),
            "rel_abs_pct_mean": float(np.mean(rel) * 100.0),
            "abs_diff_A_p50": float(np.median(abs_diff)),
            "abs_diff_A_p95": float(np.percentile(abs_diff, 95)),
            "ew_em_mean": float(np.mean(ew_em)),
            "ew_em_std": float(np.std(ew_em)),
            "ew_alt_mean": float(np.mean(alt)),
            "ew_alt_std": float(np.std(alt)),
            "pearson_r": float(np.corrcoef(ew_em, alt)[0, 1]),
        }
        return d

    summary = {
        "n_requested": args.n,
        "n_candidates": len(candidate_idx),
        "n_kept": len(results),
        "n_skipped": skipped,
        "stratum": "snr_quality=excellent (SNR>=200), non-echelle, ±100 Å coverage",
        "comparisons": [
            summarise("poly-2 on ±50 Å", ew_p2),
            summarise("poly-3 on ±100 Å (far side-bands)", ew_p3),
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_spectrum": results}, f, indent=2)

    print("\n=== SUMMARY ===")
    for c in summary["comparisons"]:
        print(f"  {c['label']}:")
        print(f"    relative |ΔEW| / max(|EW|,0.5Å): median={c['rel_abs_pct_p50']:.2f}%  "
              f"P95={c['rel_abs_pct_p95']:.2f}%  mean={c['rel_abs_pct_mean']:.2f}%")
        print(f"    absolute |ΔEW|: median={c['abs_diff_A_p50']:.3f} Å  P95={c['abs_diff_A_p95']:.3f} Å")
        print(f"    Pearson r vs edge-mean = {c['pearson_r']:.4f}")
        print(f"    EW edge-mean: μ={c['ew_em_mean']:.2f} ± {c['ew_em_std']:.2f} Å | "
              f"alt: μ={c['ew_alt_mean']:.2f} ± {c['ew_alt_std']:.2f} Å")


if __name__ == "__main__":
    main()
