"""
validate_morphology_audit.py — Visual audit of morphological classification rules.

For each profile class (absorption_pure / single_peak / double_peak /
double_asym / shell), plots 10 representative spectra with overlays:

  • detected emission peak positions   (red circles)
  • ``half_level`` used for FWHM       (horizontal line)
  • left/right FWHM bounds             (vertical segments)
  • central_depth measurement position
  • text box with EW, FWHM, V/R, central_depth, n_peaks

The goal is to visually verify that:
  1. peaks are detected at the right position
  2. FWHM covers the full emission profile (Hanuschik convention)
  3. central_depth points to the central trough, not a peak
  4. V/R matches the visual perception of asymmetry
  5. displayed examples are consistent with their assigned class

Usage
-----
    python validate_morphology_audit.py [--n-per-class 10] [--seed 42]
                                        [--strategy random|spread|extreme]

  • random  : uniform draw within the class (default)
  • spread  : 10 EW/V-R/central_depth percentiles to cover the range
  • extreme : 5 medoids + 5 boundary cases (useful for stress-testing rules)

Output: ``downstream/figures/morphology_audit.pdf`` (5 pages, 10 spectra / page).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

from compute_spectral_features import (
    TARGET_GRID, DELTA_LAMBDA, HALPHA_CENTER,
    compute_ew, compute_fwhm, compute_vr_ratio,
    compute_central_depth, find_emission_peaks, classify_profile,
)

CLASSES = [
    (0, "absorption_pure", "Pure absorption",     "#4477AA"),
    (1, "single_peak",     "Single-peak",          "#228833"),
    (2, "double_peak",     "Double-peak (V≈R)",    "#CCBB44"),
    (3, "double_asym",     "Asymmetric V/R",       "#AA3377"),
    (4, "shell",           "Shell",                "#EE6677"),
]

FEATURES = Path(__file__).parent / "results" / "spectral_features.pt"
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
def collect_class_pool(feats: dict, target_pid: int):
    """List all (star, idx, flux) entries belonging to class target_pid."""
    pool = []
    for star, d in feats.items():
        flux = d["flux_norm"].numpy()
        pids = d["profile_type_id"].numpy()
        for i in range(flux.shape[0]):
            if int(pids[i]) == target_pid and not np.isnan(flux[i]).any():
                pool.append((star, i, flux[i]))
    return pool


def select_examples(pool, n: int, strategy: str, rng: np.random.Generator):
    if not pool:
        return []
    if strategy == "random" or len(pool) <= n:
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[i] for i in idx]

    # Compute features for ranking
    fluxes = np.stack([p[2] for p in pool], axis=0)
    if strategy == "spread":
        # spread by EW: pick equispaced quantiles
        ews = np.array([compute_ew(p[2]) for p in pool])
        order = np.argsort(ews)
        qs = np.linspace(0, len(order) - 1, n).astype(int)
        return [pool[order[q]] for q in qs]

    if strategy == "extreme":
        # 5 medoids + 5 furthest from medoid (boundary cases)
        median_curve = np.median(fluxes, axis=0)
        dists = np.linalg.norm(fluxes - median_curve[None, :], axis=1)
        order = np.argsort(dists)
        n_med = n // 2
        n_ext = n - n_med
        picks = list(order[:n_med]) + list(order[-n_ext:])
        return [pool[i] for i in picks]

    raise ValueError(f"unknown strategy {strategy}")


# ────────────────────────────────────────────────────────────────────────────
def annotate_spectrum(ax, flux: np.ndarray, color: str):
    """Draw flux + diagnostic overlays.

    Returns a dict of computed features for the legend.
    """
    lam = TARGET_GRID
    center_idx = len(flux) // 2

    ew    = compute_ew(flux)
    fwhm  = compute_fwhm(flux)
    vr    = compute_vr_ratio(flux)
    depth = compute_central_depth(flux)
    peaks = find_emission_peaks(flux)
    pid, pname = classify_profile(flux, ew=ew)

    ax.axvline(HALPHA_CENTER, color="0.85", lw=0.5, zorder=0)
    ax.axhline(1.0, color="0.85", lw=0.4, zorder=0)
    ax.plot(lam, flux, color=color, lw=0.9)

    # Detected peaks
    if peaks:
        px = [p["wavelength"] for p in peaks]
        py = [p["intensity"]  for p in peaks]
        ax.scatter(px, py, s=18, facecolors="none", edgecolors="red",
                   lw=0.9, zorder=4)

    # Half-level FWHM line (only if FWHM is finite)
    if np.isfinite(fwhm) and fwhm > 0:
        # reconstruct the half_level the same way compute_fwhm does
        half_w = int(25.0 / DELTA_LAMBDA)
        i_lo = max(0, center_idx - half_w)
        i_hi = min(len(flux), center_idx + half_w)
        f_window = flux[i_lo:i_hi]
        close = int(5.0 / DELTA_LAMBDA)
        center_local = center_idx - i_lo
        core = f_window[max(0, center_local - close): center_local + close + 1]
        is_emission = (float(np.max(core)) - 1.0) > (1.0 - float(np.min(core)))
        if is_emission:
            left_seg = f_window[:center_local]
            right_seg = f_window[center_local:]
            pl = float(np.max(left_seg)) if len(left_seg) else 1.0
            pr = float(np.max(right_seg)) if len(right_seg) else 1.0
            if pl <= 1.02 or pr <= 1.02:
                peak_ref = max(pl, pr)
            else:
                peak_ref = min(pl, pr)
            half_level = 1.0 + (peak_ref - 1.0) / 2.0
        else:
            half_level = (1.0 + float(np.min(f_window))) / 2.0
        ax.hlines(half_level, HALPHA_CENTER - fwhm / 2, HALPHA_CENTER + fwhm / 2,
                  colors="orange", lw=1.2, alpha=0.85, zorder=3)

    # Central-depth marker
    if np.isfinite(depth):
        cdepth_x = HALPHA_CENTER
        # value used by compute_central_depth: min in ±3 Å around centre
        c3 = int(3.0 / DELTA_LAMBDA)
        f_at_center = float(np.min(flux[max(0, center_idx - c3):
                                        min(len(flux), center_idx + c3 + 1)]))
        ax.scatter([cdepth_x], [f_at_center], s=20, marker="v",
                   color="blue", zorder=5)

    # Annotation box
    txt = (f"EW={ew:+.2f}\nFWHM={fwhm:.1f}\n"
           f"V/R={vr:.2f}\ndepth={depth:+.2f}\n"
           f"n_peaks={len(peaks)}")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=6.5, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="0.7", lw=0.4, alpha=0.9))

    return {"ew": ew, "fwhm": fwhm, "vr": vr, "depth": depth,
            "n_peaks": len(peaks), "rederived": pname}


# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strategy", choices=["random", "spread", "extreme"],
                    default="extreme")
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "morphology_audit.pdf"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"[load] {FEATURES}")
    feats = torch.load(FEATURES, map_location="cpu", weights_only=False)
    print(f"  {len(feats)} stars\n")

    out_pdf = Path(args.out)
    out_pdf.parent.mkdir(exist_ok=True, parents=True)
    print(f"[write] {out_pdf}")

    n_disagreements = 0
    n_total = 0

    with PdfPages(out_pdf) as pdf:
        for pid, name, pretty, color in CLASSES:
            pool = collect_class_pool(feats, pid)
            print(f"  class {pid} ({name:18s}) : {len(pool):>6,d} spectra")
            picks = select_examples(pool, args.n_per_class, args.strategy, rng)
            if not picks:
                continue

            ncols = 5
            nrows = int(np.ceil(len(picks) / ncols))
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(11, 2.5 * nrows + 0.6),
                                     sharey=False)
            axes = np.array(axes).reshape(nrows, ncols)

            for i, (star, idx, flux) in enumerate(picks):
                r, c = divmod(i, ncols)
                ax = axes[r, c]
                stats = annotate_spectrum(ax, flux, color)
                tag = f"{star} #{idx}"
                if stats["rederived"] != name:
                    tag += f"  ⚠ relabel→{stats['rederived']}"
                    n_disagreements += 1
                n_total += 1
                ax.set_title(tag, fontsize=7)
                ax.set_xlim(TARGET_GRID.min(), TARGET_GRID.max())
                ax.tick_params(labelsize=7, length=2)

            for j in range(len(picks), nrows * ncols):
                axes.flat[j].axis("off")

            fig.suptitle(
                f"Class {pid}: {pretty}  —  {len(pool):,} spectra in this class  "
                f"(strategy={args.strategy}, n={len(picks)})",
                fontsize=10, y=0.995, color=color
            )
            fig.text(0.5, 0.005,
                     "red ○ peaks   |   orange ─ FWHM half-level span   |   "
                     "blue ▼ central-depth probe (Hα)",
                     ha="center", fontsize=7, color="0.4")
            fig.tight_layout(rect=(0, 0.015, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n[summary] {n_total} spectra audited, "
          f"{n_disagreements} re-derivation disagreement(s) "
          f"({100 * n_disagreements / max(1, n_total):.1f}%)")
    print(f"[ok] {out_pdf}")


if __name__ == "__main__":
    main()
