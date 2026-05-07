"""
t2_cross_line_probing.py — T2 (LineTransfer): linear probes Hβ-window -> Hα features (cross-line).

Demonstrates that the full-range BESS-Bench dataset contains predictive
signal beyond the Hα window alone. For each target Hα spectral feature
(EW, FWHM, V/R, Δv, central_depth, peak_intensity), trains Ridge CV probes
on Hβ-only window representations:

  1. PCA(10) of Hβ flux_norm         (coarse linear compression)
  2. PCA(50) of Hβ flux_norm         (fine linear compression)
  3. raw Hβ flux_norm (128D)         (direct representation)

Targets: Hα labels read from results/spectral_features.pt
(same as T1; aligned by key (star_name, mjd)).

Expected interpretation:
  - R² > 0 -> Hβ window encodes predictive signal for Hα features
    -> justifies the full-range release.
  - Low but finite R² -> limited but present signal (Balmer decrement).
  - R² ≈ 0 or negative -> Hβ carries no exploitable info for a linear
    probe.

Usage:
    python t2_cross_line_probing.py
    python t2_cross_line_probing.py \\
        --hbeta_features results/spectral_features_hbeta.pt \\
        --halpha_features results/spectral_features.pt \\
        --features ew fwhm vr_ratio delta_v
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Configuration ────────────────────────────────────────────────────────────

_BENCH_ROOT = Path(__file__).resolve().parents[1]
HBETA_FEATURES_PATH = str(_BENCH_ROOT / "reference_results" / "features" / "spectral_features_hbeta.pt")
HALPHA_FEATURES_PATH = str(_BENCH_ROOT / "reference_results" / "features" / "spectral_features.pt")
_DEFAULT_OUT = str(_BENCH_ROOT / "reference_results" / "downstream" / "probe_t2_hbeta_to_halpha.json")
_DEFAULT_FIG = str(_BENCH_ROOT / "results_reproduced" / "figures" / "t2_hbeta_to_halpha.png")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

FEATURE_NAMES = ["ew", "fwhm", "vr_ratio", "delta_v", "central_depth", "peak_intensity"]
FEATURE_LABELS = {
    "ew": "EW(Hα) [Å]",
    "fwhm": "FWHM(Hα) [Å]",
    "vr_ratio": "V/R(Hα)",
    "delta_v": "Δv(Hα) [km/s]",
    "central_depth": "Central depth(Hα)",
    "peak_intensity": "Peak intensity(Hα)",
}

PCA_DIMS = [10, 50]
CV_FOLDS = 5
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ══════════════════════════════════════════════════════════════════════════════
# ALIGN Hβ ↔ Hα by (star, mjd)
# ══════════════════════════════════════════════════════════════════════════════

def align_hbeta_halpha(hbeta_data, halpha_data, mjd_tol=0.01):
    """
    Align Hβ flux_norm ↔ Hα features by (star_name, mjd).

    Returns (X_hbeta_flux, Y_features_dict, meta_list).
    """
    all_flux_hbeta = []
    all_features = {name: [] for name in FEATURE_NAMES}
    all_meta = []

    common_stars = set(hbeta_data.keys()) & set(halpha_data.keys())
    print(f"  Common stars : {len(common_stars)} "
          f"(Hβ: {len(hbeta_data)}, Hα: {len(halpha_data)})")

    n_matched = 0
    n_unmatched = 0

    for star in sorted(common_stars):
        hb = hbeta_data[star]
        ha = halpha_data[star]

        hb_mjds = hb["mjds"].numpy()
        ha_mjds = ha["mjds"].numpy()
        hb_flux = hb["flux_norm"].numpy()   # [N_hb, 128]

        for i, hmjd in enumerate(hb_mjds):
            diffs = np.abs(ha_mjds - hmjd)
            best = int(np.argmin(diffs))
            if diffs[best] < mjd_tol:
                all_flux_hbeta.append(hb_flux[i])
                for name in FEATURE_NAMES:
                    val = ha[name][best].item() if name in ha else float("nan")
                    all_features[name].append(val)
                all_meta.append((star, float(hmjd)))
                n_matched += 1
            else:
                n_unmatched += 1

    print(f"  Alignments   : {n_matched:,} matched, "
          f"{n_unmatched:,} unmatched")

    X = np.array(all_flux_hbeta, dtype=np.float32)
    Y = {name: np.array(vals, dtype=np.float32)
         for name, vals in all_features.items()}
    return X, Y, all_meta


# ══════════════════════════════════════════════════════════════════════════════
# PROBE Ridge CV
# ══════════════════════════════════════════════════════════════════════════════

def probe_ridge_cv(X, y, n_folds=CV_FOLDS, alphas=RIDGE_ALPHAS):
    valid = np.isfinite(y)
    X_val = X[valid]
    y_val = y[valid]

    if len(y_val) < n_folds * 5:
        return float("nan"), float("nan"), len(y_val)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas)),
    ])
    scores = cross_val_score(pipe, X_val, y_val, cv=n_folds, scoring="r2")
    return float(np.mean(scores)), float(np.std(scores)), len(y_val)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="T2: linear probes Hβ window -> Hα spectral features (cross-line)"
    )
    parser.add_argument("--hbeta_features", type=str, default=HBETA_FEATURES_PATH)
    parser.add_argument("--halpha_features", type=str, default=HALPHA_FEATURES_PATH)
    parser.add_argument("--features", nargs="+", default=FEATURE_NAMES)
    parser.add_argument("--output", type=str,
                        default=_DEFAULT_OUT)
    parser.add_argument("--output_fig", type=str,
                        default=_DEFAULT_FIG)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TASK T2 : Hβ WINDOW -> Hα FEATURE PROBING (CROSS-LINE)")
    print("=" * 70)

    # ── Load ──
    print("\n1. Loading Hβ-window features (flux_norm)...")
    hbeta_data = torch.load(args.hbeta_features, map_location="cpu",
                            weights_only=False)
    print(f"   {len(hbeta_data)} stars with Hβ coverage")

    print("2. Loading Hα features (targets for T2)...")
    halpha_data = torch.load(args.halpha_features, map_location="cpu",
                             weights_only=False)
    print(f"   {len(halpha_data)} stars with Hα features")

    print("\n3. Aligning Hβ flux ↔ Hα features by (star, mjd)...")
    X_flux, Y, meta = align_hbeta_halpha(hbeta_data, halpha_data)
    print(f"   Hβ flux matrix : {X_flux.shape}")

    if X_flux.shape[0] < 200:
        print("   ⚠ Too few aligned pairs; aborting.")
        return

    # ── Representations ──
    print("\n4. Building representations from Hβ flux...")
    representations = {"Hβ flux (128D)": X_flux}

    for n_comp in PCA_DIMS:
        k = min(n_comp, X_flux.shape[0] - 1, X_flux.shape[1])
        pca = PCA(n_components=k)
        X_pca = pca.fit_transform(X_flux)
        explained = float(pca.explained_variance_ratio_.sum())
        representations[f"Hβ PCA({k})"] = X_pca
        print(f"   Hβ PCA({k}) -> variance explained: {explained:.3f}")

    # ── Probes ──
    print("\n5. Running Ridge CV probes (Hβ repr -> Hα feature)...")
    results = {}

    for feat in args.features:
        if feat not in Y:
            print(f"   ! feature '{feat}' not available, skipping")
            continue

        y = Y[feat]
        n_finite = int(np.isfinite(y).sum())
        print(f"\n   ── {feat} ({n_finite:,} finite out of {len(y):,}) ──")

        results[feat] = {}
        for repr_name, X in representations.items():
            r2, std, n_valid = probe_ridge_cv(X, y)
            results[feat][repr_name] = {
                "r2_mean": r2,
                "r2_std": std,
                "n_valid": n_valid,
            }
            r2_s = f"{r2:.4f}" if np.isfinite(r2) else "N/A"
            std_s = f"{std:.4f}" if np.isfinite(std) else "N/A"
            print(f"     {repr_name:<20s} R²={r2_s} ±{std_s}  N={n_valid}")

    # ── Save ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "T2_cross_line_probing",
        "description": "Cross-line probing: Hβ window representations -> "
                       "predict Hα spectral features. Justifies "
                       "full-spectrum release of BESS-Bench.",
        "n_aligned_pairs": int(X_flux.shape[0]),
        "n_stars_covered": len(hbeta_data),
        "hbeta_window_AA": [4811.3, 4911.3],
        "representations": list(representations.keys()),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ── Figure ──
    if results:
        fig, ax = plt.subplots(figsize=(10, 5))

        feats = list(results.keys())
        reprs = list(representations.keys())
        width = 0.25

        for j, rname in enumerate(reprs):
            means = [results[f][rname]["r2_mean"] for f in feats]
            stds = [results[f][rname]["r2_std"] for f in feats]
            xs = np.arange(len(feats)) + (j - len(reprs) / 2 + 0.5) * width
            ax.bar(xs, means, width=width, yerr=stds, capsize=3,
                   label=rname, alpha=0.85, edgecolor="black")

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
        ax.set_xticks(np.arange(len(feats)))
        ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in feats],
                           rotation=20, ha="right")
        ax.set_ylabel("R² (5-fold CV)")
        ax.set_title("T2 — Hβ window -> Hα features (cross-line probing)")
        ax.legend(loc="best")
        plt.tight_layout()

        fig_path = Path(args.output_fig)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
