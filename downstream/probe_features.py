"""
probe_features.py — T1 (SpecProbe): linear probes z → spectral features.

Implements benchmark task T1: demonstrates that the embedding z ∈ ℝ128
encodes the complete Hα profile morphology, beyond the trivial EW
(which is implicit by construction in the MAE objective).

For each spectral feature (FWHM, V/R, Δv, depth, peak_I):
  1. Ridge Regression (5-fold CV) on z -> feature
  2. Same probe on PCA(10), PCA(32), PCA(50) of normalized flux -> baselines
  3. R² ratio between z and PCA baselines

R²(z) >> R²(PCA) -> z encodes a rich learned representation.
R²(z) ≈ R²(PCA) -> z adds nothing over linear compression.
R²(z) < R²(PCA) -> z is worse than linear dimensionality reduction.

Usage:
    python probe_features.py
    python probe_features.py --features ew fwhm vr_ratio
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

# Ensure bess_bench/ is on sys.path so ``from stats.wandb_helpers import …`` works
_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))
from stats.wandb_helpers import add_cli_args, finish, init_run, log_artifact  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
# Default paths point to the canonical reference layout (see REPRODUCE.md):
#   reference_results/   ← committed JSONs / .pt are regenerated locally
#   results_reproduced/  ← reproduction outputs (override via CLI flags)
EMBEDDINGS_PATH = str(_BENCH_ROOT / "reference_results" / "embeddings" / "star_data.pt")
FEATURES_PATH = str(_BENCH_ROOT / "reference_results" / "features" / "spectral_features.pt")
RESULTS_DIR = _BENCH_ROOT / "reference_results" / "downstream"
FIGURES_DIR = _BENCH_ROOT / "results_reproduced" / "figures"

FEATURE_NAMES = ["ew", "fwhm", "vr_ratio", "delta_v", "central_depth", "peak_intensity"]
FEATURE_LABELS = {
    "ew": "EW (Å)",
    "fwhm": "FWHM (Å)",
    "vr_ratio": "V/R ratio",
    "delta_v": "Δv (km/s)",
    "central_depth": "Central Depth",
    "peak_intensity": "Peak Intensity",
}

PCA_DIMS = [10, 32, 50]
CV_FOLDS = 5
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ══════════════════════════════════════════════════════════════════════════════
# ALIGN EMBEDDINGS ↔ FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def align_data(emb_data, feat_data, mjd_tol=0.01):
    """
    Align embeddings and features by (star_name, mjd).

    Returns (embeddings, features_dict, flux_norm, metadata).
    """
    all_embeddings = []
    all_features = {name: [] for name in FEATURE_NAMES}
    all_flux = []
    all_meta = []  # (star_name, mjd)

    common_stars = set(emb_data.keys()) & set(feat_data.keys())
    print(f"  Common stars: {len(common_stars)} "
          f"(emb: {len(emb_data)}, feat: {len(feat_data)})")

    n_matched = 0
    n_unmatched = 0

    for star in sorted(common_stars):
        emb_star = emb_data[star]
        feat_star = feat_data[star]

        emb_mjds = emb_star["mjds"].numpy()
        feat_mjds = feat_star["mjds"].numpy()
        emb_vecs = emb_star["embeddings"].numpy()   # [N, 128]
        flux_vecs = feat_star["flux_norm"].numpy()   # [M, 128]

        # For each embedding MJD find the matching feature observation
        for i, emjd in enumerate(emb_mjds):
            diffs = np.abs(feat_mjds - emjd)
            best = np.argmin(diffs)
            if diffs[best] < mjd_tol:
                all_embeddings.append(emb_vecs[i])
                for name in FEATURE_NAMES:
                    val = feat_star[name][best].item()
                    all_features[name].append(val)
                all_flux.append(flux_vecs[best])
                all_meta.append((star, emjd))
                n_matched += 1
            else:
                n_unmatched += 1

    print(f"  Matches          : {n_matched:,} matched, {n_unmatched:,} unmatched")

    embeddings = np.array(all_embeddings, dtype=np.float32)
    features = {name: np.array(vals, dtype=np.float32) for name, vals in all_features.items()}
    flux = np.array(all_flux, dtype=np.float32)

    return embeddings, features, flux, all_meta


# ══════════════════════════════════════════════════════════════════════════════
# PROBE : RIDGE CV
# ══════════════════════════════════════════════════════════════════════════════

def probe_ridge_cv(X, y, n_folds=CV_FOLDS, alphas=RIDGE_ALPHAS, seed=42):
    """
    Linear probe using Ridge Regression with cross-validation.

    Uses a Pipeline(StandardScaler, RidgeCV) to prevent data leakage
    between folds (the scaler is fit only on each fold's training split).
    CV splitting uses ``KFold(shuffle=True, random_state=seed)`` to make
    results reproducible and seed-dependent (required for multi-seed runs).

    Returns mean R² (CV) and std.
    """
    # Filtrer les NaN/Inf dans y
    valid = np.isfinite(y)
    X_val = X[valid]
    y_val = y[valid]

    if len(y_val) < n_folds * 5:
        return float("nan"), float("nan"), len(y_val)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas)),
    ])

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X_val, y_val, cv=cv, scoring="r2")

    return float(np.mean(scores)), float(np.std(scores)), len(y_val)


def probe_mlp_cv(X, y, n_folds=CV_FOLDS, hidden_layers=(64, 32), seed=42):
    """
    Non-linear probe using MLP Regressor with cross-validation.

    Uses a Pipeline(StandardScaler, MLPRegressor) to test whether
    information is encoded non-linearly in z. V/R ratio in particular:
    Ridge gives R²=0.16; an MLP might reveal non-linear encoding.
    """
    valid = np.isfinite(y)
    X_val = X[valid]
    y_val = y[valid]

    if len(y_val) < n_folds * 5:
        return float("nan"), float("nan"), len(y_val)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
            learning_rate_init=1e-3,
        )),
    ])

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X_val, y_val, cv=cv, scoring="r2")
    return float(np.mean(scores)), float(np.std(scores)), len(y_val)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global RESULTS_DIR, FIGURES_DIR
    parser = argparse.ArgumentParser(
        description="T1 (SpecProbe): linear probes z -> spectral features"
    )
    parser.add_argument("--embeddings", type=str, default=EMBEDDINGS_PATH)
    parser.add_argument("--features", type=str, default=FEATURES_PATH)
    parser.add_argument("--feature_list", nargs="+", default=FEATURE_NAMES,
                        help="Which features to probe")
    parser.add_argument("--mlp", action="store_true",
                        help="Add a non-linear MLP probe (§10.7)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for KFold shuffle and MLP init (default: 42)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output JSON (e.g. '_seed42')")
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR),
                        help="Directory for output JSON (default: reference_results/downstream)")
    parser.add_argument("--figures_dir", type=str, default=str(FIGURES_DIR),
                        help="Directory for figures (default: results_reproduced/figures)")
    add_cli_args(parser)
    args = parser.parse_args()

    RESULTS_DIR = Path(args.output_dir)
    FIGURES_DIR = Path(args.figures_dir)

    np.random.seed(args.seed)
    try:
        torch.manual_seed(args.seed)
    except Exception:
        pass

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    suffix = args.output_suffix or f"_seed{args.seed}"

    # ── Load data ──
    print("=" * 60)
    print(f"T1 (SpecProbe): LINEAR PROBES z -> SPECTRAL FEATURES  (seed={args.seed})")
    print("=" * 60)

    print("\n1. Loading embeddings...")
    emb_data = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    print(f"   {len(emb_data)} stars")

    print("2. Loading spectral features...")
    feat_data = torch.load(args.features, map_location="cpu", weights_only=False)
    print(f"   {len(feat_data)} stars")

    print("3. Aligning embeddings ↔ features...")
    embeddings, features, flux_norm, meta = align_data(emb_data, feat_data)
    print(f"   Embeddings matrix  : {embeddings.shape}")
    print(f"   Flux matrix        : {flux_norm.shape}")

    # ── Prepare representations ──
    print("\n4. Computing PCA baselines...")
    representations = {"z (128D)": embeddings}

    for n_comp in PCA_DIMS:
        pca = PCA(n_components=n_comp)
        flux_pca = pca.fit_transform(flux_norm)
        explained = pca.explained_variance_ratio_.sum()
        representations[f"PCA({n_comp})"] = flux_pca
        print(f"   PCA({n_comp:2d}) → explained variance: {explained:.3f}")

    # Baseline raw-flux (no PCA): Ridge directly on normalized flux.
    # Isolates the effect of PCA compression vs. learned representation z.
    representations["raw-flux-128D"] = flux_norm
    print(f"   raw-flux-128D -> raw normalized flux, shape={flux_norm.shape}")

    # Baseline handcrafted-5D-LOO: for each target feature, predict from the
    # 5 *other* handcrafted features (leave-one-out, avoids circularity).
    # Handled specially in the probing loop below (X depends on the target).
    HANDCRAFTED_LOO_KEY = "handcrafted-5D-LOO"
    representations[HANDCRAFTED_LOO_KEY] = "__loo_marker__"
    print(f"   {HANDCRAFTED_LOO_KEY} -> 5 other physical features (LOO)")

    # ── Probes ──
    print("\n5. Ridge CV probes per feature...")
    results = {}  # {feature: {repr_name: {r2_mean, r2_std, n_valid}}}

    for feat_name in args.feature_list:
        if feat_name not in features:
            print(f"   ⚠ Feature '{feat_name}' not found, skipping")
            continue

        y = features[feat_name]
        n_finite = np.isfinite(y).sum()
        print(f"\n   ── {feat_name} ({n_finite:,} finite values out of {len(y):,}) ──")

        results[feat_name] = {}

        for repr_name, X in representations.items():
            # Special case: handcrafted-5D-LOO → X = [N, 5] matrix of the 5
            # other features (all except the target), with NaN→0 imputation.
            if isinstance(X, str) and X == "__loo_marker__":
                other_feats = [f for f in FEATURE_NAMES if f != feat_name]
                cols = []
                for of in other_feats:
                    col = features[of].astype(np.float64).copy()
                    # 0-fill NaN imputation to stay comparable with PCA(10)
                    # (which has no NaN problem); the final NaN mask on y is
                    # applied inside probe_ridge_cv.
                    mask = ~np.isfinite(col)
                    if mask.any():
                        col[mask] = 0.0
                    cols.append(col)
                X = np.stack(cols, axis=1).astype(np.float32)
            r2_mean, r2_std, n_valid = probe_ridge_cv(X, y, seed=args.seed)
            results[feat_name][repr_name] = {
                "r2_mean": r2_mean,
                "r2_std": r2_std,
                "n_valid": n_valid,
            }
            print(f"     {repr_name:15s} → R² = {r2_mean:+.4f} ± {r2_std:.4f}  (n={n_valid})")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    repr_names = list(representations.keys())
    header = f"{'Feature':18s} | " + " | ".join(f"{r:15s}" for r in repr_names)
    print(header)
    print("-" * len(header))

    for feat_name in args.feature_list:
        if feat_name not in results:
            continue
        row = f"{feat_name:18s} | "
        for rn in repr_names:
            r2 = results[feat_name][rn]["r2_mean"]
            std = results[feat_name][rn]["r2_std"]
            if np.isnan(r2):
                row += f"{'N/A':>15s} | "
            else:
                row += f"{r2:+.4f}±{std:.3f}  | "
        print(row)

    # ── Gain z vs PCA(50) ──
    print("\n\nGAIN z vs PCA(50):")
    for feat_name in args.feature_list:
        if feat_name not in results:
            continue
        r2_z = results[feat_name]["z (128D)"]["r2_mean"]
        r2_pca50 = results[feat_name][f"PCA(50)"]["r2_mean"]
        if np.isnan(r2_z) or np.isnan(r2_pca50):
            print(f"  {feat_name:18s} : N/A")
        else:
            gain = r2_z - r2_pca50
            print(f"  {feat_name:18s} : ΔR² = {gain:+.4f}  "
                  f"({'z better' if gain > 0 else 'PCA better'})")

    # ── Non-linear MLP probes (§10.7) ──
    mlp_results = {}
    if args.mlp:
        print("\n" + "=" * 70)
        print("NON-LINEAR MLP PROBES (§10.7)")
        print("=" * 70)

        for feat_name in args.feature_list:
            if feat_name not in features:
                continue

            y = features[feat_name]
            n_finite = np.isfinite(y).sum()
            print(f"\n   ── {feat_name} ({n_finite:,} finite values) ──")

            mlp_results[feat_name] = {}

            for repr_name, X in representations.items():
                if isinstance(X, str) and X == "__loo_marker__":
                    other_feats = [f for f in FEATURE_NAMES if f != feat_name]
                    cols = []
                    for of in other_feats:
                        col = features[of].astype(np.float64).copy()
                        mask = ~np.isfinite(col)
                        if mask.any():
                            col[mask] = 0.0
                        cols.append(col)
                    X = np.stack(cols, axis=1).astype(np.float32)
                r2_mean, r2_std, n_valid = probe_mlp_cv(X, y, seed=args.seed)
                mlp_results[feat_name][repr_name] = {
                    "r2_mean": r2_mean,
                    "r2_std": r2_std,
                    "n_valid": n_valid,
                }
                # Comparer with Ridge
                ridge_r2 = results.get(feat_name, {}).get(repr_name, {}).get("r2_mean", float("nan"))
                gain_str = ""
                if np.isfinite(r2_mean) and np.isfinite(ridge_r2):
                    gain = r2_mean - ridge_r2
                    gain_str = f"  (Δ vs Ridge: {gain:+.4f})"
                print(f"     {repr_name:15s} -> MLP R² = {r2_mean:+.4f} ± {r2_std:.4f}{gain_str}")

        # Table MLP vs Ridge
        print("\n\nCOMPARAISON RIDGE vs MLP (z uniquement) :")
        print(f"  {'Feature':18s} | {'Ridge R²':>10s} | {'MLP R²':>10s} | {'ΔMLP-Ridge':>12s}")
        print("  " + "-" * 60)
        for feat_name in args.feature_list:
            if feat_name not in results or feat_name not in mlp_results:
                continue
            r2_ridge = results[feat_name]["z (128D)"]["r2_mean"]
            r2_mlp = mlp_results[feat_name]["z (128D)"]["r2_mean"]
            if np.isfinite(r2_ridge) and np.isfinite(r2_mlp):
                delta = r2_mlp - r2_ridge
                print(f"  {feat_name:18s} | {r2_ridge:>+10.4f} | {r2_mlp:>+10.4f} | {delta:>+12.4f}")

    # ── Save results ──
    results_path = RESULTS_DIR / f"probe_features_results{suffix}.json"
    payload = {
        "seed": args.seed,
        "n_folds": CV_FOLDS,
        "cv_shuffle": True,
        "features": args.feature_list,
        "representations": list(representations.keys()),
        "results": results,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Ridge results saved: {results_path}")

    if mlp_results:
        mlp_path = RESULTS_DIR / f"probe_features_mlp_results{suffix}.json"
        with open(mlp_path, "w") as f:
            json.dump({"seed": args.seed, "results": mlp_results}, f, indent=2)
        print(f"  MLP results saved: {mlp_path}")

    # ── W&B logging ──
    run_tags = (args.wandb_tags or []) + ["phase3-t1-multiseed"]
    run = init_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"t1_probe_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=list(dict.fromkeys(run_tags)),
        mode=args.wandb_mode,
        group=args.wandb_group or "t1_multiseed",
        config={
            "seed": args.seed,
            "phase": "3",
            "task": "T1_probe_features",
            "n_folds": CV_FOLDS,
            "features": args.feature_list,
            "representations": list(representations.keys()),
            "n_aligned_pairs": int(embeddings.shape[0]),
        },
    )
    flat_metrics = {}
    for feat_name, by_repr in results.items():
        for rn, stats in by_repr.items():
            rn_clean = rn.replace(" ", "_").replace("(", "").replace(")", "")
            flat_metrics[f"r2/{feat_name}/{rn_clean}"] = stats["r2_mean"]
            flat_metrics[f"r2_std/{feat_name}/{rn_clean}"] = stats["r2_std"]
    run.log(flat_metrics)
    log_artifact(run, results_path, name=f"t1_probe_seed{args.seed}",
                 type="probe_results",
                 description=f"T1 probe results (seed={args.seed})")
    finish(run)

    # ── Figures ──
    _plot_probe_comparison(results, repr_names)
    _plot_scatter_best(embeddings, features, flux_norm)


def _plot_probe_comparison(results, repr_names):
    """Comparative bar plot of R² per feature and representation."""
    feat_names = [f for f in FEATURE_NAMES if f in results]
    n_feat = len(feat_names)
    n_repr = len(repr_names)

    if n_feat == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_feat)
    width = 0.8 / n_repr
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for j, rn in enumerate(repr_names):
        r2_vals = []
        r2_stds = []
        for fn in feat_names:
            r2 = results[fn][rn]["r2_mean"]
            std = results[fn][rn]["r2_std"]
            r2_vals.append(r2 if np.isfinite(r2) else 0)
            r2_stds.append(std if np.isfinite(std) else 0)

        offset = (j - n_repr / 2 + 0.5) * width
        bars = ax.bar(x + offset, r2_vals, width,
                      yerr=r2_stds, capsize=3,
                      label=rn, color=colors[j % len(colors)], alpha=0.85)

    ax.set_ylabel("R² (5-fold CV)", fontsize=12)
    ax.set_title("T1 (SpecProbe): linear probes z -> spectral features", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS.get(f, f) for f in feat_names], fontsize=10)
    ax.legend(fontsize=11)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(-0.2, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "t1_probe_comparison.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Figure saved: {fig_path}")


def _plot_scatter_best(embeddings, features, flux_norm):
    """Scatter plot: Ridge prediction vs true value for the best feature."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict

    # Find the best feature (most finite values, in emission)
    best_feat = None
    best_count = 0
    for fname in ["fwhm", "vr_ratio", "delta_v", "central_depth"]:
        if fname in features:
            n_ok = np.isfinite(features[fname]).sum()
            if n_ok > best_count:
                best_count = n_ok
                best_feat = fname

    if best_feat is None or best_count < 50:
        return

    y = features[best_feat]
    valid = np.isfinite(y)
    X = embeddings[valid]
    y = y[valid]

    # Pipeline to prevent scaler leakage
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=5)),
    ])
    y_pred = cross_val_predict(pipe, X, y, cv=5)

    r2 = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, y_pred, alpha=0.15, s=10, c="#2196F3")
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y = ŷ")
    ax.set_xlabel(f"True ({FEATURE_LABELS.get(best_feat, best_feat)})", fontsize=12)
    ax.set_ylabel("Predicted (Ridge on z)", fontsize=12)
    ax.set_title(f"T1 (SpecProbe): z -> {best_feat}   R² = {r2:.4f}", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / f"t1_probe_scatter_{best_feat}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
