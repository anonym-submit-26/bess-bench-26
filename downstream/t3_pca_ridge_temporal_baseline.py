"""
t3_pca_ridge_temporal_baseline.py — PCA+Ridge baseline for T3 temporal forecasting.

Classical baseline for EW(Hα)(t+1) forecasting:
  - Representation : PCA(K) of normalized flux (instead of learned embeddings)
  - Prediction    : Ridge regression on the K most recent PCA representations
  - Comparison    : persistence + zero-shot TS-FMs (Chronos-Bolt, TimesFM)

Metrics produced:
  - MAE model vs persistence
  - R² model
  - MAE ratio (model/persist)

Usage:
    python t3_pca_ridge_temporal_baseline.py
    python t3_pca_ridge_temporal_baseline.py --n_pca 50 --context 10
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_ROOT = SCRIPT_DIR.parent
EMBEDDINGS_PATH = _BENCH_ROOT / "reference_results" / "embeddings" / "star_data.pt"
FEATURES_PATH = _BENCH_ROOT / "reference_results" / "features" / "spectral_features.pt"
RESULTS_DIR = _BENCH_ROOT / "reference_results" / "downstream"

# Allow ``from stats.wandb_helpers import …``
_BENCH_ROOT = SCRIPT_DIR.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))
from stats.wandb_helpers import add_cli_args, finish, init_run, log_artifact  # noqa: E402

# Canonical T3 train/test cutoff (≈ 2020-12-14)
CUTOFF_MJD = 59215.0


def load_aligned_data(emb_path, feat_path, mjd_tol=0.01):
    """
    Load and align embeddings, normalized flux, and EW by (star, mjd).

    Returns a dict per star: {star: {mjds, flux, embeddings, ews}}
    """
    emb_data = torch.load(emb_path, weights_only=False, map_location="cpu")
    feat_data = torch.load(feat_path, weights_only=False, map_location="cpu")
    
    aligned = {}
    n_total = 0
    
    common_stars = set(emb_data.keys()) & set(feat_data.keys())
    
    for star in sorted(common_stars):
        emb_star = emb_data[star]
        feat_star = feat_data[star]
        
        emb_mjds = emb_star["mjds"].numpy()
        feat_mjds = feat_star["mjds"].numpy()
        
        emb_vecs = emb_star["embeddings"].numpy()  # (N, 128)
        flux_vecs = feat_star["flux_norm"].numpy()  # (N, 128)
        ews = feat_star["ew"].numpy()               # (N,)
        
        # Align by MJD
        idx_emb = []
        idx_feat = []
        for i, emjd in enumerate(emb_mjds):
            diffs = np.abs(feat_mjds - emjd)
            best = np.argmin(diffs)
            if diffs[best] < mjd_tol:
                idx_emb.append(i)
                idx_feat.append(best)
        
        if len(idx_emb) < 3:
            continue
        
        mjds = emb_mjds[idx_emb]
        flux = flux_vecs[idx_feat]
        embeddings = emb_vecs[idx_emb]
        ew = ews[idx_feat]
        
        # Filter out non-finite EW values
        valid = np.isfinite(ew)
        if valid.sum() < 3:
            continue

        # Sort by MJD
        order = np.argsort(mjds[valid])
        
        aligned[star] = {
            "mjds": mjds[valid][order],
            "flux": flux[valid][order],
            "embeddings": embeddings[valid][order],
            "ews": ew[valid][order],
        }
        n_total += valid.sum()
    
    print(f"  Aligned stars: {len(aligned)}, spectra: {n_total:,}")
    return aligned


def evaluate_pca_ridge(aligned_data, n_pca=10, context_len=5):
    """
    Evaluate the PCA+Ridge baseline for EW(t+1) prediction.

    For each star:
      1. Train/test split by cutoff MJD
      2. PCA fitted on TRAIN flux only (no leakage)
      3. Ridge fitted on TRAIN context pairs only
      4. EW(t+1) prediction on TEST pairs

    Returns a dict of metrics.
    """
    # --- Step 1: Collect all train flux to fit global PCA ---
    all_flux_train = []
    for star, data in aligned_data.items():
        train_mask = data["mjds"] <= CUTOFF_MJD
        if train_mask.any():
            all_flux_train.append(data["flux"][train_mask])
    
    all_flux_train = np.concatenate(all_flux_train, axis=0)
    print(f"  Train flux for PCA: {all_flux_train.shape}")

    pca = PCA(n_components=n_pca, random_state=42)
    pca.fit(all_flux_train)
    print(f"  PCA({n_pca}) explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # --- Step 2: Build (context, target) pairs ---
    train_X, train_y = [], []
    test_X, test_y, test_ew_current = [], [], []
    test_dt = []
    test_stars = []
    
    for star, data in aligned_data.items():
        mjds = data["mjds"]
        flux = data["flux"]
        ews = data["ews"]
        
        # Apply PCA to the full flux array
        pca_repr = pca.transform(flux)  # (N, n_pca)

        train_mask = mjds <= CUTOFF_MJD
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(~train_mask)[0]

        # Build train pairs
        for i in range(context_len, len(train_idx)):
            # Context = concatenation of the last context_len PCA representations
            ctx_indices = train_idx[i-context_len:i]
            ctx = pca_repr[ctx_indices].flatten()  # (context_len * n_pca,)
            target = ews[train_idx[i]]
            train_X.append(ctx)
            train_y.append(target)
        
        # Build test pairs
        # Use the full (train + test) context to predict test points
        all_idx = np.concatenate([train_idx, test_idx])
        sorted_all = np.sort(all_idx)

        for j, ti in enumerate(test_idx):
            # Find the position of ti in the sorted series
            pos = np.searchsorted(sorted_all, ti)
            if pos < context_len:
                continue  # not enough context
            
            ctx_indices = sorted_all[pos-context_len:pos]
            ctx = pca_repr[ctx_indices].flatten()
            target = ews[ti]
            current_ew = ews[sorted_all[pos-1]]  # dernier EW du contexte
            dt = float(mjds[ti] - mjds[sorted_all[pos-1]])
            
            test_X.append(ctx)
            test_y.append(target)
            test_ew_current.append(current_ew)
            test_dt.append(dt)
            test_stars.append(star)
    
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)
    test_ew_current = np.array(test_ew_current)
    test_dt = np.array(test_dt)
    
    print(f"  Train pairs: {len(train_X)}, Test pairs: {len(test_X)}")

    if len(train_X) == 0 or len(test_X) == 0:
        print("  ! Insufficient data to evaluate")
        return {}
    
    # --- Step 3: Fit Ridge (with internal scaling) ---
    alphas = np.logspace(-3, 5, 20)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas)),
    ])
    model.fit(train_X, train_y)
    
    # --- Step 4: Prediction and metrics ---
    pred_y = model.predict(test_X)

    # Global metrics
    mae_model = mean_absolute_error(test_y, pred_y)
    mae_persist = mean_absolute_error(test_y, test_ew_current)
    r2_model = r2_score(test_y, pred_y)
    r2_persist = r2_score(test_y, test_ew_current)
    ratio_mae = mae_model / max(mae_persist, 1e-10)
    corr_model = float(np.corrcoef(test_y, pred_y)[0, 1])
    corr_persist = float(np.corrcoef(test_y, test_ew_current)[0, 1])
    
    # Per-star metrics
    star_metrics = defaultdict(lambda: {"y": [], "pred": [], "persist": [], "dt": []})
    for i, star in enumerate(test_stars):
        star_metrics[star]["y"].append(test_y[i])
        star_metrics[star]["pred"].append(pred_y[i])
        star_metrics[star]["persist"].append(test_ew_current[i])
        star_metrics[star]["dt"].append(test_dt[i])
    
    n_stars_beating = 0
    per_star_ratios = []
    for star, m in star_metrics.items():
        y = np.array(m["y"])
        p = np.array(m["pred"])
        pers = np.array(m["persist"])
        mae_m = mean_absolute_error(y, p)
        mae_p = mean_absolute_error(y, pers)
        ratio = mae_m / max(mae_p, 1e-10)
        per_star_ratios.append(ratio)
        if ratio < 1.0:
            n_stars_beating += 1
    
    n_test_stars = len(star_metrics)
    
    # Per-horizon metrics (same bins as stratified_horizon.py)
    dt_bins = [
        {"name": "< 7d",     "min": 0,    "max": 7},
        {"name": "7-30d",    "min": 7,    "max": 30},
        {"name": "30-90d",   "min": 30,   "max": 90},
        {"name": "90-365d",  "min": 90,   "max": 365},
        {"name": "> 365d",   "min": 365,  "max": 1e6},
    ]
    
    bin_results = {}
    for b in dt_bins:
        mask = (test_dt >= b["min"]) & (test_dt < b["max"])
        n = mask.sum()
        if n < 5:
            bin_results[b["name"]] = {"n": int(n), "mae_model": None, "mae_persist": None, "ratio": None}
            continue
        
        mae_m = mean_absolute_error(test_y[mask], pred_y[mask])
        mae_p = mean_absolute_error(test_y[mask], test_ew_current[mask])
        bin_results[b["name"]] = {
            "n": int(n),
            "mae_model": round(float(mae_m), 4),
            "mae_persist": round(float(mae_p), 4),
            "ratio": round(float(mae_m / max(mae_p, 1e-10)), 4),
        }
    
    results = {
        "config": {
            "n_pca": n_pca,
            "context_len": context_len,
            "cutoff_mjd": CUTOFF_MJD,
            "n_train": int(len(train_X)),
            "n_test": int(len(test_X)),
            "pca_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
            "ridge_alpha": float(model.named_steps["ridge"].alpha_),
        },
        "global": {
            "mae_model": round(float(mae_model), 4),
            "mae_persist": round(float(mae_persist), 4),
            "ratio_mae": round(float(ratio_mae), 4),
            "r2_model": round(float(r2_model), 4),
            "r2_persist": round(float(r2_persist), 4),
            "corr_model": round(float(corr_model), 4),
            "corr_persist": round(float(corr_persist), 4),
            "n_test_stars": n_test_stars,
            "stars_beating_persist": n_stars_beating,
            "pct_beating_persist": round(100 * n_stars_beating / max(n_test_stars, 1), 1),
            "median_per_star_ratio": round(float(np.median(per_star_ratios)), 4),
        },
        "by_horizon": bin_results,
        "predictions": {
            "y_true": test_y.astype(np.float32).tolist(),
            "y_pred_model": pred_y.astype(np.float32).tolist(),
            "y_pred_persist": test_ew_current.astype(np.float32).tolist(),
            "dt_days": test_dt.astype(np.float32).tolist(),
            "star_names": list(test_stars),
        },
    }
    
    return results


def main():
    global EMBEDDINGS_PATH, FEATURES_PATH, RESULTS_DIR
    parser = argparse.ArgumentParser(
        description="PCA+Ridge temporal baseline for EW prediction"
    )
    parser.add_argument("--n_pca", type=int, default=10,
                        help="Number of PCA components (default: 10)")
    parser.add_argument("--context", type=int, default=5,
                        help="Number of past observations as context (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed (logged only; PCA+Ridge pipeline is deterministic)")
    parser.add_argument("--embeddings", type=str, default=str(EMBEDDINGS_PATH),
                        help="Path to star_data.pt embeddings (default: bench-relative)")
    parser.add_argument("--features", type=str, default=str(FEATURES_PATH),
                        help="Path to spectral_features.pt (default: results/)")
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR),
                        help="Directory for output JSON (default: <script>/results)")
    add_cli_args(parser)
    args = parser.parse_args()

    EMBEDDINGS_PATH = Path(args.embeddings)
    FEATURES_PATH = Path(args.features)
    RESULTS_DIR = Path(args.output_dir)
    
    print("=" * 70)
    print("T3 BASELINE: PCA+Ridge -> EW(t+1)")
    print("=" * 70)

    print(f"\n1. Loading and aligning data...")
    aligned = load_aligned_data(EMBEDDINGS_PATH, FEATURES_PATH)

    # Test multiple configurations
    configs = [
        (args.n_pca, args.context),
        (10, 5),
        (10, 10),
        (50, 5),
        (50, 10),
    ]
    # Deduplicate
    seen = set()
    unique_configs = []
    for c in configs:
        if c not in seen:
            seen.add(c)
            unique_configs.append(c)
    
    all_results = {}
    
    for n_pca, ctx in unique_configs:
        print(f"\n{'='*70}")
        print(f"Configuration: PCA({n_pca}), context={ctx}")
        print(f"{'='*70}")

        results = evaluate_pca_ridge(aligned, n_pca=n_pca, context_len=ctx)
        key = f"pca{n_pca}_ctx{ctx}"
        all_results[key] = results
        
        if results and "global" in results:
            g = results["global"]
            print(f"\n  --- Global results ---")
            print(f"  Model MAE        : {g['mae_model']:.4f}")
            print(f"  Persistence MAE  : {g['mae_persist']:.4f}")
            print(f"  MAE ratio        : {g['ratio_mae']:.4f}  ({'✓' if g['ratio_mae'] < 1 else '✗'})")
            print(f"  Model R²         : {g['r2_model']:.4f}")
            print(f"  Persistence R²   : {g['r2_persist']:.4f}")
            print(f"  Stars > persist  : {g['stars_beating_persist']}/{g['n_test_stars']} ({g['pct_beating_persist']:.1f}%)")
            print(f"  Median ratio/star: {g['median_per_star_ratio']:.4f}")

            print(f"\n  --- By horizon ---")
            for bname, bdata in results.get("by_horizon", {}).items():
                if bdata["ratio"] is not None:
                    status = "✓" if bdata["ratio"] < 1 else "✗"
                    print(f"    {bname:>10s} : MAE={bdata['mae_model']:.3f} "
                          f"persist={bdata['mae_persist']:.3f} "
                          f"ratio={bdata['ratio']:.3f} {status} (n={bdata['n']})")
                else:
                    print(f"    {bname:>10s} : n={bdata['n']} (too few)")

    # Save
    output_path = RESULTS_DIR / "t3_pca_ridge_temporal_baseline.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {output_path}")

    # ── W&B logging ──
    tags = (args.wandb_tags or []) + ["phase2-t3-baseline"]
    run = init_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"t3_pca_ridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=list(dict.fromkeys(tags)),
        mode=args.wandb_mode,
        group=args.wandb_group or "t3_baseline",
        config={
            "seed": args.seed,
            "phase": "2",
            "task": "T3_pca_ridge_baseline",
            "n_pca_values": [c[0] for c in unique_configs],
            "context_values": [c[1] for c in unique_configs],
            "cutoff_mjd": CUTOFF_MJD,
        },
    )
    wb_metrics = {}
    for key, res in all_results.items():
        if res and "global" in res:
            g = res["global"]
            for m in ("ratio_mae", "mae_model", "mae_persist", "r2_model"):
                wb_metrics[f"{key}/{m}"] = g.get(m)
            wb_metrics[f"{key}/pct_beating_persist"] = g.get("pct_beating_persist")
    run.log(wb_metrics)
    log_artifact(run, output_path, name="t3_pca_ridge_baseline",
                 type="baseline_results",
                 description="PCA+Ridge temporal baseline with per-prediction arrays")
    finish(run)


if __name__ == "__main__":
    main()
