"""
t3_z_ridge_temporal_baseline.py — Encoder embeddings + Ridge for T3 forecasting.

Same protocol as t3_pca_ridge_temporal_baseline.py but the per-spectrum
representation is the **learned encoder embedding z (128D)** instead of
PCA(K) of the raw flux. Single deterministic encoder (seed=42); uncertainty
is reported as a 10 000-sample bootstrap 95% CI on the test predictions,
identical to the protocol used for PCA+Ridge and the TS-FM baselines
(see ``stats/results/t3_bootstrap_cis.json``).

Outputs results/t3_z_ridge_temporal_baseline_ctx{context}.json.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
_BENCH_ROOT = SCRIPT_DIR.parent
# Default paths point to the canonical reference layout (see REPRODUCE.md):
#   reference_results/   <- committed JSONs / .pt are regenerated locally
#   results_reproduced/  <- reproduction outputs (override via CLI flags)
RESULTS_DIR = _BENCH_ROOT / "reference_results" / "downstream"
EMBEDDINGS_PATH = _BENCH_ROOT / "reference_results" / "embeddings" / "star_data.pt"
FEATURES_PATH = _BENCH_ROOT / "reference_results" / "features" / "spectral_features.pt"

CUTOFF_MJD = 59215.0
SEEDS = [42]
N_BOOT = 10_000
BOOT_SEED = 42


def load_aligned(emb_path: Path, feat_path: Path, mjd_tol: float = 0.01):
    emb_data = torch.load(emb_path, weights_only=False, map_location="cpu")
    feat_data = torch.load(feat_path, weights_only=False, map_location="cpu")
    aligned = {}
    common = set(emb_data) & set(feat_data)
    for star in sorted(common):
        e = emb_data[star]
        f = feat_data[star]
        emb_mjds = e["mjds"].numpy()
        feat_mjds = f["mjds"].numpy()
        emb_vecs = e["embeddings"].numpy()
        ews = f["ew"].numpy()
        ie, ife = [], []
        for i, m in enumerate(emb_mjds):
            d = np.abs(feat_mjds - m)
            j = int(np.argmin(d))
            if d[j] < mjd_tol:
                ie.append(i)
                ife.append(j)
        if len(ie) < 3:
            continue
        mjds = emb_mjds[ie]
        Z = emb_vecs[ie]
        ew = ews[ife]
        valid = np.isfinite(ew)
        if valid.sum() < 3:
            continue
        order = np.argsort(mjds[valid])
        aligned[star] = {
            "mjds": mjds[valid][order],
            "Z": Z[valid][order],
            "ews": ew[valid][order],
        }
    return aligned


def evaluate_z_ridge(aligned, context_len: int = 5):
    train_X, train_y = [], []
    test_X, test_y, test_cur, test_dt, test_stars = [], [], [], [], []

    for star, d in aligned.items():
        mjds = d["mjds"]
        Z = d["Z"]
        ews = d["ews"]
        train_idx = np.where(mjds <= CUTOFF_MJD)[0]
        test_idx = np.where(mjds > CUTOFF_MJD)[0]

        for i in range(context_len, len(train_idx)):
            ctx = Z[train_idx[i - context_len : i]].flatten()
            train_X.append(ctx)
            train_y.append(ews[train_idx[i]])

        all_idx = np.sort(np.concatenate([train_idx, test_idx]))
        for ti in test_idx:
            pos = int(np.searchsorted(all_idx, ti))
            if pos < context_len:
                continue
            ctx = Z[all_idx[pos - context_len : pos]].flatten()
            test_X.append(ctx)
            test_y.append(ews[ti])
            test_cur.append(ews[all_idx[pos - 1]])
            test_dt.append(float(mjds[ti] - mjds[all_idx[pos - 1]]))
            test_stars.append(star)

    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)
    test_X = np.asarray(test_X)
    test_y = np.asarray(test_y)
    test_cur = np.asarray(test_cur)
    test_dt = np.asarray(test_dt)

    if len(train_X) == 0 or len(test_X) == 0:
        return {}

    alphas = np.logspace(-3, 5, 20)
    model = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=alphas))])
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)

    mae_m = float(mean_absolute_error(test_y, pred_y))
    mae_p = float(mean_absolute_error(test_y, test_cur))
    r2_m = float(r2_score(test_y, pred_y))
    r2_p = float(r2_score(test_y, test_cur))
    ratio = mae_m / max(mae_p, 1e-10)

    star_metrics = defaultdict(lambda: {"y": [], "p": [], "c": []})
    for i, s in enumerate(test_stars):
        star_metrics[s]["y"].append(test_y[i])
        star_metrics[s]["p"].append(pred_y[i])
        star_metrics[s]["c"].append(test_cur[i])
    n_beat = 0
    per_star_ratios = []
    for s, m in star_metrics.items():
        y = np.array(m["y"]); p = np.array(m["p"]); c = np.array(m["c"])
        rm = mean_absolute_error(y, p) / max(mean_absolute_error(y, c), 1e-10)
        per_star_ratios.append(rm)
        if rm < 1.0:
            n_beat += 1

    bins = [(0, 7, "< 7d"), (7, 30, "7-30d"), (30, 90, "30-90d"),
            (90, 365, "90-365d"), (365, 1e6, "> 365d")]
    by_h = {}
    for lo, hi, name in bins:
        msk = (test_dt >= lo) & (test_dt < hi)
        n = int(msk.sum())
        if n < 5:
            by_h[name] = {"n": n, "mae_model": None, "mae_persist": None, "ratio": None}
            continue
        mm = float(mean_absolute_error(test_y[msk], pred_y[msk]))
        mp = float(mean_absolute_error(test_y[msk], test_cur[msk]))
        by_h[name] = {"n": n, "mae_model": round(mm, 4),
                      "mae_persist": round(mp, 4),
                      "ratio": round(mm / max(mp, 1e-10), 4)}

    # Bootstrap CI on ratio_mae by resampling the test predictions
    rng_boot = np.random.default_rng(BOOT_SEED)
    n = len(test_y)
    boot_ratios = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        idx = rng_boot.integers(0, n, n)
        mm = mean_absolute_error(test_y[idx], pred_y[idx])
        mp = mean_absolute_error(test_y[idx], test_cur[idx])
        boot_ratios[b] = mm / max(mp, 1e-10)
    ci_lo, ci_hi = np.percentile(boot_ratios, [2.5, 97.5])
    frac_lt_1 = float((boot_ratios < 1.0).mean())

    return {
        "config": {
            "context_len": context_len,
            "cutoff_mjd": CUTOFF_MJD,
            "n_train": int(len(train_X)),
            "n_test": int(len(test_X)),
            "ridge_alpha": float(model.named_steps["ridge"].alpha_),
            "n_bootstrap": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
        },
        "global": {
            "mae_model": round(mae_m, 4),
            "mae_persist": round(mae_p, 4),
            "ratio_mae": round(ratio, 4),
            "ratio_mae_ci95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
            "ratio_mae_boot_mean": round(float(boot_ratios.mean()), 4),
            "frac_boot_ratios_lt_1": round(frac_lt_1, 4),
            "r2_model": round(r2_m, 4),
            "r2_persist": round(r2_p, 4),
            "n_test_stars": len(star_metrics),
            "stars_beating_persist": n_beat,
            "pct_beating_persist": round(100 * n_beat / max(len(star_metrics), 1), 1),
            "median_per_star_ratio": round(float(np.median(per_star_ratios)), 4),
        },
        "by_horizon": by_h,
    }


def bootstrap_ci(values, n_boot=10000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(np.mean(arr)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=5)
    ap.add_argument("--embeddings", type=str, default=str(EMBEDDINGS_PATH),
                    help="Path to star_data.pt (default: reference_results/embeddings)")
    ap.add_argument("--features", type=str, default=str(FEATURES_PATH),
                    help="Path to spectral_features.pt (default: reference_results/features)")
    ap.add_argument("--output_dir", type=str, default=str(RESULTS_DIR),
                    help="Directory for output JSON (default: reference_results/downstream)")
    args = ap.parse_args()

    emb_path = Path(args.embeddings)
    feat_path = Path(args.features)
    out_dir = Path(args.output_dir)

    out = {"context_len": args.context, "seeds": SEEDS, "per_seed": {}}
    for seed in SEEDS:
        print(f"[seed {seed}] loading {emb_path}")
        aligned = load_aligned(emb_path, feat_path)
        print(f"  stars={len(aligned)}")
        res = evaluate_z_ridge(aligned, context_len=args.context)
        print(f"  ratio_MAE={res['global']['ratio_mae']:.4f}  "
              f"CI95={res['global']['ratio_mae_ci95']}  "
              f"MAE_model={res['global']['mae_model']:.3f}  "
              f"pct_beat={res['global']['pct_beating_persist']}%")
        out["per_seed"][str(seed)] = res

    g = out["per_seed"][str(SEEDS[0])]["global"]
    out["aggregate"] = {
        "note": "Single-seed point estimate; uncertainty via bootstrap CI on test predictions.",
        "ratio_mae": g["ratio_mae"],
        "ratio_mae_ci95": g["ratio_mae_ci95"],
        "frac_boot_ratios_lt_1": g["frac_boot_ratios_lt_1"],
    }
    print(f"\nratio_MAE = {g['ratio_mae']:.4f}  CI95 {g['ratio_mae_ci95']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"t3_z_ridge_temporal_baseline_ctx{args.context}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
