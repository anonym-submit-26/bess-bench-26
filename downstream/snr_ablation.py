"""
snr_ablation.py — P0-2: SpecProbe per-SNR-bin diagnostic.

For each spectral feature and each frozen representation
(z=BeMAE-128, PCA(10), raw flux 128), runs the standard SpecProbe
ridge probe with out-of-fold predictions on the full corpus,
then reports R² per SNR bin (low/medium/good/excellent).

Output: results/snr_ablation_seed{ENC}.json + a console summary.

Usage:
    python snr_ablation.py --embeddings ../data/embeddings_halpha/star_data.pt
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ["fwhm", "central_depth", "delta_v", "vr_ratio"]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
SNR_BINS = [(0, 60, "low"), (60, 120, "medium"),
            (120, 300, "good"), (300, np.inf, "excellent")]


def align(emb_data, feat_data, mjd_tol=0.01):
    X, flux, snrs, feats = [], [], [], {n: [] for n in FEATURE_NAMES}
    for star in sorted(set(emb_data) & set(feat_data)):
        ed, fd = emb_data[star], feat_data[star]
        em = ed["mjds"].numpy(); fm = fd["mjds"].numpy()
        ev = ed["embeddings"].numpy(); sn = ed["snrs"].numpy()
        flx = fd["flux_norm"].numpy()
        for i, m in enumerate(em):
            j = int(np.argmin(np.abs(fm - m)))
            if abs(fm[j] - m) < mjd_tol:
                X.append(ev[i]); flux.append(flx[j]); snrs.append(sn[i])
                for n in FEATURE_NAMES:
                    feats[n].append(float(fd[n][j].item()))
    return (np.asarray(X, np.float32),
            np.asarray(flux, np.float32),
            np.asarray(snrs, np.float32),
            {n: np.asarray(v, np.float32) for n, v in feats.items()})


def probe_oof(X, y, seed=42, n_folds=5):
    valid = np.isfinite(y)
    Xv, yv = X[valid], y[valid]
    pipe = Pipeline([("s", StandardScaler()), ("r", RidgeCV(alphas=ALPHAS))])
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    yhat = cross_val_predict(pipe, Xv, yv, cv=cv)
    return valid, yhat, yv


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--features",
                   default="results/spectral_features.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/snr_ablation.json")
    args = p.parse_args()

    print(f"[snr_ablation] seed={args.seed}")
    emb = torch.load(args.embeddings, weights_only=False)
    feat = torch.load(args.features, weights_only=False)
    X, flux, snrs, feats = align(emb, feat)
    print(f"  aligned {len(X):,} spectra")
    print(f"  SNR median={np.median(snrs):.0f} q25={np.quantile(snrs,0.25):.0f} q75={np.quantile(snrs,0.75):.0f}")

    pca10 = PCA(n_components=10, random_state=args.seed).fit_transform(
        StandardScaler().fit_transform(flux))

    representations = {"BeMAE": X, "PCA10": pca10, "Flux128": flux}
    out = {"task": "snr_ablation", "seed": args.seed,
           "n_spectra": int(len(X)), "bins": {}, "global": {}}

    # bin counts
    for lo, hi, name in SNR_BINS:
        mask = (snrs >= lo) & (snrs < hi)
        out["bins"][name] = {"snr_lo": float(lo),
                             "snr_hi": (None if not np.isfinite(hi) else float(hi)),
                             "n": int(mask.sum())}
        print(f"  bin {name:9s} [{lo:>4.0f}, {hi:>5.0f}): n={mask.sum():,}")

    # for each rep × feature: get OOF predictions and stratify R² by bin
    for rep_name, Xrep in representations.items():
        out["bins"][rep_name] = {} if False else None  # placeholder
        for feat_name in FEATURE_NAMES:
            y = feats[feat_name]
            valid, yhat, yv = probe_oof(Xrep, y, seed=args.seed)
            global_r2 = r2(yv, yhat)
            out["global"].setdefault(rep_name, {})[feat_name] = global_r2
            # per-bin R² (use SNR of valid spectra)
            snrs_v = snrs[valid]
            for lo, hi, bname in SNR_BINS:
                bm = (snrs_v >= lo) & (snrs_v < hi)
                if bm.sum() < 20:
                    rr = float("nan")
                else:
                    rr = r2(yv[bm], yhat[bm])
                out["bins"].setdefault(bname, {}) \
                    .setdefault(rep_name, {})[feat_name] = rr

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None
                  if isinstance(x, float) and not np.isfinite(x) else x)

    # console summary table
    print("\n  R² per bin × representation × feature")
    print(f"  {'feat':14s} | {'rep':8s} | " +
          " | ".join(f"{n:>9s}" for _, _, n in SNR_BINS) + " | global")
    for feat_name in FEATURE_NAMES:
        for rep_name in ("BeMAE", "PCA10", "Flux128"):
            row = [out["bins"][bn][rep_name][feat_name]
                   for _, _, bn in SNR_BINS]
            row_s = " | ".join(f"{v:>9.3f}" if v == v else f"{'    n/a':>9s}"
                               for v in row)
            print(f"  {feat_name:14s} | {rep_name:8s} | {row_s} | "
                  f"{out['global'][rep_name][feat_name]:.3f}")
    print(f"\n  saved → {args.output}")


if __name__ == "__main__":
    main()
