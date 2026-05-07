"""Aggregate Phase B BeMAE-Hα ablation results → table_ablations.tex.

For each ablation variant (mask_ratio_030/075, n_layers_2/6, d_model_64/256,
patch_size_4/16) and the baseline (default config), compute the headline
**aggregated test** : ratio of mean R²(z) over mean R²(PCA(10)) on
{fwhm, central_depth, delta_v}, with bootstrap CI95 on n=3 seeds
(10 000 resamples, RNG=20260428 for reproducibility).

Inputs :
  - downstream/results/probe_features_results_seed{42,123,456}.json  (baseline)
  - downstream/results/probe_features_results_ablation_<tag>_seed{42,123,456}.json

Outputs :
  - paper/tables/table_ablations.tex
  - stats/results/ablations_summary.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parent
PROBE_DIR = BENCH_ROOT / "reference_results" / "downstream"
TABLES_DIR = BENCH_ROOT / "paper" / "tables"
SUMMARY_OUT = BENCH_ROOT / "reference_results" / "stats" / "ablations_summary.json"

HEAD_FEATURES = ["fwhm", "central_depth", "delta_v"]
SEEDS = [42, 123, 456]
N_BOOT = 10_000
BOOT_SEED = 20260428

VARIANTS = [
    # (tag, axis, value, batch_size_override)
    ("baseline",       "—",          "default",  None),
    ("mask_ratio_030", "mask_ratio", "0.30",     None),
    ("mask_ratio_075", "mask_ratio", "0.75",     None),
    ("n_layers_2",     "n_layers",   "2",        None),
    ("n_layers_6",     "n_layers",   "6",        None),
    ("d_model_64",     "d_model",    "64",       None),
    ("d_model_256",    "d_model",    "256",      None),
    ("patch_size_4",   "patch_size", "4",        128),
    ("patch_size_16",  "patch_size", "16",       None),
]


def probe_path(tag: str, seed: int) -> Path:
    if tag == "baseline":
        return PROBE_DIR / f"probe_features_results_seed{seed}.json"
    return PROBE_DIR / f"probe_features_results_ablation_{tag}_seed{seed}.json"


def collect_ratio_per_seed(tag: str) -> dict | None:
    """Return {seed: {z_mean, pca10_mean, ratio}} for the head features.

    Returns None if any seed file is missing.
    """
    out = {}
    for s in SEEDS:
        p = probe_path(tag, s)
        if not p.exists():
            return None
        d = json.load(open(p))
        results = d["results"]
        # Skip if any head feature is missing (e.g. probe was run with
        # a restricted feature_list).
        if not all(f in results for f in HEAD_FEATURES):
            return None
        z_vals = [results[f]["z (128D)"]["r2_mean"] for f in HEAD_FEATURES]
        pca_vals = [results[f]["PCA(10)"]["r2_mean"] for f in HEAD_FEATURES]
        z_mean = float(np.mean(z_vals))
        pca_mean = float(np.mean(pca_vals))
        if pca_mean <= 0:
            return None
        out[s] = {
            "z_mean": z_mean,
            "pca10_mean": pca_mean,
            "ratio": z_mean / pca_mean,
            "z_per_feature": dict(zip(HEAD_FEATURES, z_vals)),
        }
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = values[idx].mean()
    return float(np.mean(boots)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fmt_pm(mean, std):
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def main():
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "head_features": HEAD_FEATURES,
        "seeds": SEEDS,
        "n_bootstrap": N_BOOT,
        "boot_seed": BOOT_SEED,
        "variants": {},
    }

    rows = []
    for tag, axis, value, _bs in VARIANTS:
        per_seed = collect_ratio_per_seed(tag)
        if per_seed is None:
            print(f"  [skip] {tag} : missing probe results")
            summary["variants"][tag] = {"status": "missing"}
            rows.append({"tag": tag, "axis": axis, "value": value, "status": "missing"})
            continue

        ratios = np.array([per_seed[s]["ratio"] for s in SEEDS], dtype=np.float64)
        z_means = np.array([per_seed[s]["z_mean"] for s in SEEDS], dtype=np.float64)
        ratio_mean, ratio_lo, ratio_hi = bootstrap_ci(ratios)
        z_mean_mean, z_lo, z_hi = bootstrap_ci(z_means)

        per_feat_z = {}
        for f in HEAD_FEATURES:
            vals = np.array([per_seed[s]["z_per_feature"][f] for s in SEEDS])
            per_feat_z[f] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
            }

        entry = {
            "axis": axis,
            "value": value,
            "ratio_per_seed": ratios.tolist(),
            "ratio_mean": ratio_mean,
            "ratio_ci95": [ratio_lo, ratio_hi],
            "ratio_significantly_above_1": bool(ratio_lo > 1.0),
            "z_mean_per_seed": z_means.tolist(),
            "z_mean": z_mean_mean,
            "z_mean_ci95": [z_lo, z_hi],
            "per_feature_z": per_feat_z,
        }
        summary["variants"][tag] = entry
        rows.append({
            "tag": tag,
            "axis": axis,
            "value": value,
            "z_mean": z_mean_mean,
            "z_std": float(z_means.std(ddof=1)),
            "ratio_mean": ratio_mean,
            "ratio_lo": ratio_lo,
            "ratio_hi": ratio_hi,
            "per_feat": per_feat_z,
            "status": "ok",
        })
        print(f"  {tag:20s} | {axis:11s}={value:8s} | "
              f"R²(z)={z_mean_mean:.3f} | "
              f"ratio={ratio_mean:.2f} [{ratio_lo:.2f}, {ratio_hi:.2f}]")

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved : {SUMMARY_OUT}")

    # ── LaTeX table ────────────────────────────────────────────────────────
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = TABLES_DIR / "table_ablations.tex"
    lines = []
    lines.append("% Auto-generated by stats/aggregate_ablations.py")
    lines.append("% DO NOT EDIT BY HAND — re-run the script.")
    lines.append("\\begin{tabular}{llrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Axis & Value & "
        "$R^2$(\\texttt{fwhm}) & $R^2$(\\texttt{cd}) & $R^2$(\\texttt{dv}) & "
        "Mean $R^2$ & Ratio $z/\\mathrm{PCA}(10)$ \\\\"
    )
    lines.append("\\midrule")
    for r in rows:
        # Escape underscores for LaTeX text mode (\texttt would also work).
        axis_disp = r["axis"].replace("_", "\\_")
        value_disp = r["value"].replace("_", "\\_")
        if r["tag"] == "baseline":
            axis_disp = "\\textbf{baseline}"
            value_disp = "\\textbf{(8/4/128/0.60)}"
        if r["status"] != "ok":
            lines.append(
                f"{axis_disp} & {value_disp} & "
                f"\\multicolumn{{5}}{{c}}{{\\textit{{(missing --- re-run probe)}}}} \\\\"
            )
            continue
        pf = r["per_feat"]
        ratio_str = f"{r['ratio_mean']:.2f} [{r['ratio_lo']:.2f}, {r['ratio_hi']:.2f}]"
        lines.append(
            f"{axis_disp} & {value_disp} & "
            f"{pf['fwhm']['mean']:.3f} & "
            f"{pf['central_depth']['mean']:.3f} & "
            f"{pf['delta_v']['mean']:.3f} & "
            f"{r['z_mean']:.3f} $\\pm$ {r['z_std']:.3f} & "
            f"{ratio_str} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX table saved : {tex_path}")


if __name__ == "__main__":
    main()
