#!/usr/bin/env python3
"""
figure_multiseed.py — Two key figures for the paper.

fig1 : T3 multi-seed negative result
  - horizontal bars ratio_mae per seed
  - vertical line at ratio = 1 (persistence)
  - annotation mean ± sd

fig2 : T1 embedding vs PCA probe R^2 (bar grouped)
  - 4 features × 3 methods (z, PCA(10), PCA(50))
  - 95% Gaussian CIs (from stats/bootstrap_ci.py)

Input: paper/tables/summary.json, paper/tables/ci_results.json
Outputs:
  figures/fig_t3_multiseed.pdf, .png
  figures/fig_t1_probes.pdf, .png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig_t3_multiseed():
    summary = json.load(open(ROOT / "paper" / "tables" / "summary.json"))
    if "t3_multiseed" not in summary:
        print("[.] t3_multiseed not in summary.json — T3 is single-seed in v1.0; skipping fig_t3_multiseed")
        return
    s = summary["t3_multiseed"]
    ci = json.load(open(ROOT / "paper" / "tables" / "ci_results.json"))["T3"]

    seeds = list(s["per_seed"].keys())
    ratios = [s["per_seed"][k]["ratio_mae"] for k in seeds]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    colors = ["#c0392b" if r > 1 else "#27ae60" for r in ratios]
    y = np.arange(len(seeds))
    ax.barh(y, ratios, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, label="persistence")
    ax.set_yticks(y)
    ax.set_yticklabels([f"seed {s}" for s in seeds])
    ax.set_xlabel(r"MAE ratio: $\mathrm{MAE}_{\mathrm{model}} / \mathrm{MAE}_{\mathrm{persist}}$")
    ax.set_xlim(0, max(ratios) * 1.12)

    mean_r = s["ratio_mean"]
    std_r  = s["ratio_std"]
    ax.annotate(
        f"mean $\\pm$ sd : {mean_r:.3f} $\\pm$ {std_r:.3f}\n"
        f"bootstrap CI95: [{ci['ratio_ci95'][0]:.3f}, {ci['ratio_ci95'][1]:.3f}]\n"
        f"p(ratio > 1) = {ci['p_ratio_gt_1']:.3f}",
        xy=(0.98, 0.05), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.3"),
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.95), fontsize=8)
    ax.set_title("T3 — Causal transformer vs persistence (EW(H$\\alpha$) tracking)",
                 fontsize=10)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"fig_t3_multiseed.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[+] {path}")
    plt.close(fig)


def fig_t1_probes():
    t1 = json.load(open(ROOT / "paper" / "tables" / "ci_results.json"))["T1"]
    features = ["fwhm", "central_depth", "delta_v", "vr_ratio"]
    methods = [("z", "z (128D)"), ("pca10", "PCA(10)"), ("pca50", "PCA(50)")]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(len(features))
    width = 0.26
    colors = {"z": "#2c3e50", "pca10": "#7f8c8d", "pca50": "#bdc3c7"}

    for i, (key, label) in enumerate(methods):
        means = [t1[f][key]["r2_mean"] for f in features]
        lo    = [t1[f][key]["ci95"][0] for f in features]
        hi    = [t1[f][key]["ci95"][1] for f in features]
        err = np.array([[m - l for m, l in zip(means, lo)],
                        [h - m for m, h in zip(means, hi)]])
        ax.bar(x + (i - 1) * width, means, width, yerr=err,
               label=label, color=colors[key],
               edgecolor="black", linewidth=0.5, capsize=3)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.set_ylabel(r"$R^2$ (held-out, 5-fold CV)")
    ax.set_ylim(-0.15, 1.0)
    ax.set_title("T1 — Ridge probes of spectral features from frozen representations",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"fig_t1_probes.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[+] {path}")
    plt.close(fig)


def main():
    fig_t3_multiseed()
    fig_t1_probes()


if __name__ == "__main__":
    main()
