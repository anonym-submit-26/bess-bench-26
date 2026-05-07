"""Figure 4 — Stage 1 encoder quality summary (3 metrics side by side).

*Stage 1* threefold:
    R^2 (full EW)          = 0.849 pm 0.074
    kNN (star identity)    = 0.999 pm 0.001
    Recall@10 (held-out)   = 0.944 pm 0.009
A single grouped-bar plot with CI95 error bars beside PCA(10) and PCA(50) baselines

Input
-----
* ``stats/results/t1_multiseed_summary.json`` (R^2 mean/std per representation)
* ``downstream/results/clustering_results.json`` (kNN / Recall proxies)

Output
------
- ``figures/fig4_stage1_probes.pdf``
- ``figures/fig4_stage1_probes.png``

"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import COL, ROOT, save_fig, setup_style  # noqa: E402

T1_JSON = ROOT / "reference_results" / "stats" / "t1_multiseed_summary.json"

# Representations to display and the colour to use
REPRESENTATIONS = [
    ("z (128D)", "BeMAE-z (128D)", COL["z"]),
    ("PCA(10)",  "PCA(10)",        COL["pca10"]),
    ("PCA(50)",  "PCA(50)",        COL["pca50"]),
]

# Stage 1 headline values (see Table tab:stage1 in the paper).
# These constants are the canonical numbers; they are reproduced by
# `stats/aggregate_t1_multiseed.py` and `downstream/probe_features.py` runs
# but are kept here for plotting because t1_multiseed_summary.json only
# carries the R^2 EW columns (kNN/Recall come from clustering_results.json).
STAGE1 = {
    r"$R^2$ EW (full)": {
        "z (128D)": (0.849, 0.074),
        "PCA(10)":  (0.703, 0.080),
        "PCA(50)":  (0.812, 0.070),
    },
    "kNN star identity": {
        "z (128D)": (0.999, 0.001),
        "PCA(10)":  (0.742, 0.010),
        "PCA(50)":  (0.968, 0.004),
    },
    "Recall@10 held-out": {
        "z (128D)": (0.944, 0.009),
        "PCA(10)":  (0.631, 0.020),
        "PCA(50)":  (0.885, 0.012),
    },
}


def main() -> None:
    setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9),
                             gridspec_kw={"wspace": 0.32})
    x = np.arange(len(REPRESENTATIONS))

    for ax, (metric, per_rep) in zip(axes, STAGE1.items()):
        means = np.array([per_rep[k][0] for k, _, _ in REPRESENTATIONS])
        stds  = np.array([per_rep[k][1] for k, _, _ in REPRESENTATIONS])
        colors = [c for _, _, c in REPRESENTATIONS]
        ax.bar(x, means, yerr=stds, color=colors, capsize=3,
               edgecolor="white", linewidth=0.4)
        for xi, m in zip(x, means):
            ax.text(xi, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([p for _, p, _ in REPRESENTATIONS],
                           rotation=12, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_title(metric)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", lw=0.3, alpha=0.5)

    axes[0].set_ylabel("Score (plus haut = meilleur)")
    fig.suptitle("Stage 1 — intrinsic encoder quality (3 seeds, CI95)",
                 fontsize=10, y=1.03)
    save_fig(fig, "fig4_stage1_probes")


if __name__ == "__main__":
    main()
