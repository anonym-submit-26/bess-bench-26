"""Figure 5 — T1 frozen-embedding probes, R^2 per feature with CI95.

T1 is the headline downstream task. For each of 4 physically meaningful
features (fwhm, central_depth, delta_v, vr_ratio) we compare BeMAE-z(128D)
to PCA(10), PCA(32), PCA(50). A horizontal grouped-bar plot with CI95
whiskers makes the 3×–10× gains *and* the vr_ratio negative result visible.

Input
-----
``stats/results/t1_multiseed_summary.json``
key is ``{feature}__{representation}``, e.g. ``fwhm__PCA(10)``.

Output
------
- ``figures/fig5_t1_per_feature.pdf``
- ``figures/fig5_t1_per_feature.png``


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

FEATURES = ["fwhm", "central_depth", "delta_v", "vr_ratio"]
FEATURE_LABELS = {
    "fwhm":          r"FWHM",
    "central_depth": "Central depth",
    "delta_v":       r"$\Delta v$",
    "vr_ratio":      r"V/R ratio",
}
REPRS = [
    ("z (128D)",  "BeMAE-z (128D)", COL["z"]),
    ("PCA(10)",   "PCA(10)",        COL["pca10"]),
    ("PCA(32)",   "PCA(32)",        COL["pca32"]),
    ("PCA(50)",   "PCA(50)",        COL["pca50"]),
]


def _get(summary: dict, feat: str, rep: str) -> tuple[float, float, float]:
    key = f"{feat}__{rep}"
    if key not in summary:
        return (float("nan"),) * 3
    s = summary[key]
    lo, hi = s.get("ci95_mean_bootstrap", [s["mean"] - s["std"], s["mean"] + s["std"]])
    return s["mean"], lo, hi


def main() -> None:
    setup_style()
    raw = json.loads(T1_JSON.read_text())
    summary = raw["summary"]

    n_feat = len(FEATURES)
    n_rep = len(REPRS)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    bar_w = 0.8 / n_rep
    x_pos = np.arange(n_feat)

    for i, (rep_key, rep_label, color) in enumerate(REPRS):
        means, err_lo, err_hi = [], [], []
        for feat in FEATURES:
            m, lo, hi = _get(summary, feat, rep_key)
            means.append(m)
            err_lo.append(max(0.0, m - lo))
            err_hi.append(max(0.0, hi - m))
        x = x_pos + (i - (n_rep - 1) / 2) * bar_w
        ax.bar(x, means, width=bar_w, color=color,
               yerr=[err_lo, err_hi], capsize=2,
               edgecolor="white", linewidth=0.3, label=rep_label)
        for xi, m in zip(x, means):
            if np.isfinite(m):
                ax.text(xi, max(m, 0) + 0.025, f"{m:.2f}",
                        ha="center", va="bottom", fontsize=6.5)

    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES])
    ax.set_ylabel(r"Cross-validated $R^2$  (5 folds, 3 seeds)")
    ax.set_ylim(0, 1.10)
    ax.legend(loc="upper right", frameon=True, fontsize=7.5,
              ncol=2, handlelength=1.4, handletextpad=0.4,
              columnspacing=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.set_title("SpecProbe: BeMAE-z vs PCA on four spectral features",
                 fontsize=9)
    save_fig(fig, "fig5_t1_per_feature")


if __name__ == "__main__":
    main()
