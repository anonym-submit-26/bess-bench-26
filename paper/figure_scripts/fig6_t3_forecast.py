"""Figure 6 — T3 forecasting: ratio MAE / persistence with bootstrap CI95.

T3 is our temporal-dynamics task: given a short context of EW(Halpha) values,
predict the next one. We compare 4 configurations of PCA(k)+Ridge and show
that all beat persistence (ratio < 1). A dot-and-whisker plot highlights that
the best ratio (0.942) is modest but statistically significant.

Input
-----
``stats/results/t3_bootstrap_cis.json``
Structure:
    pca_ridge_bootstrap.by_config[cfg] has ratio_mae_point, ratio_mae_ci_lo, ratio_mae_ci_hi.

Output
------
- ``figures/fig6_t3_forecast.pdf``
- ``figures/fig6_t3_forecast.png``

"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import COL, ROOT, save_fig, setup_style  # noqa: E402

T3_JSON = ROOT / "reference_results" / "stats" / "t3_bootstrap_cis.json"

# Display order + colors
CONFIGS = [
    ("pca10_ctx5",  "PCA(10) · ctx=5",  COL["pca10"]),
    ("pca10_ctx10", "PCA(10) · ctx=10", COL["pca10"]),
    ("pca50_ctx5",  "PCA(50) · ctx=5",  COL["pca50"]),
    ("pca50_ctx10", "PCA(50) · ctx=10", COL["pca50"]),
]


def main() -> None:
    setup_style()
    data = json.loads(T3_JSON.read_text())["pca_ridge_bootstrap"]["by_config"]

    n = len(CONFIGS)
    y = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 2.8))

    for yi, (key, label, color) in zip(y, CONFIGS):
        d = data[key]
        m  = d["ratio_mae_point"]
        lo = d["ratio_mae_ci_lo"]
        hi = d["ratio_mae_ci_hi"]
        ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]],
                    fmt="o", color=color, markersize=8,
                    capsize=3, elinewidth=1.2, markeredgecolor="white")
        ax.text(hi + 0.003, yi,
                f"{m:.3f} [{lo:.3f}, {hi:.3f}]",
                va="center", fontsize=8)

    ax.axvline(1.0, color=COL["bad"], lw=0.8, ls="--",
               label="Persistence (= 1.0, baseline naïve)")
    ax.set_yticks(y)
    ax.set_yticklabels([lbl for _, lbl, _ in CONFIGS])
    ax.set_xlabel("MAE(model) / MAE(persistence)  —  lower = better")
    ax.set_xlim(0.92, 1.01)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax.set_title("T3 — forecasting EW(H$\\alpha$) : all configs beat "
                 "persistence (CI95 $<$ 1)")
    save_fig(fig, "fig6_t3_forecast")


if __name__ == "__main__":
    main()
