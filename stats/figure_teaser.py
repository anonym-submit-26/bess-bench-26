"""Figure 1 — Teaser: EW(Halpha) vs time for three emblematic Be stars.

Shows 35 years of EW(Halpha) variability across a quiescent cyclical
star (28 Cyg), a star with a strong 2020 outburst (Pleione) and the
archetypal active Be star (gamma Cas). Motivates the benchmark in one
figure.

Sources
-------
- downstream/results/spectral_features.pt   per-star EW and MJD tensors
- paper/tables/metadata_slim.parquet        observer_type flag for marker colour

Outputs
-------
- figures/fig1_teaser.pdf
- figures/fig1_teaser.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.time import Time
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "reference_results" / "features" / "spectral_features.pt"
OUT_PDF = ROOT / "figures" / "fig1_teaser.pdf"
OUT_PNG = ROOT / "figures" / "fig1_teaser.png"

STARS = [
    ("28 CYG",   "28 Cyg",     "quiescent cyclical Be star"),
    ("PLEIONE",  "Pleione",    "2020 disk-loss event"),
    ("GAM CAS",  r"$\gamma$ Cas", "archetypal active Be star, X-ray binary"),
]

COL_POINT = "#1f4e79"


def mjd_to_year(mjd: np.ndarray) -> np.ndarray:
    return Time(mjd, format="mjd").decimalyear


def main() -> None:
    feats = torch.load(FEATURES, map_location="cpu", weights_only=False)

    fig, axes = plt.subplots(len(STARS), 1, figsize=(7.0, 5.4), sharex=True,
                             gridspec_kw={"hspace": 0.12})

    mjd_all_min, mjd_all_max = np.inf, -np.inf

    for ax, (key, label, subtitle) in zip(axes, STARS):
        d = feats[key]
        mjd = d["mjds"].numpy()
        ew = d["ew"].numpy()
        mjd_all_min = min(mjd_all_min, float(mjd.min()))
        mjd_all_max = max(mjd_all_max, float(mjd.max()))

        years = mjd_to_year(mjd)
        order = np.argsort(years)

        ax.plot(years[order], ew[order], color=COL_POINT, lw=0.4, alpha=0.35)
        ax.scatter(years, ew, s=5, c=COL_POINT, alpha=0.55, rasterized=True,
                   edgecolors="none")

        ax.axhline(0.0, color="0.5", lw=0.6, ls=":")
        ax.set_ylabel(r"EW(H$\alpha$)  [$\mathrm{\AA}$]", fontsize=10)

        ax.text(0.01, 0.96, label,
                transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="top", ha="left")
        ax.text(0.01, 0.82, subtitle,
                transform=ax.transAxes, fontsize=8.5, color="0.35",
                va="top", ha="left", style="italic")
        ax.text(0.99, 0.96, f"N = {len(ew)} spectra",
                transform=ax.transAxes, fontsize=8.5, color="0.35",
                va="top", ha="right")

        ax.grid(True, which="major", ls="-", lw=0.3, color="0.9")
        ax.yaxis.set_major_locator(MaxNLocator(4))

    axes[-1].set_xlabel("Year", fontsize=10)
    y0, y1 = mjd_to_year(np.array([mjd_all_min, mjd_all_max]))
    axes[-1].set_xlim(np.floor(y0 - 0.5), np.ceil(y1 + 0.5))

    fig.suptitle(
        r"Three decades of Be-star H$\alpha$ variability captured by BESS-Bench",
        fontsize=11, y=0.995,
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=180)
    print(f"[+] {OUT_PDF}")
    print(f"[+] {OUT_PNG}")


if __name__ == "__main__":
    main()
