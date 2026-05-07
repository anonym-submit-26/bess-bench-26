"""Figure 2 — Dataset quality profile: four histograms in one panel.

Panels
------
(a) SNR distribution (log x, truncated at 99th pct)
(b) Spectral resolution (log x)
(c) Year of observation (1990-2025)
(d) Observer type (amateur / professional / unknown) stacked by decade

Sources
-------
- paper/tables/metadata_slim.parquet

Outputs
-------
- figures/fig2_quality_profile.pdf
- figures/fig2_quality_profile.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "paper" / "tables" / "metadata_slim.parquet"
OUT_PDF = ROOT / "figures" / "fig2_quality_profile.pdf"
OUT_PNG = ROOT / "figures" / "fig2_quality_profile.png"

COL_AMATEUR = "#d95f02"
COL_PRO = "#1b9e77"
COL_OTHER = "#7570b3"


def main() -> None:
    df = pd.read_parquet(META)
    n_total = len(df)
    year = Time(df["mjd"].to_numpy(), format="mjd").decimalyear

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.28})

    # ------------------------------------------------------------------ (a) SNR
    ax = axes[0, 0]
    snr = df["snr"].dropna().to_numpy()
    q99 = np.quantile(snr, 0.99)
    snr_c = snr[(snr > 0) & (snr <= q99)]
    bins = np.logspace(np.log10(max(snr_c.min(), 1.0)), np.log10(q99), 40)
    ax.hist(snr_c, bins=bins, color="0.3", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("SNR (header-reported)", fontsize=9)
    ax.set_ylabel("Spectra", fontsize=9)
    ax.set_title("(a) Signal-to-noise ratio", fontsize=10, loc="left")
    ax.text(0.97, 0.92,
            f"median = {np.median(snr_c):.0f}\n99th pct = {q99:.0f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(fc="white", ec="0.85", pad=2))

    # --------------------------------------------------- (b) Spectral resolution
    ax = axes[0, 1]
    res = df["spectral_resolution"].dropna().to_numpy()
    res = res[(res > 0) & (res < 2e5)]
    bins = np.logspace(np.log10(res.min()), np.log10(res.max()), 40)
    ax.hist(res, bins=bins, color="0.3", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel(r"Spectral resolution $R = \lambda/\Delta\lambda$", fontsize=9)
    ax.set_ylabel("Spectra", fontsize=9)
    ax.set_title("(b) Spectral resolution", fontsize=10, loc="left")
    # annotate amateur / pro typical bands
    ax.axvspan(500, 3000, color=COL_AMATEUR, alpha=0.10)
    ax.axvspan(10000, 80000, color=COL_PRO, alpha=0.10)
    ax.text(1400, ax.get_ylim()[1] * 0.88, "amateur\nrange",
            fontsize=7.5, ha="center", color=COL_AMATEUR)
    ax.text(30000, ax.get_ylim()[1] * 0.88, "professional\nrange",
            fontsize=7.5, ha="center", color=COL_PRO)

    # ------------------------------------------------------------------ (c) Year
    ax = axes[1, 0]
    ax.hist(year, bins=np.arange(1990, 2027, 1),
            color="0.3", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Year of observation", fontsize=9)
    ax.set_ylabel("Spectra / yr", fontsize=9)
    ax.set_title("(c) Temporal coverage (1990\u20132025)", fontsize=10, loc="left")
    ax.axvline(2005, color="0.5", ls=":", lw=0.7)
    ax.text(2005.5, ax.get_ylim()[1] * 0.88, "BeSS launch\n2005",
            fontsize=7.5, color="0.35")

    # ------------------------------------------------------------- (d) Observer
    ax = axes[1, 1]
    obs_type = df["observer_type"].fillna("unknown")
    year_rounded = np.floor(year).astype(int)
    years_unique = np.arange(1990, 2026)
    amateur = np.zeros_like(years_unique, dtype=int)
    pro = np.zeros_like(years_unique, dtype=int)
    other = np.zeros_like(years_unique, dtype=int)
    for i, y in enumerate(years_unique):
        mask = year_rounded == y
        sub = obs_type[mask]
        amateur[i] = (sub == "amateur").sum()
        pro[i] = (sub == "professional").sum()
        other[i] = len(sub) - amateur[i] - pro[i]

    width = 0.9
    ax.bar(years_unique, amateur, width=width, color=COL_AMATEUR,
           label=f"amateur  ({amateur.sum()/n_total:.1%})", edgecolor="none")
    ax.bar(years_unique, pro, width=width, bottom=amateur, color=COL_PRO,
           label=f"professional  ({pro.sum()/n_total:.1%})", edgecolor="none")
    ax.bar(years_unique, other, width=width, bottom=amateur + pro,
           color=COL_OTHER, alpha=0.8,
           label=f"unknown  ({other.sum()/n_total:.1%})", edgecolor="none")
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Spectra / yr", fontsize=9)
    ax.set_title("(d) Observer type over time", fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False,
              handlelength=1.2, handletextpad=0.4)

    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, axis="y", ls="-", lw=0.3, color="0.9")
        ax.set_axisbelow(True)

    fig.suptitle(
        f"BESS-Bench dataset composition · {n_total:,} spectra · 1,468 stars · 1990\u20132025",
        fontsize=10.5, y=0.995,
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=180)
    print(f"[+] {OUT_PDF}")
    print(f"[+] {OUT_PNG}")


if __name__ == "__main__":
    main()
