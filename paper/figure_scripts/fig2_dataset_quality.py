"""Figure 2 — Dataset quality profile (4 panels).

Input
-----
``paper/tables/metadata_slim.parquet`` (one row per observation).

Output
------
- ``figures/fig2_quality_v2.pdf``
- ``figures/fig2_quality_v2.png``

"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, save_fig, setup_style  # noqa: E402

META = ROOT / "paper" / "tables" / "metadata_slim.parquet"

COL_PRO     = "#1b9e77"
COL_AMATEUR = "#d95f02"
COL_OTHER   = "#7570b3"


def main() -> None:
    setup_style()
    df = pd.read_parquet(META)
    n_total = len(df)
    year = Time(df["mjd"].to_numpy(), format="mjd").decimalyear

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.32})

    # ---------------------------------------------------------- (a) SNR
    ax = axes[0, 0]
    snr = df["snr"].dropna().to_numpy()
    q99 = np.quantile(snr, 0.99)
    snr_c = snr[(snr > 0) & (snr <= q99)]
    bins = np.logspace(np.log10(max(snr_c.min(), 1.0)), np.log10(q99), 40)
    ax.hist(snr_c, bins=bins, color="0.3", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.axvline(10, color="#d62728", lw=0.8, ls="--")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Spectra")
    ax.set_title("(a) Signal-to-noise", loc="left")
    ax.text(0.97, 0.92, f"median = {np.median(snr_c):.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.text(0.03, 0.92, "SNR cutoff = 10",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="#d62728")

    # ------------------------------------------------------- (b) Year
    ax = axes[0, 1]
    ax.hist(year, bins=np.arange(int(year.min()), int(year.max()) + 2),
            color="0.3", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Year")
    ax.set_ylabel("Spectra")
    ax.set_title(f"(b) Temporal coverage ({int(year.min())}\u2013{int(year.max())})",
                 loc="left")

    # ------------------------------------------------- (c) Observer type
    ax = axes[1, 0]
    if "observer_type" in df.columns:
        counts = df["observer_type"].fillna("unknown").value_counts()
    else:
        counts = pd.Series({"unknown": n_total})
    colors = [COL_PRO if k == "professional" else
              COL_AMATEUR if k == "amateur" else COL_OTHER
              for k in counts.index]
    ax.barh(counts.index.astype(str), counts.values, color=colors)
    for y, v in enumerate(counts.values):
        ax.text(v, y, f" {v:,}  ({100*v/n_total:.1f}%)",
                va="center", fontsize=7.5)
    ax.set_xlabel("Spectra")
    ax.set_title("(c) Observer community", loc="left")
    # Give room for the right-hand text labels so they aren't clipped.
    ax.set_xlim(0, counts.values.max() * 1.85)

    # ------------------------------------------- (d) Spectra per star
    ax = axes[1, 1]
    per_star = df.groupby("star_name").size().values
    bins = np.logspace(0, np.log10(per_star.max() + 1), 40)
    ax.hist(per_star, bins=bins, color="0.3", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.axvline(30, color="#d62728", lw=0.8, ls="--")
    ax.set_xlabel("Spectra per star")
    ax.set_ylabel("Stars")
    ax.set_title("(d) Sampling richness", loc="left")
    n_30 = int((per_star >= 30).sum())
    ax.text(0.97, 0.92,
            f"{n_30} stars $\\geq$30 spectra\n(benchmark threshold)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#d62728")

    fig.suptitle(f"BESS-Bench v1.0 \u2014 {n_total:,} spectra, "
                 f"{df['star_name'].nunique():,} stars",
                 fontsize=10, y=0.995)
    save_fig(fig, "fig2_quality_v2")


if __name__ == "__main__":
    main()
