"""Shared style & paths for  paper figures.

Import  ``from _common import ROOT, FIG_DIR, setup_style, save_fig``.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths --
ROOT = Path(__file__).resolve().parents[2]  # repo root (anon_bess_files/)
PAPER_DIR = Path(__file__).resolve().parents[1]  # paper dir
# Output dir for figures: defaults to paper/figures/ (committed canonical assets);
# override via BESS_FIG_DIR (used by scripts/reproduce_paper_seed42.sh to write
# under results_reproduced/figures/ without overwriting committed paper/figures/).
FIG_DIR = Path(os.environ.get("BESS_FIG_DIR", str(PAPER_DIR / "figures")))
FIG_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------- colors --
# Consistent palette across the paper.
COL = {
    "z":       "#1f4e79",  # deep blue for BeMAE / z(128D)
    "pca10":   "#d95f02",  # orange for low-dim baseline
    "pca32":   "#e7b416",  # yellow-orange
    "pca50":   "#6c3483",  # purple for large baseline
    "persist": "#7f7f7f",  # gray for persistence
    "chronos": "#2ca02c",  # green for TS-FM
    "timesfm": "#17becf",  # cyan for TS-FM
    "feat6":   "#8c564b",  # brown for 6D features
    "ok":      "#2ca02c",
    "bad":     "#d62728",
    "held":    "#000000",
}

PROFILE_COL = {
    "absorption_pure": "#4477AA",
    "shell":           "#EE6677",
    "single_peak":     "#228833",
    "double_peak":     "#CCBB44",
    "double_asym":     "#AA3377",
}


# --------------------------------------------------------------- style --
def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid":         False,
        "figure.dpi":        110,
        "savefig.dpi":       300,
        "pdf.fonttype":      42,  
        "ps.fonttype":       42,
    })


def save_fig(fig: plt.Figure, name: str) -> tuple[Path, Path]:
    """Save both PDF (vector) and PNG (preview) and print paths."""
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    print(f"[ok] {pdf}")
    print(f"[ok] {png}")
    return pdf, png
