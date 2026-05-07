#!/usr/bin/env python
"""Generate Fig. T3-horizon: MAE(model) / MAE(persistence) stratified by
forecast-horizon bucket, for the best BeMAE-derived baseline (PCA10 ctx5)
and the two zero-shot time-series foundation-model baselines
(Chronos-Bolt-base, TimesFM-2.0-500M).

Reads (from --input-dir, defaults to reference_results/downstream/):
  - t3_pca_ridge_temporal_baseline.json
  - t3_ts_fm_baselines.json (full schema with `predictions` field)
  - t3_z_ridge_temporal_baseline_ctx5.json
Writes:
  - paper/figures/fig_t3_horizon.pdf (default), overridable via --output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "reference_results" / "downstream"
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "fig_t3_horizon.pdf"

BUCKETS = [
    ("< 7d", 0, 7),
    ("7-30d", 7, 30),
    ("30-90d", 30, 90),
    ("90-365d", 90, 365),
    ("> 365d", 365, float("inf")),
]

DISPLAY_LABELS = {b[0]: b[0] for b in BUCKETS}


def horizon_ratios(preds: dict) -> dict:
    y_true = np.asarray(preds["y_true"], dtype=float)
    y_pred = np.asarray(preds["y_pred_model"], dtype=float)
    y_pers = np.asarray(preds["y_pred_persist"], dtype=float)
    dt = np.asarray(preds["dt_days"], dtype=float)

    out = {}
    for label, lo, hi in BUCKETS:
        mask = (dt >= lo) & (dt < hi)
        n = int(mask.sum())
        if n == 0:
            out[label] = {"n": 0, "ratio": np.nan}
            continue
        mae_m = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
        mae_p = float(np.mean(np.abs(y_true[mask] - y_pers[mask])))
        out[label] = {"n": n, "ratio": mae_m / mae_p if mae_p > 0 else np.nan,
                      "mae_model": mae_m, "mae_persist": mae_p}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory containing t3_*.json files "
                             "(default: reference_results/downstream).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output PDF path (default: paper/figures/fig_t3_horizon.pdf).")
    args = parser.parse_args()

    in_dir = args.input_dir
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pca = json.loads((in_dir / "t3_pca_ridge_temporal_baseline.json").read_text())["pca10_ctx5"]["by_horizon"]
    tsfm = json.loads((in_dir / "t3_ts_fm_baselines.json").read_text())
    zridge = json.loads((in_dir / "t3_z_ridge_temporal_baseline_ctx5.json").read_text())

    chronos = horizon_ratios(tsfm["chronos_bolt_base"]["predictions"])
    timesfm = horizon_ratios(tsfm["timesfm_2_0_500m"]["predictions"])

    labels = [b[0] for b in BUCKETS]
    pca_ratios = [pca[lbl]["ratio"] for lbl in labels]
    chr_ratios = [chronos[lbl]["ratio"] for lbl in labels]
    tfm_ratios = [timesfm[lbl]["ratio"] for lbl in labels]

    # z+Ridge: single deterministic encoder (seed=42); no error bars.
    # Global-ratio uncertainty is reported separately as a 10k bootstrap
    # 95% CI in t3_z_ridge_temporal_baseline_ctx5.json -> per_seed/42/global.
    z_seed = zridge["per_seed"]["42"]
    z_ratios = np.array([z_seed["by_horizon"][lbl]["ratio"] for lbl in labels])

    pca_n = [pca[lbl]["n"] for lbl in labels]

    x = np.arange(len(labels))
    width = 0.21

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.bar(x - 1.5 * width, pca_ratios, width,
           label="BeMAE+PCA(10), ctx=5", color="#1f77b4",
           edgecolor="black", linewidth=0.4)
    ax.bar(x - 0.5 * width, z_ratios, width,
           label=r"BeMAE $z_{128}$+Ridge, ctx=5",
           color="#9467bd", edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.5 * width, chr_ratios, width,
           label="Chronos-Bolt-base (zero-shot)", color="#ff7f0e",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + 1.5 * width, tfm_ratios, width,
           label="TimesFM-2.0-500M (zero-shot)", color="#2ca02c",
           edgecolor="black", linewidth=0.4)

    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.7)
    ax.text(len(labels) - 0.4, 1.005, "persistence", fontsize=7,
            ha="right", va="bottom", style="italic", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_LABELS[lbl] for lbl in labels], fontsize=8)
    ax.set_xlabel("Forecast horizon (days between consecutive epochs)", fontsize=8,
                  labelpad=18)
    ax.set_ylabel(r"$\mathrm{MAE}_\mathrm{model} / \mathrm{MAE}_\mathrm{persist}$",
                  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0.85, 1.20)
    ax.legend(fontsize=7.5, loc="upper right", frameon=True, framealpha=0.92,
              ncol=1, handlelength=1.6, handletextpad=0.5)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    # n annotations placed between x-tick labels and x-axis title.
    trans = ax.get_xaxis_transform()
    for xi, n in zip(x, pca_n):
        ax.text(xi, -0.16, f"n={n}", fontsize=6, ha="center",
                color="dimgray", transform=trans)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")
    print("\nratios per bucket:")
    print(f"  {'bucket':<10} {'PCA10':>7} {'z+Ridge':>8} {'Chronos':>9} {'TimesFM':>9} {'n_pca':>6}")
    for lbl, p, z, c, t, n in zip(labels, pca_ratios, z_ratios,
                                   chr_ratios, tfm_ratios, pca_n):
        print(f"  {lbl:<10} {p:>7.3f} {z:>8.3f} {c:>9.3f} {t:>9.3f} {n:>6d}")


if __name__ == "__main__":
    main()
