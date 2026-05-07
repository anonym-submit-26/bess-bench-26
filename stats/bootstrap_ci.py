#!/usr/bin/env python3
"""
bootstrap_ci.py — 95% confidence intervals for T1 probes.

**Probe CI (T1)** — Gaussian CI via
    CI = mean ± 1.96 * std / sqrt(k)
k=5 (folds). For each feature × method (z / PCA(10) / PCA(50)), we
report the interval and test non-overlap with PCA(10).

T3 CIs are computed separately by ``stats/bootstrap_ci_t3.py``
(non-parametric bootstrap on test residuals; PCA+Ridge +
time-series foundation models zero-shot).

Outputs:
  paper/tables/ci_results.json
  paper/tables/table_ci_t1.tex
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Output dir: defaults to paper/tables/ (committed); override via env var.
OUT = Path(os.environ.get("BESS_TABLES_DIR", str(ROOT / "paper" / "tables")))
OUT.mkdir(parents=True, exist_ok=True)
DOWNSTREAM_DIR = Path(os.environ.get(
    "BESS_DOWNSTREAM_DIR", str(ROOT / "reference_results" / "downstream")))

CV_FOLDS = 5
Z95 = 1.959963984540054


def t1_gaussian_ci(mean: float, std: float, k: int = CV_FOLDS) -> tuple[float, float]:
    half = Z95 * std / math.sqrt(k)
    return mean - half, mean + half


def t1_probe_ci() -> dict:
    # Prefer the seed-42 file (path A repro produces this) over the legacy unseeded one.
    p_seed = DOWNSTREAM_DIR / "probe_features_results_seed42.json"
    p_legacy = DOWNSTREAM_DIR / "probe_features_results.json"
    src = p_seed if p_seed.exists() else p_legacy
    d = json.load(open(src))
    # Schema: { seed, n_folds, features, representations, results: { feat: { rep: {r2_mean, r2_std, ...} } } }
    results = d["results"] if "results" in d else d
    out = {}
    for feat in ["fwhm", "central_depth", "delta_v", "vr_ratio"]:
        out[feat] = {}
        for method, key_short in [
            ("z (128D)", "z"), ("PCA(10)", "pca10"), ("PCA(50)", "pca50"),
        ]:
            m = results[feat][method]
            lo, hi = t1_gaussian_ci(m["r2_mean"], m["r2_std"])
            out[feat][key_short] = {
                "r2_mean": m["r2_mean"],
                "r2_std":  m["r2_std"],
                "ci95":    [lo, hi],
            }
        # z-vs-PCA comparison: are CIs disjoint?
        z_lo = out[feat]["z"]["ci95"][0]
        p10_hi = out[feat]["pca10"]["ci95"][1]
        out[feat]["z_beats_pca10_nonoverlap"] = (z_lo > p10_hi)
    return out


def render_tex_t1(t1: dict) -> str:
    lines = []
    for feat, m in t1.items():
        # Best R^2 across the three representations gets bold.
        best_method = max(("z", "pca10", "pca50"), key=lambda k: m[k]["r2_mean"])
        for method in ["z", "pca10", "pca50"]:
            mm = m[method]
            r2_str = f"{mm['r2_mean']:.3f}"
            ci_str = f"[{mm['ci95'][0]:.3f}, {mm['ci95'][1]:.3f}]"
            if method == best_method:
                r2_str = f"\\textbf{{{r2_str}}}"
                ci_str = f"\\textbf{{{ci_str}}}"
            lines.append(
                f"  {feat.replace('_', chr(92)+'_'):<20} & {method:<5} & "
                f"{r2_str} & {ci_str} \\\\"
            )
        lines.append("  \\midrule")
    return (
        "\\begin{tabular}{llrr}\n"
        "  \\toprule\n"
        "  Feature & Method & $R^2$ & 95\\% CI \\\\\n"
        "  \\midrule\n"
        + "\n".join(lines[:-1]) + "\n"  # drop trailing midrule
        + "\n  \\bottomrule\n"
        "\\end{tabular}\n"
        "% Gaussian 95\\% CI from 5-fold CV: $\\bar{R}^2 \\pm 1.96 s / \\sqrt{5}$. "
        "Bold = highest $R^2$ for that feature.\n"
    )


def main():
    t1 = t1_probe_ci()

    (OUT / "ci_results.json").write_text(
        json.dumps({"T1": t1}, indent=2)
    )
    (OUT / "table_ci_t1.tex").write_text(render_tex_t1(t1))

    print(f"[+] {OUT / 'ci_results.json'}")
    print(f"[+] {OUT / 'table_ci_t1.tex'}")

    print("\n[=] T1 embedding vs PCA(10) disjoint-CI ('strictly better')")
    for feat, m in t1.items():
        mark = "YES" if m["z_beats_pca10_nonoverlap"] else "no"
        print(f"  {feat:<15} {mark}")


if __name__ == "__main__":
    main()
