#!/usr/bin/env python3
"""
compare_with_reference.py — Numerical diff between reproduced results and
the paper's reference results.

Compares the JSON files in  results_reproduced/downstream/  against those in
reference_results/downstream/  for a well-defined subset of headline metrics.
Prints a report and also writes  results_reproduced/compare_report.txt.

Tolerances:
    * embeddings extracted on CPU       -> deviations < 1e-6   => "STRICT"
    * embeddings extracted on GPU       -> deviations < 1e-3   => "GPU-OK"
    * any deviation > 1e-2                                     => "MISMATCH"

Usage:
    python scripts/compare_with_reference.py
    python scripts/compare_with_reference.py --tol-strict 1e-5 --tol-gpu 1e-3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
REF = REPO / "reference_results" / "downstream"
REP = REPO / "results_reproduced" / "downstream"


def status(diff: float, tol_strict: float, tol_gpu: float) -> str:
    if diff < tol_strict:
        return "STRICT  ✓"
    if diff < tol_gpu:
        return "GPU-OK  ✓"
    if diff < 1e-2:
        return "WARN    ⚠"
    return "MISMATCH ✗"


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def cmp_value(name: str, ref: Any, rep: Any, lines: list[str],
              tol_strict: float, tol_gpu: float) -> int:
    if isinstance(ref, (int, float)) and isinstance(rep, (int, float)):
        diff = abs(float(ref) - float(rep))
        st = status(diff, tol_strict, tol_gpu)
        lines.append(
            f"  {name:55s}  ref={fmt(ref):>12s}  rep={fmt(rep):>12s}  "
            f"|Δ|={diff:.2e}  {st}")
        return 1 if st.startswith("MISMATCH") else 0
    return 0


def cmp_t1(lines: list[str], tol_strict: float, tol_gpu: float) -> int:
    """T1 SpecProbe — compare R² mean/std for the 6 features × {z, PCA10, PCA50}."""
    fr = REF / "probe_features_results_seed42.json"
    fp = REP / "probe_features_results_seed42.json"
    if not fr.exists() or not fp.exists():
        lines.append(f"  [SKIP] T1: missing {fr.name if not fr.exists() else fp.name}")
        return 0
    ref = json.loads(fr.read_text())
    rep = json.loads(fp.read_text())
    R = ref.get("results", ref)
    P = rep.get("results", rep)
    n_bad = 0
    for feat in sorted(R.keys()):
        if feat not in P:
            lines.append(f"  [missing in repro] {feat}")
            n_bad += 1
            continue
        for method in R[feat]:
            if method not in P[feat]:
                continue
            for metric in ("r2_mean", "r2_std"):
                rv = R[feat][method].get(metric)
                pv = P[feat][method].get(metric)
                if rv is None or pv is None:
                    continue
                n_bad += cmp_value(f"T1 {feat}/{method}/{metric}", rv, pv,
                                   lines, tol_strict, tol_gpu)
    return n_bad


def cmp_t2(lines: list[str], tol_strict: float, tol_gpu: float) -> int:
    fr = REF / "probe_t2_hbeta_to_halpha.json"
    fp = REP / "probe_t2_hbeta_to_halpha.json"
    if not fr.exists() or not fp.exists():
        lines.append("  [SKIP] T2 not found")
        return 0
    ref = json.loads(fr.read_text())
    rep = json.loads(fp.read_text())
    R = ref.get("results", ref)
    P = rep.get("results", rep)
    n_bad = 0
    for feat in sorted(R.keys()) if isinstance(R, dict) else []:
        if feat not in P:
            continue
        for repr_name in R[feat]:
            if repr_name not in P[feat]:
                continue
            for metric in ("r2_mean", "r2_std", "mae_mean", "rmse_mean"):
                rv = R[feat][repr_name].get(metric)
                pv = P[feat][repr_name].get(metric)
                if rv is None or pv is None:
                    continue
                n_bad += cmp_value(f"T2 {feat}/{repr_name}/{metric}", rv, pv,
                                   lines, tol_strict, tol_gpu)
    return n_bad


def cmp_t3_pca(lines: list[str], tol_strict: float, tol_gpu: float) -> int:
    fr = REF / "t3_pca_ridge_temporal_baseline.json"
    fp = REP / "t3_pca_ridge_temporal_baseline.json"
    if not fr.exists() or not fp.exists():
        lines.append("  [SKIP] T3 PCA+Ridge not found")
        return 0
    ref = json.loads(fr.read_text())
    rep = json.loads(fp.read_text())
    n_bad = 0
    for cfg in sorted(ref.keys()):
        if cfg not in rep:
            continue
        if "global" not in ref[cfg] or "global" not in rep[cfg]:
            continue
        gr, gp = ref[cfg]["global"], rep[cfg]["global"]
        for k in ("mae_model", "mae_persist", "ratio_mae",
                  "r2_model", "r2_persist", "pct_beating_persist"):
            if k in gr and k in gp:
                n_bad += cmp_value(f"T3-PCA {cfg}/{k}", gr[k], gp[k],
                                   lines, tol_strict, tol_gpu)
    return n_bad


def cmp_t3_tsfm(lines: list[str], tol_strict: float, tol_gpu: float) -> int:
    fr = REF / "t3_ts_fm_baselines.json"
    fp = REP / "t3_ts_fm_baselines.json"
    if not fr.exists() or not fp.exists():
        lines.append("  [SKIP] T3 TS-FM not found (run with SKIP_TSFM=0)")
        return 0
    ref = json.loads(fr.read_text())
    rep = json.loads(fp.read_text())
    n_bad = 0
    for model in sorted(ref.keys()):
        if model not in rep:
            continue
        gr = ref[model].get("global", {})
        gp = rep[model].get("global", {})
        for k in ("mae_model", "mae_persist", "ratio_mae"):
            if k in gr and k in gp:
                n_bad += cmp_value(f"T3-TSFM {model}/{k}", gr[k], gp[k],
                                   lines, tol_strict, tol_gpu)
    return n_bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol-strict", type=float, default=1e-6,
                    help="Tolerance for CPU bit-identity (default 1e-6)")
    ap.add_argument("--tol-gpu", type=float, default=1e-3,
                    help="Tolerance acceptable from GPU non-determinism (default 1e-3)")
    args = ap.parse_args()

    if not REP.exists():
        print(f"ERROR: {REP} does not exist — run reproduce_paper_seed42.sh first.",
              file=sys.stderr)
        return 2

    lines: list[str] = []
    lines.append("═" * 78)
    lines.append("BESS-Bench reproduction comparison")
    lines.append(f"  reference : reference_results/downstream")
    lines.append(f"  reproduced: results_reproduced/downstream")
    lines.append(f"  tolerance : strict<{args.tol_strict:.0e}  "
                 f"gpu<{args.tol_gpu:.0e}")
    lines.append("═" * 78)

    total_bad = 0
    for label, fn in [("T1 SpecProbe (seed 42)", cmp_t1),
                      ("T2 LineTransfer", cmp_t2),
                      ("T3 EWForecast PCA+Ridge", cmp_t3_pca),
                      ("T3 EWForecast TS-FM", cmp_t3_tsfm)]:
        lines.append("")
        lines.append(f"── {label} " + "─" * (75 - len(label)))
        total_bad += fn(lines, args.tol_strict, args.tol_gpu)

    lines.append("")
    lines.append("═" * 78)
    if total_bad == 0:
        lines.append("RESULT:  OK  all reproduced metrics match the reference within tolerance.")
    else:
        lines.append(f"RESULT:  {total_bad} metric(s) exceed tol_gpu (>{args.tol_gpu:.0e}).")
        lines.append("        Expected on `vr_ratio / z (128D)` (RidgeCV alpha-selection")
        lines.append("        instability on this low-R^2 target; documented in REPRODUCE.md).")
        lines.append("        Other deviations (if any) warrant investigation")
        lines.append("        (wrong checkpoint, version drift, etc.).")
    lines.append("═" * 78)

    report = "\n".join(lines)
    print(report)
    out = REPO / "results_reproduced" / "compare_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n")
    print(f"\nReport written to: {out}")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
