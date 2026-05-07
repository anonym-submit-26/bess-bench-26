#!/usr/bin/env python3
"""
aggregate_results.py — Aggregates result JSONs and produces
table LaTeX for the paper.

Inputs (read from bess_bench/):
  stage1_encoder/runs/Halpha_all/eval_results.json   (BeMAE pretrain)
  reference_results/downstream/probe_features_results.json     (T1 probe features)
  reference_results/downstream/t3_pca_ridge_temporal_baseline.json (T3 PCA+Ridge)
  reference_results/downstream/t3_ts_fm_baselines.json          (T3 TS-FM zero-shot, optional)

Outputs:
  paper/tables/table_stage1.tex
  paper/tables/table_t1_probes.tex
  paper/tables/table_t3_baselines.tex
  paper/tables/summary.json   (all figures exported as machine-readable)

All references to the defunct ``stage2_temporal/`` (custom STFM transformer,
T3 negative result via seeds) have been removed: T3 baselines
are now PCA+Ridge + time-series foundation models zero-shot.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Output dir: defaults to paper/tables/ (canonical committed copy);
# override via BESS_TABLES_DIR for reproduction runs.
OUT_TABLES = Path(os.environ.get(
    "BESS_TABLES_DIR", str(ROOT / "paper" / "tables")))
OUT_TABLES.mkdir(parents=True, exist_ok=True)
# Source for downstream JSONs (probe_features_results_seed*, t3_*).
# Defaults to the committed reference_results/ but can be redirected to
# results_reproduced/downstream/ to render tables from a fresh repro run.
DOWNSTREAM_DIR = Path(os.environ.get(
    "BESS_DOWNSTREAM_DIR", str(ROOT / "reference_results" / "downstream")))


def load_json(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)


# ─── Stage 1 encoder (BeMAE) ────────────────────────────────────────────────
def table_stage1() -> tuple[str, dict]:
    """Multi-seed aggregate over seeds {42,123,456}.

    Prefers ``runs/multiseed_summary.json`` (new pipeline). Falls back to the
    legacy single-run ``runs/Halpha_all/eval_results.json`` if present.
    """
    multi = ROOT / "reference_results" / "stage1_encoder" / "multiseed_summary.json"
    if multi.exists():
        d = load_json(multi)
        agg = d["aggregate"]
        def fmt(key):
            a = agg.get(key, {})
            m, s, n = a.get("mean"), a.get("std"), a.get("n", 0)
            if m is None:
                return "n/a", {"mean": None, "std": None, "n": n}
            return f"{m:.3f} $\\pm$ {s:.3f}", {"mean": m, "std": s, "n": n}
        label_keys = [
            ("kNN intra-star acc.",                     "knn_intra_star"),
            ("Recall@10 (held-out)",                    "recall_at_10_test"),
        ]
        metrics = {}
        rows_tex = []
        for label, key in label_keys:
            txt, val = fmt(key)
            metrics[label] = val
            rows_tex.append(f"  {label.ljust(26)} & {txt} \\\\")
        rows = "\n".join(rows_tex)
        tex = (
            "\\begin{tabular}{lr}\n"
            "  \\toprule\n"
            "  Metric & Value (mean $\\pm$ std, 3 seeds) \\\\\n"
            "  \\midrule\n"
            f"{rows}\n"
            "  \\bottomrule\n"
            "\\end{tabular}\n"
        )
        metrics["_seeds"] = d.get("seeds", [])
        metrics["_per_seed"] = d.get("per_seed", [])
        return tex, metrics

    # Fallback: legacy single-seed path
    d = load_json(ROOT / "reference_results" / "stage1_encoder" / "Halpha_all_seed42" / "eval_results.json")
    metrics = {
        "R^2 CV (EW, all stars)":   d["r2_cv_ew_all"]["r2_mean"],
        "R^2 CV median (all)":      d["r2_cv_ew_all"]["r2_median"],
        "kNN intra-star acc.":      d["knn_intra_star"]["mean_accuracy"],
        "Centroid cos. sim. (mean)": d["cos_centroids"]["cos_mean"],
        "Recall@10 (held-out)":     d["recall_at_10_test"],
    }
    rows = "\n".join(
        f"  {k.ljust(26)} & {v:.3f} \\\\"
        for k, v in metrics.items()
    )
    tex = (
        "\\begin{tabular}{lr}\n"
        "  \\toprule\n"
        "  Metric & Test value \\\\\n"
        "  \\midrule\n"
        f"{rows}\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )
    return tex, metrics


# ─── T1 probes (frozen embeddings vs PCA) ────────────────────────────────────
def table_t1_probes() -> tuple[str, dict]:
    """Aggregate T1 probes over encoder seeds {42,123,456} if per-seed files exist."""
    res_dir = DOWNSTREAM_DIR
    seed_files = [res_dir / f"probe_features_results_seed{s}.json" for s in (42, 123, 456)]
    seed_files = [p for p in seed_files if p.exists()]

    if seed_files:
        per_seed = [load_json(p) for p in seed_files]
        # All files share the same schema: d["results"][feat][rep] = {r2_mean, r2_std, ...}
        # We aggregate the per-fold mean across encoder seeds (mean of means, std across seeds).
        def get(d, feat, rep):
            return d["results"][feat][rep]
        schema_src = per_seed[0]
        n_seeds = len(per_seed)
    else:
        d = load_json(res_dir / "probe_features_results.json")
        per_seed = [d]
        def get(dd, feat, rep):
            return dd["results"][feat][rep]
        schema_src = d
        n_seeds = 1

    features = ["fwhm", "central_depth", "delta_v", "vr_ratio"]  # skip ew/peak (trivially linear in flux)
    reps = ["z (128D)", "PCA(10)", "PCA(32)", "PCA(50)"]
    import statistics as _stats

    summary = {}
    lines = []
    for feat in features:
        cell = {}
        cell_tex = []
        for rep in reps:
            means = [get(dd, feat, rep)["r2_mean"] for dd in per_seed]
            m = _stats.mean(means)
            s = _stats.stdev(means) if len(means) > 1 else get(per_seed[0], feat, rep)["r2_std"]
            cell[rep] = {"r2_mean": m, "r2_std": s, "n_seeds": n_seeds}
            cell_tex.append(f"{m:.3f} $\\pm$ {s:.3f}")
        summary[feat] = {
            "z":     cell["z (128D)"],
            "pca10": cell["PCA(10)"],
            "pca32": cell["PCA(32)"],
            "pca50": cell["PCA(50)"],
        }
        lines.append(f"  {feat.replace('_', chr(92)+'_'):<20} & " + " & ".join(cell_tex) + " \\\\")

    suffix = (
        f"% R^2, mean over {n_seeds} encoder seeds $\\pm$ std across seeds. "
        if n_seeds > 1
        else "% R^2 (single encoder seed), mean $\\pm$ std across CV folds. "
    )
    tex = (
        "\\begin{tabular}{lcccc}\n"
        "  \\toprule\n"
        "  Feature & Embedding $z$ (128D) & PCA(10) & PCA(32) & PCA(50) \\\\\n"
        "  \\midrule\n"
        + "\n".join(lines) + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
        + suffix
        + "PCA(32) matches the dimensionality of a linear probe on $z$.\n"
    )
    return tex, summary


# ─── T3 PCA+Ridge + zero-shot TS-FM baselines ──────────────────────
def table_t3_baselines() -> tuple[str, dict]:
    d = load_json(DOWNSTREAM_DIR / "t3_pca_ridge_temporal_baseline.json")
    tsfm_path = DOWNSTREAM_DIR / "t3_ts_fm_baselines.json"
    d_tsfm = load_json(tsfm_path) if tsfm_path.exists() else {}

    summary = {}
    lines = []
    for cfg_name in ["pca10_ctx5", "pca10_ctx10", "pca50_ctx5", "pca50_ctx10"]:
        g = d[cfg_name]["global"]
        c = d[cfg_name]["config"]
        summary[cfg_name] = {
            "mae_model":   g["mae_model"],
            "mae_persist": g["mae_persist"],
            "ratio_mae":   g["ratio_mae"],
            "r2_model":    g["r2_model"],
            "n_pca":       c["n_pca"],
            "context_len": c["context_len"],
        }
        bolded = (
            f"\\textbf{{{g['ratio_mae']:.3f}}}"
            if g["ratio_mae"] < 1.0 else f"{g['ratio_mae']:.3f}"
        )
        lines.append(
            f"  PCA({c['n_pca']}) ctx={c['context_len']} & "
            f"{g['mae_model']:.3f} & {g['mae_persist']:.3f} & "
            f"{bolded} & {g['r2_model']:.3f} \\\\"
        )

    tsfm_lines = []
    tsfm_display = [
        ("chronos_bolt_small", "Chronos-Bolt-S (48\\,M)"),
        ("chronos_bolt_base",  "Chronos-Bolt-B (205\\,M)"),
        ("timesfm_2_0_500m",   "TimesFM-2.0 (500\\,M)"),
    ]
    for key, label in tsfm_display:
        if key not in d_tsfm:
            continue
        g = d_tsfm[key]["global"]
        c = d_tsfm[key]["config"]
        summary[key] = {
            "mae_model":   g["mae_model"],
            "mae_persist": g["mae_persist"],
            "ratio_mae":   g["ratio_mae"],
            "params_M":    c.get("params_M"),
            "max_context": c.get("max_context"),
        }
        bolded = (
            f"\\textbf{{{g['ratio_mae']:.3f}}}"
            if g["ratio_mae"] < 1.0 else f"{g['ratio_mae']:.3f}"
        )
        tsfm_lines.append(
            f"  {label} & "
            f"{g['mae_model']:.3f} & {g['mae_persist']:.3f} & "
            f"{bolded} & -- \\\\"
        )

    # ── BeMAE z(128D) + Ridge baseline (full encoder embedding, no PCA) ──
    z_lines = []
    z_path = DOWNSTREAM_DIR / "t3_z_ridge_temporal_baseline_ctx5.json"
    if z_path.exists():
        dz = load_json(z_path)
        # Single-seed schema: per_seed["42"]["global"]
        gz = dz["per_seed"][str(dz["seeds"][0])]["global"]
        cz = dz["per_seed"][str(dz["seeds"][0])]["config"]
        summary["z128_ctx5"] = {
            "mae_model":   gz["mae_model"],
            "mae_persist": gz["mae_persist"],
            "ratio_mae":   gz["ratio_mae"],
            "ratio_mae_ci95": gz.get("ratio_mae_ci95"),
            "r2_model":    gz["r2_model"],
            "context_len": cz["context_len"],
            "n_features":  128,
        }
        bolded = (
            f"\\textbf{{{gz['ratio_mae']:.3f}}}"
            if gz["ratio_mae"] < 1.0 else f"{gz['ratio_mae']:.3f}"
        )
        z_lines.append(
            f"  $z_{{128}}$+Ridge ctx={cz['context_len']} & "
            f"{gz['mae_model']:.3f} & {gz['mae_persist']:.3f} & "
            f"{bolded} & {gz['r2_model']:.3f} \\\\"
        )

    body = "\n".join(lines)
    if z_lines:
        body += "\n  \\midrule\n" + "\n".join(z_lines)
    if tsfm_lines:
        body += "\n  \\midrule\n" + "\n".join(tsfm_lines)

    tex = (
        "\\begin{tabular}{lrrrr}\n"
        "  \\toprule\n"
        "  Baseline & MAE (model) & MAE (persist) & Ratio & $R^2$ \\\\\n"
        "  \\midrule\n"
        + body + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
        "% PCA(10)+Ridge with context 5 dominates every baseline, including "
        "the larger BeMAE $z_{128}$+Ridge encoder representation and "
        "the much larger zero-shot Chronos-Bolt and TimesFM-2.0 models. "
        "$R^2$ unavailable for zero-shot foundation models. Bold = ratio $<$ 1.\n"
    )
    return tex, summary


def main():
    bundles = {}
    for name, fn in [
        ("stage1",        table_stage1),
        ("t1_probes",     table_t1_probes),
        ("t3_baselines",  table_t3_baselines),
    ]:
        tex, data = fn()
        out_tex = OUT_TABLES / f"table_{name}.tex"
        out_tex.write_text(tex)
        print(f"[+] {out_tex}")
        bundles[name] = data

    # Dump a machine-readable summary so the paper and the figures can pull
    # from a single source of truth.
    summary_path = OUT_TABLES / "summary.json"
    summary_path.write_text(json.dumps(bundles, indent=2))
    print(f"[+] {summary_path}")

    print("\n[=] Headline numbers:")
    t3b = bundles["t3_baselines"]
    best = min(t3b.items(), key=lambda kv: kv[1]["ratio_mae"])
    print(f"  Best PCA+Ridge   : {best[0]} → ratio {best[1]['ratio_mae']:.3f}")


if __name__ == "__main__":
    main()
