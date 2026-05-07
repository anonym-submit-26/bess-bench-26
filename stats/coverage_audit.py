"""
coverage_audit.py — Reproductible Measure of Spectral Coverage by Line

Read paper/tables/metadata_slim.parquet (339,115 spectra, columns
``lambda_min`` and ``lambda_max`` per spectrum) and compute, for each line of interest,
 the fraction of spectra whose [lambda_min, lambda_max] strictly covers
the window +/-50 A around the line.


Lines measured:
  - Halpha   (6562.8 A)
  - Hbeta    (4861.3 A)
  - Hgamma   (4340.5 A)
  - Hdelta   (4101.7 A)
  - He I     (5876.0 A)
  - Fe II    (5169.0 A)
  - Na D     (5893.0 A)
  - O I      (7772.0 A)

out : /stats/results/coverage_audit.json (reproductible).

Usage :
    python -m stats.coverage_audit
    python stats/coverage_audit.py --output stats/results/coverage_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parent
DEFAULT_METADATA = BENCH_ROOT / "paper" / "tables" / "metadata_slim.parquet"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "coverage_audit.json"

sys.path.insert(0, str(BENCH_ROOT))
from stats.wandb_helpers import add_cli_args, finish, init_run, log_artifact  # noqa: E402

LINES = {
    "Halpha":    6562.8,
    "Hbeta":     4861.3,
    "Hgamma":    4340.5,
    "Hdelta":    4101.7,
    "HeI_5876":  5876.0,
    "FeII_5169": 5169.0,
    "NaD":       5893.0,
    "OI_7772":   7772.0,
}
HALF_WINDOW_AA = 50.0


def _quantiles(arr: np.ndarray) -> dict:
    qs = {5: "p5", 25: "p25", 50: "p50", 75: "p75", 95: "p95"}
    return {label: float(np.quantile(arr, q / 100)) for q, label in qs.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Measure true per-line spectral coverage from metadata_slim.parquet"
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                        help="Path to metadata_slim.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSON path")
    parser.add_argument("--half_window_AA", type=float, default=HALF_WINDOW_AA,
                        help="Half-window in angstroms (default: 50)")
    parser.add_argument("--echelle_orders", choices=["exclude", "include", "only"],
                        default="include",
                        help="How to treat is_echelle_order rows")
    add_cli_args(parser)
    args = parser.parse_args()

    print(f"Loading metadata from: {args.metadata}")
    df = pd.read_parquet(args.metadata)
    n_raw = len(df)
    print(f"  {n_raw:,} spectra loaded")

    # Handle echelle orders policy
    if "is_echelle_order" in df.columns:
        if args.echelle_orders == "exclude":
            df = df[~df["is_echelle_order"].astype(bool)].reset_index(drop=True)
        elif args.echelle_orders == "only":
            df = df[df["is_echelle_order"].astype(bool)].reset_index(drop=True)

    # Drop rows with missing wavelength bounds
    df = df.dropna(subset=["lambda_min", "lambda_max"]).reset_index(drop=True)
    lam_min = df["lambda_min"].to_numpy(dtype=np.float64)
    lam_max = df["lambda_max"].to_numpy(dtype=np.float64)
    n_valid = len(df)
    print(f"  {n_valid:,} spectra with valid lambda_min/lambda_max "
          f"({n_raw - n_valid:,} dropped)")

    # Coverage per line
    coverage = {}
    for name, lam0 in LINES.items():
        lo = lam0 - args.half_window_AA
        hi = lam0 + args.half_window_AA
        covered = (lam_min <= lo) & (lam_max >= hi)
        n_cov = int(covered.sum())
        coverage[name] = {
            "line_center_AA": lam0,
            "half_window_AA": args.half_window_AA,
            "window_AA": [lo, hi],
            "n_covered": n_cov,
            "n_total": n_valid,
            "fraction": round(n_cov / max(1, n_valid), 6),
            "percent": round(100.0 * n_cov / max(1, n_valid), 3),
        }

    lambda_stats = {
        "lambda_min_AA": {
            **_quantiles(lam_min),
            "min": float(lam_min.min()),
            "max": float(lam_min.max()),
            "mean": float(lam_min.mean()),
        },
        "lambda_max_AA": {
            **_quantiles(lam_max),
            "min": float(lam_max.min()),
            "max": float(lam_max.max()),
            "mean": float(lam_max.mean()),
        },
    }

    # Breakdown by echelle flag (informational)
    breakdown = None
    if "is_echelle_order" in df.columns:
        breakdown = {}
        for flag, label in [(False, "single_range"), (True, "echelle_order")]:
            sub = df[df["is_echelle_order"].astype(bool) == flag]
            sub_lm = sub["lambda_min"].to_numpy(dtype=np.float64)
            sub_lM = sub["lambda_max"].to_numpy(dtype=np.float64)
            sub_cov = {}
            for name, lam0 in LINES.items():
                lo = lam0 - args.half_window_AA
                hi = lam0 + args.half_window_AA
                c = int(((sub_lm <= lo) & (sub_lM >= hi)).sum()) if len(sub) else 0
                sub_cov[name] = {
                    "n_covered": c,
                    "percent": round(100.0 * c / max(1, len(sub)), 3),
                }
            breakdown[label] = {"n": int(len(sub)), "coverage": sub_cov}

    # ── Coverage by PHYSICAL OBSERVATION (star × MJD) ──
    # One echelle observation typically spans ~19–40 orders, each stored as a
    # separate parquet row. Grouping by (star_name, mjd) collapses an observation
    # to a single unit and aggregates wavelength bounds across its orders.
    physical_obs = None
    granularity = None
    if {"star_name", "mjd"}.issubset(df.columns):
        grp = df.groupby(["star_name", "mjd"], sort=False)
        obs_lm = grp["lambda_min"].min().to_numpy(dtype=np.float64)
        obs_lM = grp["lambda_max"].max().to_numpy(dtype=np.float64)
        n_obs = len(obs_lm)
        obs_cov = {}
        for name, lam0 in LINES.items():
            lo = lam0 - args.half_window_AA
            hi = lam0 + args.half_window_AA
            c = int(((obs_lm <= lo) & (obs_lM >= hi)).sum())
            obs_cov[name] = {
                "line_center_AA": lam0,
                "n_covered": c,
                "n_total": n_obs,
                "fraction": round(c / max(1, n_obs), 6),
                "percent": round(100.0 * c / max(1, n_obs), 3),
            }
        physical_obs = {
            "n_physical_observations": int(n_obs),
            "grouping": "(star_name, mjd)",
            "aggregation": "lambda_min=min, lambda_max=max across orders",
            "coverage_by_line": obs_cov,
        }

        # Granularity summary: rows / physical obs / unique stars
        granularity = {
            "n_rows": int(n_raw),
            "n_physical_observations": int(n_obs),
            "n_unique_stars": int(df["star_name"].nunique()),
            "single_range": {
                "n_rows": int((~df["is_echelle_order"].astype(bool)).sum()),
                "n_physical_observations": int(
                    df[~df["is_echelle_order"].astype(bool)]
                    .groupby(["star_name", "mjd"]).ngroups
                ),
                "n_unique_stars": int(
                    df[~df["is_echelle_order"].astype(bool)]["star_name"].nunique()
                ),
            },
            "echelle_order": {
                "n_rows": int(df["is_echelle_order"].astype(bool).sum()),
                "n_physical_observations": int(
                    df[df["is_echelle_order"].astype(bool)]
                    .groupby(["star_name", "mjd"]).ngroups
                ),
                "n_unique_stars": int(
                    df[df["is_echelle_order"].astype(bool)]["star_name"].nunique()
                ),
                "median_orders_per_observation": float(
                    df[df["is_echelle_order"].astype(bool)]
                    .groupby(["star_name", "mjd"]).size().median()
                ),
                "p5_orders_per_observation": float(
                    df[df["is_echelle_order"].astype(bool)]
                    .groupby(["star_name", "mjd"]).size().quantile(0.05)
                ),
                "p95_orders_per_observation": float(
                    df[df["is_echelle_order"].astype(bool)]
                    .groupby(["star_name", "mjd"]).size().quantile(0.95)
                ),
            },
        }

    # ── Observer type / resolution family breakdown ──
    observer_breakdown = None
    if "observer_type" in df.columns:
        observer_breakdown = {
            "by_observer_type": {
                str(k): {
                    "n_rows": int(v),
                    "n_physical_observations": int(
                        df[df["observer_type"] == k].groupby(["star_name", "mjd"]).ngroups
                    ),
                }
                for k, v in df["observer_type"].value_counts(dropna=False).items()
            },
        }
        if "spectral_resolution" in df.columns:
            def _family(row):
                if bool(row["is_echelle_order"]):
                    return "echelle_order"
                r = row["spectral_resolution"]
                if pd.isna(r):
                    return "single_unknown_R"
                if r < 1000:
                    return "single_R_lt_1000"
                if r < 3000:
                    return "single_R_1000_3000"
                if r < 8000:
                    return "single_R_3000_8000"
                return "single_R_ge_8000"

            fam = df.apply(_family, axis=1)
            fam_stats = {}
            for label in sorted(fam.unique()):
                mask = (fam == label)
                sub = df[mask]
                fam_stats[label] = {
                    "n_rows": int(mask.sum()),
                    "n_physical_observations": int(
                        sub.groupby(["star_name", "mjd"]).ngroups
                    ),
                    "n_unique_stars": int(sub["star_name"].nunique()),
                }
            observer_breakdown["by_resolution_family"] = fam_stats

    payload = {
        "dataset": "anonym-submit-26/bess-bench-26",
        "metadata_source": str(args.metadata.relative_to(BENCH_ROOT)
                               if args.metadata.is_absolute()
                               and BENCH_ROOT in args.metadata.parents
                               else args.metadata),
        "n_spectra_total": int(n_raw),
        "n_spectra_valid": int(n_valid),
        "half_window_AA": args.half_window_AA,
        "echelle_orders_policy": args.echelle_orders,
        "coverage_by_line": coverage,
        "coverage_by_physical_observation": physical_obs,
        "granularity": granularity,
        "observer_breakdown": observer_breakdown,
        "lambda_quantiles_AA": lambda_stats,
        "breakdown_by_kind": breakdown,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved: {args.output}\n")
    print("── Line coverage (strict window +/-{:.0f} A) ──".format(args.half_window_AA))
    for name, c in coverage.items():
        print(f"  {name:<11s} {c['percent']:7.3f}%  ({c['n_covered']:>7,}/{c['n_total']:,})")
    lm = lambda_stats["lambda_min_AA"]
    lM = lambda_stats["lambda_max_AA"]
    print(f"\n  lambda_min p5/p50/p95: {lm['p5']:.0f} / {lm['p50']:.0f} / {lm['p95']:.0f} A "
          f"(global min {lm['min']:.0f})")
    print(f"  lambda_max p5/p50/p95: {lM['p5']:.0f} / {lM['p50']:.0f} / {lM['p95']:.0f} A "
          f"(global max {lM['max']:.0f})")

    if physical_obs is not None:
        print(
            f"\n── Coverage per PHYSICAL OBSERVATION (grouped by star × MJD, "
            f"N = {physical_obs['n_physical_observations']:,}) ──"
        )
        for name, c in physical_obs["coverage_by_line"].items():
            print(
                f"  {name:<11s} {c['percent']:7.3f}%  "
                f"({c['n_covered']:>6,}/{c['n_total']:,})"
            )
    if granularity is not None:
        g = granularity
        print(
            f"\n  Granularity: {g['n_rows']:,} rows  →  "
            f"{g['n_physical_observations']:,} physical observations  →  "
            f"{g['n_unique_stars']:,} unique stars"
        )
        print(
            f"    single_range : {g['single_range']['n_rows']:,} rows = "
            f"{g['single_range']['n_physical_observations']:,} obs "
            f"({g['single_range']['n_unique_stars']:,} stars)"
        )
        print(
            f"    echelle_order: {g['echelle_order']['n_rows']:,} rows = "
            f"{g['echelle_order']['n_physical_observations']:,} obs "
            f"({g['echelle_order']['n_unique_stars']:,} stars, "
            f"median {g['echelle_order']['median_orders_per_observation']:.0f} orders/obs)"
        )

    # ── W&B logging ──
    tags = (args.wandb_tags or []) + ["phase1-coverage"]
    run = init_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"coverage_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=list(dict.fromkeys(tags)),
        mode=args.wandb_mode,
        group=args.wandb_group,
        config={
            "half_window_AA": args.half_window_AA,
            "n_spectra_total": n_raw,
            "n_spectra_valid": n_valid,
            "echelle_orders_policy": args.echelle_orders,
            "phase": "1",
            "task": "coverage_audit",
        },
    )
    metrics = {f"coverage_{name}_pct": c["percent"] for name, c in coverage.items()}
    metrics.update({f"coverage_{name}_n": c["n_covered"] for name, c in coverage.items()})
    metrics["lambda_min_p50"] = lm["p50"]
    metrics["lambda_max_p50"] = lM["p50"]
    if physical_obs is not None:
        metrics["n_physical_observations"] = physical_obs["n_physical_observations"]
        for name, c in physical_obs["coverage_by_line"].items():
            metrics[f"coverage_obs_{name}_pct"] = c["percent"]
            metrics[f"coverage_obs_{name}_n"] = c["n_covered"]
    if granularity is not None:
        metrics["n_unique_stars"] = granularity["n_unique_stars"]
    run.log(metrics)
    log_artifact(run, args.output, name="coverage_audit",
                 type="audit", description="Per-line spectral coverage audit")
    finish(run)


if __name__ == "__main__":
    main()
