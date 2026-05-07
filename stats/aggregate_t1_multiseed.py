"""Aggregate T1 multi-seed probe_features results -> stats/results/t1_multiseed_summary.json."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parent
RESULTS_IN = BENCH_ROOT / "reference_results" / "downstream"
RESULTS_OUT = BENCH_ROOT / "reference_results" / "stats" / "t1_multiseed_summary.json"

sys.path.insert(0, str(BENCH_ROOT))
from stats.wandb_helpers import add_cli_args, finish, init_run, log_artifact  # noqa: E402


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    add_cli_args(parser)
    args = parser.parse_args()

    agg = {}
    n_aligned = None
    for s in args.seeds:
        p = RESULTS_IN / f"probe_features_results_seed{s}.json"
        if not p.exists():
            print(f"WARNING: {p} not found, skipping.")
            continue
        d = json.load(open(p))
        if n_aligned is None:
            n_aligned = d.get("n_aligned_pairs")
        for feat, reps in d["results"].items():
            for rep, v in reps.items():
                agg.setdefault((feat, rep), []).append((s, v["r2_mean"], v.get("r2_std")))

    print(f"{'feature':20s} | {'repr':14s} | mean R² (across seeds)   | std    | CI95 mean")
    print("-" * 100)
    summary = {}
    for (feat, rep), triples in sorted(agg.items()):
        seeds_used = [t[0] for t in triples]
        arr = np.asarray([t[1] for t in triples], dtype=np.float64)
        mean, std = float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        # bootstrap CI95 on mean
        rng = np.random.default_rng(42)
        boots = np.array([arr[rng.integers(0, len(arr), size=len(arr))].mean()
                          for _ in range(10000)])
        ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        print(f"{feat:20s} | {rep:14s} | {mean:+.4f} (n={len(arr)})          | "
              f"{std:.4f} | [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        summary[f"{feat}__{rep}"] = {
            "seeds": seeds_used,
            "r2_per_seed": arr.tolist(),
            "mean": mean,
            "std": std,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "ci95_mean_bootstrap": [ci_lo, ci_hi],
        }

    # ── Aggregated test: R²(z) / R²(PCA10) ratio on non-trivial features ──
    # Replaces the "3 disjoint CI95" (FWER ~14%) with a single aggregated bootstrap test.
    headline_aggregated = None
    head_features = ["fwhm", "central_depth", "delta_v"]
    z_keys = [f"{f}__z (128D)" for f in head_features]
    pca_keys = [f"{f}__PCA(10)" for f in head_features]
    if all(k in summary for k in z_keys + pca_keys):
        # Align seeds: assume the order list is identical for all keys
        seeds_ref = summary[z_keys[0]]["seeds"]
        # mean R² (sur les 3 features) par seed, pour z et PCA(10)
        z_mat = np.asarray(
            [summary[k]["r2_per_seed"] for k in z_keys], dtype=np.float64
        )  # shape (3 features, n_seeds)
        pca_mat = np.asarray(
            [summary[k]["r2_per_seed"] for k in pca_keys], dtype=np.float64
        )
        z_per_seed = z_mat.mean(axis=0)  # shape (n_seeds,)
        pca_per_seed = pca_mat.mean(axis=0)
        # Ratio per seed (pairing preserved: same seed → same fold splits)
        ratios = z_per_seed / pca_per_seed  # shape (n_seeds,)
        # Bootstrap CI95 sur les ratios (resample seeds with replacement)
        rng = np.random.default_rng(20260428)
        n_boot = 10000
        boot_ratios = np.empty(n_boot)
        n_seeds = len(ratios)
        for b in range(n_boot):
            idx = rng.integers(0, n_seeds, size=n_seeds)
            boot_ratios[b] = ratios[idx].mean()
        ci_lo = float(np.percentile(boot_ratios, 2.5))
        ci_hi = float(np.percentile(boot_ratios, 97.5))
        headline_aggregated = {
            "features": head_features,
            "comparison": "z (128D) vs PCA(10)",
            "metric": "ratio of mean R² across {fwhm, central_depth, delta_v}",
            "n_seeds": int(n_seeds),
            "n_bootstrap": n_boot,
            "z_mean_r2_per_seed": z_per_seed.tolist(),
            "pca10_mean_r2_per_seed": pca_per_seed.tolist(),
            "ratio_per_seed": ratios.tolist(),
            "ratio_point": float(ratios.mean()),
            "ratio_ci95": [ci_lo, ci_hi],
            "ratio_significantly_above_1": bool(ci_lo > 1.0),
            "rationale": (
                "Replaces the per-feature 'disjoint CI95' test (FWER ~14% over 3 "
                "features without correction) by a single aggregate test on the "
                "ratio of mean R². Bootstrap is over seeds (n=3) which is the "
                "true population unit of variability; folds are nested within."
            ),
        }
        print("\n" + "=" * 70)
        print("HEADLINE AGGREGATED RATIO (z vs PCA(10) on non-trivial features)")
        print("=" * 70)
        print(f"  features        : {head_features}")
        print(f"  ratio (point)   : {ratios.mean():.4f}")
        print(f"  ratio CI95      : [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  significant > 1 : {ci_lo > 1.0}")
    else:
        missing = [k for k in z_keys + pca_keys if k not in summary]
        print(f"WARNING: cannot compute aggregated ratio, missing keys: {missing}")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "seeds": args.seeds,
        "n_folds": 5,
        "n_aligned_pairs": n_aligned,
        "cv_shuffle": True,
        "summary": summary,
        "headline_aggregated_ratio": headline_aggregated,
    }
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {RESULTS_OUT}")

    tags = (args.wandb_tags or []) + ["phase3-t1-multiseed", "aggregate"]
    run = init_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"t1_multiseed_agg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=list(dict.fromkeys(tags)),
        mode=args.wandb_mode,
        group=args.wandb_group or "t1_multiseed_agg",
        config={"phase": "3", "task": "T1_multiseed_aggregate",
                "seeds": args.seeds, "n_folds": 5,
                "n_aligned_pairs": n_aligned},
    )
    wb_metrics = {}
    for k, v in summary.items():
        wb_metrics[f"{k}/mean"] = v["mean"]
        wb_metrics[f"{k}/std"] = v["std"]
        wb_metrics[f"{k}/ci95_lo"] = v["ci95_mean_bootstrap"][0]
        wb_metrics[f"{k}/ci95_hi"] = v["ci95_mean_bootstrap"][1]
    if headline_aggregated is not None:
        wb_metrics["headline_aggregated/ratio_point"] = headline_aggregated["ratio_point"]
        wb_metrics["headline_aggregated/ratio_ci95_lo"] = headline_aggregated["ratio_ci95"][0]
        wb_metrics["headline_aggregated/ratio_ci95_hi"] = headline_aggregated["ratio_ci95"][1]
    run.log(wb_metrics)
    log_artifact(run, RESULTS_OUT, name="t1_multiseed_summary", type="stats",
                 description="T1 multi-seed aggregate (mean, std, bootstrap CI95)")
    finish(run)


if __name__ == "__main__":
    main()
