"""
bootstrap_ci_t3.py — Bootstrap CI95 pour T3 forecasting EW(Hα).

Consolidates T3 results into a single protocol-authoritative JSON:
  1. PCA+Ridge baselines : lit
     ``downstream/results/t3_pca_ridge_temporal_baseline.json`` (predictions
     per test pair) and computes a non-parametric bootstrap
     (1000 resamples, CI95 percentile 2.5/97.5) sur le ratio MAE
     model/persistence.
  2. Time-series foundation models zero-shot (Chronos-Bolt, TimesFM…) :
     reads ``downstream/results/t3_ts_fm_baselines.json`` (predictions per
     test pair) and applies the same bootstrap.

The decision rule (``beats persistence``) is CI95 strictly < 1.0.

Sortie : ``stats/results/t3_bootstrap_cis.json``.

Usage :
    conda run -n envglobal python stats/bootstrap_ci_t3.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parent
DEFAULT_PCA_RIDGE = BENCH_ROOT / "reference_results" / "downstream" / "t3_pca_ridge_temporal_baseline.json"
DEFAULT_TS_FM = BENCH_ROOT / "reference_results" / "downstream" / "t3_ts_fm_baselines.json"
DEFAULT_OUTPUT = BENCH_ROOT / "reference_results" / "stats" / "t3_bootstrap_cis.json"

sys.path.insert(0, str(BENCH_ROOT))
from stats.wandb_helpers import add_cli_args, finish, init_run, log_artifact  # noqa: E402


def bootstrap_ratio_mae(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_persist: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
    ci: float = 95.0,
) -> dict:
    """Non-parametric bootstrap of the MAE(model)/MAE(persist) ratio."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_model = np.asarray(y_pred_model, dtype=np.float64)
    y_pred_persist = np.asarray(y_pred_persist, dtype=np.float64)

    mae_model_point = float(np.mean(np.abs(y_pred_model - y_true)))
    mae_persist_point = float(np.mean(np.abs(y_pred_persist - y_true)))
    ratio_point = mae_model_point / max(mae_persist_point, 1e-10)

    ratios = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = float(np.mean(np.abs(y_pred_model[idx] - y_true[idx])))
        p = float(np.mean(np.abs(y_pred_persist[idx] - y_true[idx])))
        ratios[b] = m / max(p, 1e-10)

    lo_q = (100.0 - ci) / 2.0
    hi_q = 100.0 - lo_q
    return {
        "n_predictions": n,
        "n_bootstrap": n_boot,
        "ci_level": ci,
        "ratio_mae_point": ratio_point,
        "mae_model_point": mae_model_point,
        "mae_persist_point": mae_persist_point,
        "ratio_mae_boot_mean": float(np.mean(ratios)),
        "ratio_mae_boot_median": float(np.median(ratios)),
        "ratio_mae_ci_lo": float(np.percentile(ratios, lo_q)),
        "ratio_mae_ci_hi": float(np.percentile(ratios, hi_q)),
        "frac_ratios_lt_1": float(np.mean(ratios < 1.0)),
    }


def load_prediction_dict(path: Path) -> dict:
    """Load a JSON where each key exposes ``config`` and ``predictions``.

    Expected format for each entry ``d[key]``:
      - ``d[key]["config"]``      : metadata dict (n_test, context_len, …)
      - ``d[key]["predictions"]`` : dict avec ``y_true``, ``y_pred_model``,
                                    ``y_pred_persist`` (arrays of equal length)
    """
    with open(path) as f:
        data = json.load(f)
    out = {}
    for key, cfg_res in data.items():
        if not isinstance(cfg_res, dict):
            continue
        preds = cfg_res.get("predictions")
        if preds is None:
            continue
        cfg = cfg_res.get("config", {})
        out[key] = {
            "y_true": np.asarray(preds["y_true"], dtype=np.float64),
            "y_pred_model": np.asarray(preds["y_pred_model"], dtype=np.float64),
            "y_pred_persist": np.asarray(preds["y_pred_persist"], dtype=np.float64),
            "n_test": int(cfg.get("n_test", len(preds["y_true"]))),
            "config": cfg,
        }
    return out


def compute_bootstrap_group(pred_by_cfg: dict, n_boot: int, seed: int,
                            label: str) -> dict:
    """Apply ``bootstrap_ratio_mae`` to each config and log results."""
    out: dict = {}
    for cfg_key, d in pred_by_cfg.items():
        cfg_note = ", ".join(f"{k}={v}" for k, v in d["config"].items()
                             if k in ("n_pca", "context_len", "model", "horizon"))
        print(f"\n   ── [{label}] {cfg_key} (n={d['n_test']}, {cfg_note}) ──")
        ci = bootstrap_ratio_mae(
            y_true=d["y_true"],
            y_pred_model=d["y_pred_model"],
            y_pred_persist=d["y_pred_persist"],
            n_boot=n_boot,
            seed=seed,
        )
        out[cfg_key] = {**ci, "config": d["config"]}
        print(f"     ratio_mae (point) : {ci['ratio_mae_point']:.4f}")
        print(f"     ratio_mae (boot)  : {ci['ratio_mae_boot_median']:.4f} "
              f"[{ci['ratio_mae_ci_lo']:.4f}, {ci['ratio_mae_ci_hi']:.4f}]")
        beats = ci["ratio_mae_ci_hi"] < 1.0
        fails = ci["ratio_mae_ci_lo"] > 1.0
        print(f"     CI excludes 1.0 LEFT  (beats persistence)  : {beats}")
        print(f"     CI excludes 1.0 RIGHT (fails persistence)  : {fails}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap CI95 for T3 forecasting (PCA+Ridge + TS-FM baselines)"
    )
    parser.add_argument("--pca_ridge_results", type=Path, default=DEFAULT_PCA_RIDGE)
    parser.add_argument("--ts_fm_results", type=Path, default=DEFAULT_TS_FM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    add_cli_args(parser)
    args = parser.parse_args()

    print("=" * 70)
    print("BOOTSTRAP CI95 — T3 SHORT-HORIZON EW(Hα) FORECASTING")
    print("=" * 70)

    # ── 1. PCA+Ridge bootstrap ──
    print(f"\n1. Loading PCA+Ridge predictions from {args.pca_ridge_results.name}...")
    if args.pca_ridge_results.exists():
        pr_preds = load_prediction_dict(args.pca_ridge_results)
        pca_ridge_cis = compute_bootstrap_group(
            pr_preds, args.n_boot, args.bootstrap_seed, "PCA+Ridge")
    else:
        print(f"   ⚠ File not found: {args.pca_ridge_results}")
        pca_ridge_cis = {}

    # ── 2. TS-FM (Chronos-Bolt, TimesFM…) bootstrap ──
    print(f"\n2. Loading TS-FM predictions from {args.ts_fm_results.name}...")
    if args.ts_fm_results.exists():
        tf_preds = load_prediction_dict(args.ts_fm_results)
        ts_fm_cis = compute_bootstrap_group(
            tf_preds, args.n_boot, args.bootstrap_seed, "TS-FM")
    else:
        print(f"   ⚠ File not found: {args.ts_fm_results} "
              "(run downstream/t3_ts_fm_baselines.py first)")
        ts_fm_cis = {}

    # ── 3. Sauvegarder ──
    payload = {
        "generated_at": datetime.now().isoformat(),
        "n_bootstrap": args.n_boot,
        "bootstrap_seed": args.bootstrap_seed,
        "rule_definition": {
            "name": "CI95_strictly_below_1",
            "description": (
                "A baseline beats persistence iff the 95% bootstrap "
                "confidence interval on ratio_mae = MAE(model)/MAE(persist) "
                "lies strictly below 1.0. Conversely, a CI strictly above "
                "1.0 is unambiguous failure."
            ),
        },
        "pca_ridge_bootstrap": {"by_config": pca_ridge_cis},
        "ts_fm_bootstrap": {"by_config": ts_fm_cis},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n\nSaved: {args.output}")

    # ── 4. W&B ──
    tags = (args.wandb_tags or []) + ["t3-bootstrap"]
    run = init_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"t3_bootstrap_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=list(dict.fromkeys(tags)),
        mode=args.wandb_mode,
        group=args.wandb_group or "t3_bootstrap",
        config={
            "task": "T3_bootstrap_ci",
            "n_boot": args.n_boot,
            "bootstrap_seed": args.bootstrap_seed,
            "rule": "CI95_strictly_below_1",
        },
    )
    wb_metrics: dict = {}
    for key, ci in pca_ridge_cis.items():
        wb_metrics[f"pca_ridge/{key}/ratio_point"] = ci["ratio_mae_point"]
        wb_metrics[f"pca_ridge/{key}/ci95_lo"] = ci["ratio_mae_ci_lo"]
        wb_metrics[f"pca_ridge/{key}/ci95_hi"] = ci["ratio_mae_ci_hi"]
    for key, ci in ts_fm_cis.items():
        wb_metrics[f"ts_fm/{key}/ratio_point"] = ci["ratio_mae_point"]
        wb_metrics[f"ts_fm/{key}/ci95_lo"] = ci["ratio_mae_ci_lo"]
        wb_metrics[f"ts_fm/{key}/ci95_hi"] = ci["ratio_mae_ci_hi"]
    if wb_metrics:
        run.log(wb_metrics)
    log_artifact(run, args.output, name="t3_bootstrap_cis",
                 type="stats", description="T3 bootstrap CI")
    finish(run)


if __name__ == "__main__":
    main()
