"""
t3_ts_fm_baselines.py — Zero-shot time-series foundation model baselines for T3.

Evaluates three TS-FMs on 1-step-ahead EW(Hα) forecasting:

  - **Chronos-Bolt Small** (Amazon, 48M params)
  - **Chronos-Bolt Base** (Amazon, 205M params)
  - **TimesFM-2.0-500M** (Google, 500M params)

All three are applied strictly zero-shot (no fine-tuning, no tuning of
any hyperparameter other than context length). For each test point,
the causal context is the previous ``max_context`` EW(Hα) values of the
same star; the model returns a point forecast for the next step.

Per-prediction arrays (``y_true``, ``y_pred_model``, ``y_pred_persist``)
are saved so that ``stats/bootstrap_ci_t3.py`` can compute CI95 with
1000-resample bootstrap on the residuals.

Usage::

    python downstream/t3_ts_fm_baselines.py                    # all three models
    python downstream/t3_ts_fm_baselines.py --models chronos_bolt_small
    python downstream/t3_ts_fm_baselines.py --max_context 32 --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parent
# Default paths point to the canonical reference layout (see REPRODUCE.md):
#   reference_results/   ← committed JSONs / .pt are regenerated locally
#   results_reproduced/  ← reproduction outputs (override via CLI flags)
RESULTS_DIR = BENCH_ROOT / "reference_results" / "downstream"
EMBEDDINGS_PATH = BENCH_ROOT / "reference_results" / "embeddings" / "star_data.pt"
FEATURES_PATH = BENCH_ROOT / "reference_results" / "features" / "spectral_features.pt"

if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
from stats.wandb_helpers import add_cli_args, finish, init_run, log_artifact  # noqa: E402

# Reuse the aligned-data loader from PCA+Ridge baseline
from downstream.t3_pca_ridge_temporal_baseline import (  # noqa: E402
    CUTOFF_MJD,
    load_aligned_data,
)


MODEL_REGISTRY = {
    "chronos_bolt_small": {
        "family": "chronos",
        "hf_id": "amazon/chronos-bolt-small",
        "params_M": 48,
    },
    "chronos_bolt_base": {
        "family": "chronos",
        "hf_id": "amazon/chronos-bolt-base",
        "params_M": 205,
    },
    "timesfm_2_0_500m": {
        "family": "timesfm",
        "hf_id": "google/timesfm-2.0-500m-pytorch",
        "params_M": 500,
    },
}


def build_test_contexts(aligned_data, max_context):
    """Build causal contexts for every test point.

    For each star we split the chronological EW series by ``CUTOFF_MJD`` and,
    for each test index ``t``, we take the last ``max_context`` EW values
    strictly before position ``t`` as the causal context.

    Returns four parallel lists aligned by test-prediction index:
      - contexts: list of 1-D np.ndarrays (variable length ≤ max_context)
      - y_true:   list of floats (target EW at test position)
      - y_pred_persist: list of floats (EW at position t-1 = naive predictor)
      - meta:     list of dicts {"star": str, "mjd": float, "dt_days": float}
    """
    contexts, y_true, y_pred_persist, meta = [], [], [], []

    for star, data in aligned_data.items():
        mjds = data["mjds"]
        ews = data["ews"]
        n = len(mjds)
        if n < 2:
            continue

        test_positions = np.where(mjds > CUTOFF_MJD)[0]
        for t in test_positions:
            if t == 0:
                # No causal context whatsoever — skip (same convention as PCA+Ridge)
                continue
            lo = max(0, t - max_context)
            ctx = ews[lo:t].astype(np.float32)
            if ctx.size < 1 or not np.all(np.isfinite(ctx)):
                continue
            y = float(ews[t])
            if not np.isfinite(y):
                continue
            contexts.append(ctx)
            y_true.append(y)
            y_pred_persist.append(float(ews[t - 1]))
            meta.append({
                "star": star,
                "mjd": float(mjds[t]),
                "dt_days": float(mjds[t] - mjds[t - 1]),
            })

    return contexts, y_true, y_pred_persist, meta


# ──────────────────────────────────────────────────────────────────────
#   Chronos-Bolt wrapper
# ──────────────────────────────────────────────────────────────────────
def run_chronos(hf_id: str, contexts, batch_size: int, device: str) -> np.ndarray:
    """Zero-shot 1-step forecast with Chronos-Bolt; returns median."""
    from chronos import BaseChronosPipeline

    print(f"    Loading {hf_id} on {device} ...")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipeline = BaseChronosPipeline.from_pretrained(
        hf_id,
        device_map=device,
        torch_dtype=dtype,
    )

    preds = np.empty(len(contexts), dtype=np.float32)
    n_batches = (len(contexts) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        s, e = bi * batch_size, min((bi + 1) * batch_size, len(contexts))
        batch = [torch.as_tensor(c, dtype=torch.float32) for c in contexts[s:e]]
        # predict_quantiles(inputs=..., prediction_length=..., quantile_levels=...)
        # returns (quantiles: [B, H, Q], mean: [B, H])
        q_tensor, _mean = pipeline.predict_quantiles(
            inputs=batch,
            prediction_length=1,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        median = q_tensor[:, 0, 1].float().cpu().numpy()  # H=0, Q=median
        preds[s:e] = median
        if (bi + 1) % 20 == 0 or bi == n_batches - 1:
            print(f"      batch {bi + 1}/{n_batches}")

    del pipeline
    if device == "cuda":
        torch.cuda.empty_cache()
    return preds


# ──────────────────────────────────────────────────────────────────────
#   TimesFM wrapper
# ──────────────────────────────────────────────────────────────────────
def run_timesfm(hf_id: str, contexts, batch_size: int, device: str) -> np.ndarray:
    """Zero-shot 1-step forecast with TimesFM-2.0; returns point forecast."""
    import timesfm

    print(f"    Loading {hf_id} on {device} ...")
    backend = "gpu" if device == "cuda" else "cpu"
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=batch_size,
            horizon_len=1,
            num_layers=50,
            use_positional_embedding=False,
            context_len=2048,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=hf_id),
    )

    # TimesFM accepts a list of 1-D arrays; freq=0 -> high-frequency / irregular
    preds = np.empty(len(contexts), dtype=np.float32)
    n_batches = (len(contexts) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        s, e = bi * batch_size, min((bi + 1) * batch_size, len(contexts))
        inputs = [np.asarray(c, dtype=np.float32) for c in contexts[s:e]]
        freq = [0] * len(inputs)
        point_forecast, _ = tfm.forecast(inputs=inputs, freq=freq)
        # point_forecast: [B, horizon] -> take horizon=0
        preds[s:e] = np.asarray(point_forecast)[:, 0].astype(np.float32)
        if (bi + 1) % 20 == 0 or bi == n_batches - 1:
            print(f"      batch {bi + 1}/{n_batches}")

    del tfm
    if device == "cuda":
        torch.cuda.empty_cache()
    return preds


# ──────────────────────────────────────────────────────────────────────
#   Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    global EMBEDDINGS_PATH, FEATURES_PATH, RESULTS_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Subset of TS-FMs to evaluate",
    )
    parser.add_argument("--max_context", type=int, default=64,
                        help="Max number of past EW values fed as context")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embeddings", type=str, default=str(EMBEDDINGS_PATH),
                        help="Path to star_data.pt embeddings (default: bench-relative)")
    parser.add_argument("--features", type=str, default=str(FEATURES_PATH),
                        help="Path to spectral_features.pt (default: results/)")
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR),
                        help="Directory for output JSON (default: reference_results/downstream)")
    add_cli_args(parser)
    args = parser.parse_args()

    EMBEDDINGS_PATH = Path(args.embeddings)
    FEATURES_PATH = Path(args.features)
    RESULTS_DIR = Path(args.output_dir)

    print("=" * 70)
    print("T3 BASELINES — Time-series foundation models (zero-shot)")
    print(f"models      : {args.models}")
    print(f"max_context : {args.max_context}")
    print(f"device      : {args.device}")
    print("=" * 70)

    print("\n1. Loading aligned spectra / EW series ...")
    aligned = load_aligned_data(EMBEDDINGS_PATH, FEATURES_PATH)

    print("\n2. Building causal test contexts ...")
    contexts, y_true, y_pred_persist, meta = build_test_contexts(
        aligned, args.max_context)
    n = len(contexts)
    y_true_np = np.asarray(y_true, dtype=np.float32)
    y_persist_np = np.asarray(y_pred_persist, dtype=np.float32)
    print(f"   n_test_predictions = {n}")
    print(f"   persistence MAE (point) = "
          f"{float(np.mean(np.abs(y_persist_np - y_true_np))):.4f}")

    if n == 0:
        raise RuntimeError("No valid test contexts — aborting.")

    # ── W&B parent run ──
    tags = (args.wandb_tags or []) + ["t3-ts-fm-baseline"]
    run = init_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"t3_ts_fm_{datetime.now():%Y%m%d_%H%M%S}",
        tags=list(dict.fromkeys(tags)),
        mode=args.wandb_mode,
        group=args.wandb_group or "t3_ts_fm",
        config={
            "task": "T3_ts_fm_baselines",
            "models": args.models,
            "max_context": args.max_context,
            "batch_size": args.batch_size,
            "cutoff_mjd": CUTOFF_MJD,
            "n_test_predictions": n,
        },
    )

    # ── 3. Run each model ──
    all_results: dict = {}
    for model_key in args.models:
        spec = MODEL_REGISTRY[model_key]
        print(f"\n3. Running {model_key} ({spec['hf_id']}) ...")
        if spec["family"] == "chronos":
            y_pred = run_chronos(spec["hf_id"], contexts,
                                 args.batch_size, args.device)
        elif spec["family"] == "timesfm":
            y_pred = run_timesfm(spec["hf_id"], contexts,
                                 args.batch_size, args.device)
        else:
            raise ValueError(f"Unknown family: {spec['family']}")

        mae_model = float(np.mean(np.abs(y_pred - y_true_np)))
        mae_persist = float(np.mean(np.abs(y_persist_np - y_true_np)))
        ratio = mae_model / max(mae_persist, 1e-10)
        print(f"   MAE model   : {mae_model:.4f}")
        print(f"   MAE persist : {mae_persist:.4f}")
        print(f"   ratio_mae   : {ratio:.4f}  "
              f"({'beats' if ratio < 1.0 else 'fails'} persistence at point)")

        # Stratify by forecast-horizon bucket (matches PCA+Ridge schema and is
        # consumed by stats/figure_t3_horizon.py without any post-processing).
        dt_arr = np.asarray([m["dt_days"] for m in meta], dtype=np.float32)
        BUCKETS = [("< 7d", 0, 7), ("7-30d", 7, 30), ("30-90d", 30, 90),
                   ("90-365d", 90, 365), ("> 365d", 365, float("inf"))]
        by_horizon = {}
        for lbl, lo, hi in BUCKETS:
            msk = (dt_arr >= lo) & (dt_arr < hi)
            nb = int(msk.sum())
            if nb == 0:
                by_horizon[lbl] = {"n": 0, "ratio": None}
                continue
            mae_m_b = float(np.mean(np.abs(y_true_np[msk] - y_pred[msk])))
            mae_p_b = float(np.mean(np.abs(y_true_np[msk] - y_persist_np[msk])))
            by_horizon[lbl] = {"n": nb,
                               "mae_model":   round(mae_m_b, 6),
                               "mae_persist": round(mae_p_b, 6),
                               "ratio":       round(mae_m_b / max(mae_p_b, 1e-10), 6)}

        all_results[model_key] = {
            "config": {
                "model": model_key,
                "hf_id": spec["hf_id"],
                "params_M": spec["params_M"],
                "max_context": args.max_context,
                "horizon": 1,
                "cutoff_mjd": CUTOFF_MJD,
                "n_test": n,
            },
            "global": {
                "mae_model": round(mae_model, 6),
                "mae_persist": round(mae_persist, 6),
                "ratio_mae": round(ratio, 6),
            },
            "by_horizon": by_horizon,
            "predictions": {
                "y_true": y_true_np.tolist(),
                "y_pred_model": y_pred.astype(np.float32).tolist(),
                "y_pred_persist": y_persist_np.tolist(),
                "dt_days": [m["dt_days"] for m in meta],
                "star_names": [m["star"] for m in meta],
                "mjd_target": [m["mjd"] for m in meta],
            },
        }

        run.log({
            f"{model_key}/mae_model": mae_model,
            f"{model_key}/mae_persist": mae_persist,
            f"{model_key}/ratio_mae": ratio,
        })

    # ── 4. Save ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "t3_ts_fm_baselines.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n4. Results saved → {output_path}")

    log_artifact(run, output_path, name="t3_ts_fm_baselines",
                 type="baseline_results",
                 description="Zero-shot Chronos-Bolt + TimesFM on EW(Hα) 1-step forecast")
    finish(run)


if __name__ == "__main__":
    main()
