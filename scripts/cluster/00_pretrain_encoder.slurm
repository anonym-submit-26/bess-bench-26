#!/bin/bash
#SBATCH --job-name=bess-00-encoder
#SBATCH --partition=shortrun,longrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/00_encoder_%j.out
#SBATCH --error=logs/00_encoder_%j.err

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0 — BeMAE Hα encoder pretraining (multi-seed)
# ══════════════════════════════════════════════════════════════════════════════
#
# Produces, for each seed ∈ {42, 123, 456} :
#   - stage1_encoder/runs/Halpha_all_seed<S>/best.pt          (MAE checkpoint)
#   - stage1_encoder/runs/Halpha_all_seed<S>/eval_results.json (diagnostics)
#   - data/embeddings_halpha_seed<S>/star_data.pt              (frozen embeddings)
#
# And once :
#   - data/embeddings_halpha/star_data.pt  → symlink to seed 42 (canonical)
#   - stage1_encoder/hf_export/bemae-halpha-v1/  (HuggingFace-ready model)
#   - stage1_encoder/runs/multiseed_summary.json (aggregate metrics)
#
# W&B : project anonym-bess-26, group bemae_pretrain_v1
# Duration : ~1h30 on an A100 80GB
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

echo "=============================================================="
echo "PHASE 0 — BeMAE-Hα multi-seed pretraining"
echo "Date : $(date)  |  Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================================="

# Repo root = parent of scripts/. Use SLURM_SUBMIT_DIR when running under sbatch.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/stage1_encoder"

eval "$(conda shell.bash hook)"
conda activate envglobal

mkdir -p "$REPO_ROOT/scripts/logs" runs hf_export "$REPO_ROOT/data"

SEEDS=(42 123 456)
OFFICIAL_SEED=42

# ── 1. Train + eval + embeddings for each seed ─────────────────────────────
for SEED in "${SEEDS[@]}"; do
    RUN_NAME="Halpha_all_seed${SEED}"
    RUN_DIR="runs/${RUN_NAME}"
    EMB_DIR="$REPO_ROOT/data/embeddings_halpha_seed${SEED}"

    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo " seed=${SEED}  |  run=${RUN_NAME}"
    echo "──────────────────────────────────────────────────────────────"

    # 1a. Training
    if [[ -f "${RUN_DIR}/best.pt" ]]; then
        echo ">>> SKIP train (${RUN_DIR}/best.pt exists)"
    else
        echo ">>> TRAIN seed=${SEED}  $(date)"
        python train.py \
            --run_name "$RUN_NAME" \
            --seed "$SEED" \
            --output_dir runs \
            --wandb_project anonym-bess-26 \
            --wandb_entity anon-entity \
            --wandb_group bemae_pretrain_v1 \
            --wandb_tags "seed${SEED}" "encoder" "halpha" "v1" "multiseed"
    fi

    # 1b. Evaluation (diagnostics)
    if [[ -f "${RUN_DIR}/eval_results.json" ]]; then
        echo ">>> SKIP eval (${RUN_DIR}/eval_results.json exists)"
    else
        echo ">>> EVAL seed=${SEED}  $(date)"
        python evaluate.py --checkpoint "${RUN_DIR}/best.pt"
    fi

    # 1c. Embeddings
    if [[ -f "${EMB_DIR}/star_data.pt" ]]; then
        echo ">>> SKIP embeddings (${EMB_DIR}/star_data.pt exists)"
    else
        echo ">>> EMBEDDINGS seed=${SEED}  $(date)"
        python precompute_embeddings.py \
            --checkpoint "${RUN_DIR}/best.pt" \
            --output_dir "${EMB_DIR}"
    fi
done

# ── 2. Canonical symlink : embeddings_halpha -> embeddings_halpha_seed42 ─────
CANONICAL="$REPO_ROOT/data/embeddings_halpha"
if [[ ! -e "$CANONICAL" ]]; then
    ln -s "embeddings_halpha_seed${OFFICIAL_SEED}" "$CANONICAL"
    echo ">>> Created symlink : data/embeddings_halpha → embeddings_halpha_seed${OFFICIAL_SEED}"
fi

# ── 3. Aggregate multi-seed summary ─────────────────────────────────────────
echo ""
echo ">>> Aggregating multi-seed metrics"
python - <<'PYEOF'
import json, statistics
from pathlib import Path

seeds = [42, 123, 456]
runs = Path("runs")

def pick(d, *keys):
    """Follow dotted path inside nested dicts; return None if any key missing or value is not numeric."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None

rows = []
for s in seeds:
    evalp = runs / f"Halpha_all_seed{s}" / "eval_results.json"
    if not evalp.exists():
        print(f"  ! missing {evalp}")
        continue
    with open(evalp) as f:
        d = json.load(f)
    rows.append({
        "seed": s,
        "r2_cv_ew_test":           pick(d, "r2_cv_ew_test", "r2_mean"),
        "r2_cv_ew_all":            pick(d, "r2_cv_ew_all", "r2_mean"),
        "knn_intra_star":          pick(d, "knn_intra_star", "mean_accuracy"),
        "cos_centroids":           pick(d, "cos_centroids", "cos_mean"),
        "recall_at_10_test":       pick(d, "recall_at_10_test"),
        "recall_at_10":            pick(d, "recall_at_10"),
    })

def agg(key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if len(vals) == 0:
        return {"mean": None, "std": None, "n": 0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1}
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals), "n": len(vals)}

metrics = ["r2_cv_ew_test", "r2_cv_ew_all", "knn_intra_star",
           "cos_centroids", "recall_at_10_test", "recall_at_10"]
summary = {
    "seeds": seeds,
    "per_seed": rows,
    "aggregate": {k: agg(k) for k in metrics},
}
out = runs / "multiseed_summary.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  -> {out}")
for k, v in summary["aggregate"].items():
    if v.get("mean") is not None:
        print(f"     {k:<24s} {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")
PYEOF

# ── 4. Export encoder seed=42 to HuggingFace format ──────────────────────────
if [[ ! -f "hf_export/bemae-halpha-v1/pytorch_model.bin" ]]; then
    echo ""
    echo ">>> HF export (seed=${OFFICIAL_SEED})  $(date)"
    python export_hf.py \
        --checkpoint "runs/Halpha_all_seed${OFFICIAL_SEED}/best.pt" \
        --output_dir hf_export/bemae-halpha-v1
else
    echo ">>> SKIP HF export (already present)"
fi

echo ""
echo "=============================================================="
echo "PHASE 0 done  —  $(date)"
echo "=============================================================="
