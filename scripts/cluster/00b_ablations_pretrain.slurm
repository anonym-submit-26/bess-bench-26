#!/bin/bash
#SBATCH --job-name=bess-00b-ablations
#SBATCH --partition=shortrun,longrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/00b_ablation_%A_%a.out
#SBATCH --error=logs/00b_ablation_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0b — BeMAE-Hα ablation pretraining (4 axes × 2 values × 3 seeds = 24)
# ══════════════════════════════════════════════════════════════════════════════
#
# Job index = variant_idx * 3 + seed_idx, with
#   variants[0..7] = mask_ratio_030, mask_ratio_075,
#                    n_layers_2,    n_layers_6,
#                    d_model_64,    d_model_256,
#                    patch_size_4,  patch_size_16
#   seeds[0..2]    = 42, 123, 456
#
# Produces, for each (variant, seed) :
#   - stage1_encoder/runs/ablation_<tag>_seed<S>/best.pt
#   - data/embeddings_ablation_<tag>_seed<S>/star_data.pt
#
# Configs are read from configs/ablations/<tag>.yaml ; overrides are applied
# verbatim to train.py via CLI flags (mask_ratio / n_layers / d_model /
# patch_size / batch_size).
#
# W&B : project anonym-bess-26, group bemae_ablations_v1
# Duration : ~30 min/job on A100 80GB (24 jobs × 30 min ≈ 12 GPU-hours total)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

VARIANTS=(
    mask_ratio_030
    mask_ratio_075
    n_layers_2
    n_layers_6
    d_model_64
    d_model_256
    patch_size_4
    patch_size_16
)
SEEDS=(42 123 456)

VARIANT_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
TAG="${VARIANTS[$VARIANT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "=============================================================="
echo "PHASE 0b — Ablation : tag=${TAG}  seed=${SEED}"
echo "Date : $(date)  |  Node : $(hostname)"
echo "Array : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================================="

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$REPO_ROOT/configs/ablations/${TAG}.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: ablation config not found : $CONFIG_FILE" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate envglobal

# ── Parse YAML overrides into CLI flags (no PyYAML needed; grep/awk) ────────
# Each line under "overrides:" is "  KEY: VALUE" → "--KEY VALUE".
OVERRIDES=()
while IFS= read -r line; do
    # match leading 2-space indent + key: value (skip empty / comments / non-indented)
    if [[ "$line" =~ ^[[:space:]]{2}([a-z_]+):[[:space:]]*([0-9.eE+-]+) ]]; then
        OVERRIDES+=("--${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")
    fi
done < <(awk '/^overrides:/{flag=1; next} /^[a-z]/{flag=0} flag' "$CONFIG_FILE")

echo ">>> Overrides : ${OVERRIDES[*]}"

cd "$REPO_ROOT/stage1_encoder"
mkdir -p "$REPO_ROOT/scripts/logs" runs

RUN_NAME="ablation_${TAG}_seed${SEED}"
RUN_DIR="runs/${RUN_NAME}"
EMB_DIR="$REPO_ROOT/data/embeddings_ablation_${TAG}_seed${SEED}"

# ── 1. Train ───────────────────────────────────────────────────────────────
if [[ -f "${RUN_DIR}/best.pt" ]]; then
    echo ">>> SKIP train (${RUN_DIR}/best.pt exists)"
else
    echo ">>> TRAIN  $(date)"
    python train.py \
        --run_name "$RUN_NAME" \
        --seed "$SEED" \
        --output_dir runs \
        "${OVERRIDES[@]}" \
        --wandb_project anonym-bess-26 \
        --wandb_entity anon-entity \
        --wandb_group bemae_ablations_v1 \
        --wandb_tags "ablation" "${TAG}" "seed${SEED}" "v1"
fi

# ── 2. Embeddings ──────────────────────────────────────────────────────────
if [[ -f "${EMB_DIR}/star_data.pt" ]]; then
    echo ">>> SKIP embeddings (${EMB_DIR}/star_data.pt exists)"
else
    echo ">>> EMBEDDINGS  $(date)"
    python precompute_embeddings.py \
        --checkpoint "${RUN_DIR}/best.pt" \
        --output_dir "${EMB_DIR}"
fi

echo ""
echo "=============================================================="
echo "PHASE 0b done : ${TAG} seed=${SEED}  $(date)"
echo "=============================================================="
