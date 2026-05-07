#!/bin/bash
#SBATCH --job-name=bess-00a-train
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/00a_train_%j.out
#SBATCH --error=logs/00a_train_%j.err

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0a — Train a SINGLE BeMAE Hα seed (parallelisable)
# ══════════════════════════════════════════════════════════════════════════════
#
# Required env var :  SEED  (e.g. 42, 123, 456)
#
# Usage :
#   sbatch --export=ALL,SEED=42 00a_train_seed.slurm
#
# Produces : stage1_encoder/runs/Halpha_all_seed${SEED}/best.pt
#            stage1_encoder/runs/Halpha_all_seed${SEED}/config.json
#
# Duration : ~1h30 on A6000 / A100
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

if [[ -z "${SEED:-}" ]]; then
    echo "ERROR: SEED env var must be set (e.g. --export=ALL,SEED=42)"
    exit 1
fi

echo "=============================================================="
echo "PHASE 0a — BeMAE-Hα train (seed=${SEED})"
echo "Date : $(date)  |  Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================================="

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/stage1_encoder"

eval "$(conda shell.bash hook)"
conda activate envglobal

mkdir -p "$REPO_ROOT/scripts/logs" runs

RUN_NAME="Halpha_all_seed${SEED}"
RUN_DIR="runs/${RUN_NAME}"

if [[ -f "${RUN_DIR}/best.pt" ]]; then
    echo ">>> SKIP train (${RUN_DIR}/best.pt already exists)"
    exit 0
fi

echo ">>> TRAIN seed=${SEED}  $(date)"
python train.py \
    --run_name "$RUN_NAME" \
    --seed "$SEED" \
    --output_dir runs \
    --wandb_project anonym-bess-26 \
    --wandb_entity anon-entity \
    --wandb_group bemae_pretrain_v1 \
    --wandb_tags "seed${SEED}" "encoder" "halpha" "v1" "multiseed" "helio_corrected"

echo ">>> Done seed=${SEED}  $(date)"
