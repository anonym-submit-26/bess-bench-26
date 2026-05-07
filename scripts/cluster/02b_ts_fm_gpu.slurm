#!/bin/bash
#SBATCH --job-name=bess-02b-tsfm
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/02b_tsfm_%j.out
#SBATCH --error=logs/02b_tsfm_%j.err

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2b — T3 zero-shot time-series foundation models (GPU)
# ══════════════════════════════════════════════════════════════════════════════
#
# Evaluates Chronos-Bolt (small + base) and TimesFM-2.0-500M zero-shot on the
# EW(t+1) forecasting task. GPU strongly recommended (TimesFM is ~500M params).
#
# Requires Phase 1 (spectral_features.pt). Independent of Phase 0 / 2a.
# group t3_ts_fm_v1.
# Duration : ~15–30 min on an A100.
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

echo "=============================================================="
echo "PHASE 2b — T3 zero-shot TS-FM baselines (Chronos + TimesFM)"
echo "Date : $(date)  |  Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================================="

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/downstream"

eval "$(conda shell.bash hook)"
conda activate envglobal

mkdir -p "$REPO_ROOT/scripts/logs" results

if [[ -f "results/t3_ts_fm_baselines.json" ]]; then
    echo ">>> SKIP t3_ts_fm_baselines (already present)"
else
    python t3_ts_fm_baselines.py \
        --device cuda \
        --wandb_project anonym-bess-26 \
        --wandb_entity anon-entity \
        --wandb_group t3_ts_fm_v2 \
        --wandb_tags "t3" "zero_shot" "ts_fm" "v2" "morphology_v2"
fi

echo ""
echo "=============================================================="
echo "PHASE 2b done  —  $(date)"
echo "=============================================================="
