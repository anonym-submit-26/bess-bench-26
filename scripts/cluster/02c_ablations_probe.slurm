#!/bin/bash
#SBATCH --job-name=bess-02c-ablations-probe
#SBATCH --partition=shortrun,longrun
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/02c_ablations_probe_%A_%a.out
#SBATCH --error=logs/02c_ablations_probe_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2c — Ablations T1 probing (CPU)
# ══════════════════════════════════════════════════════════════════════════════
#
# For each ablation variant [0..7], runs probe_features.py on the 3 encoder
# seeds {42, 123, 456} restricted to the 3 headline features
# {fwhm, central_depth, delta_v}.
#
# Output : downstream/results/probe_features_ablation_<tag>_seed<S>.json
#
# Requires Phase 0b (ablation embeddings) to be complete.
# Duration : ~10 min/job on 8 CPU cores (3 features × 3 seeds × 6 reprs).
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

TAG="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
SEEDS=(42 123 456)
CV_SEED=0

echo "=============================================================="
echo "PHASE 2c — Ablations T1 probe : tag=${TAG}"
echo "Date : $(date)  |  Node : $(hostname)"
echo "=============================================================="

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/downstream"

eval "$(conda shell.bash hook)"
conda activate envglobal

mkdir -p "$REPO_ROOT/scripts/logs" results figures

for ENC in "${SEEDS[@]}"; do
    EMB_FILE="$REPO_ROOT/data/embeddings_ablation_${TAG}_seed${ENC}/star_data.pt"
    OUT_SUFFIX="_ablation_${TAG}_seed${ENC}"
    OUT_FILE="results/probe_features_results${OUT_SUFFIX}.json"

    if [[ -f "$OUT_FILE" ]]; then
        echo ">>> SKIP ${TAG} seed=${ENC} (${OUT_FILE} exists)"
        continue
    fi
    if [[ ! -f "$EMB_FILE" ]]; then
        echo "WARNING: missing embeddings ${EMB_FILE}, skipping."
        continue
    fi

    echo ">>> PROBE ${TAG} seed=${ENC}  $(date)"
    python probe_features.py \
        --embeddings "$EMB_FILE" \
        --feature_list fwhm central_depth delta_v \
        --seed "$CV_SEED" \
        --output_suffix "$OUT_SUFFIX" \
        --wandb_project anonym-bess-26 \
        --wandb_entity anon-entity \
        --wandb_group t1_ablations_v2 \
        --wandb_name "t1_ablation_${TAG}_enc${ENC}_$(date +%Y%m%d_%H%M%S)" \
        --wandb_tags "ablation" "${TAG}" "encoder_seed${ENC}" "t1" "v2" "morphology_v2"
done

echo ""
echo "=============================================================="
echo "PHASE 2c done : ${TAG}  $(date)"
echo "=============================================================="
