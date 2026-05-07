#!/bin/bash
# =============================================================================
# export_to_hf.sh — Export BeSS → HuggingFace Hub (via SLURM)
# =============================================================================
# Commenter/uncomment the steps selon besoin, then : sbatch export_to_hf.sh
# Suivre : tail -f ../../logs/export_hf_<jobid>.out
# =============================================================================

#SBATCH --job-name=bess-hf-export
#SBATCH --output=../../logs/export_hf_%j.out
#SBATCH --error=../../logs/export_hf_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --nodelist=sn3



# ── Configuration ────────────────────────────────────────────────────────────
HF_REPO_ID="anonym-submit-26/bess-bench-26"
N_WORKERS=${SLURM_CPUS_PER_TASK:-8}
# EXTRACT_LIMIT=100   # Uncomment for test rapide

# ── Environnement ────────────────────────────────────────────────────────────
set -euo pipefail
PROJECT_DIR="."
EXPORT_DIR="${PROJECT_DIR}/data_pipeline/03_export"
cd "${EXPORT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# Load .env
[[ -f "${PROJECT_DIR}/.env" ]] && { set -a; source "${PROJECT_DIR}/.env"; set +a; }

# Conda
source /share/common/anaconda/etc/profile.d/conda.sh 2>/dev/null \
    || source /share/apps/anaconda/anaconda3-2024/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERREUR: impossible de trouver conda.sh"; exit 1; }
conda activate envglobal

echo "=== EXPORT BESS | $(date -Iseconds) | Job ${SLURM_JOB_ID:-local} ==="

# ── Step 1 : Extraction FITS → Parquet (~30-60 min) ────────────────────────
#python 01_extract_spectra.py --workers ${N_WORKERS} ${EXTRACT_LIMIT:+--limit $EXTRACT_LIMIT}

# ── Step 2 : Assemblage DatasetDict HuggingFace (~15-30 min) ───────────────
#python 02_build_dataset.py

# ── Step 3 : Push to HuggingFace Hub (~30-60 min) ────────────────────────
python 03_push_to_hub.py --repo-id "${HF_REPO_ID}" --private

echo "=== DONE | $(date -Iseconds) ==="
