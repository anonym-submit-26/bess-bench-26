#!/bin/bash
#SBATCH --job-name=bess-01-features
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/01_features_%j.out
#SBATCH --error=logs/01_features_%j.err

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Spectral feature extraction (CPU)
# ══════════════════════════════════════════════════════════════════════════════
#
# Extracts the 6 Hα spectral features (EW, FWHM, V/R, Δv, depth, peak_I) and
# profile-type labels from the full HuggingFace dataset, plus Hβ features
# needed by T2 (LineTransfer).
#
# Outputs :
#   - downstream/results/spectral_features.pt         (Hα, all stars)
#   - downstream/results/spectral_features_hbeta.pt   (Hβ, stars with ±50 Å coverage)
#
# Independent of Phase 0 — can run in parallel with it.
# Duration : ~1–2 h (sequential ; reads the full HF dataset twice).
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

echo "=============================================================="
echo "PHASE 1 — Spectral feature extraction"
echo "Date : $(date)  |  Node : $(hostname)"
echo "=============================================================="

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/downstream"

eval "$(conda shell.bash hook)"
conda activate envglobal

mkdir -p "$REPO_ROOT/scripts/logs" results

# ── Hα features ─────────────────────────────────────────────────────────────
if [[ -f "results/spectral_features.pt" ]]; then
    echo ">>> SKIP Hα features (already present)"
else
    echo ">>> Extracting Hα features  $(date)"
    python compute_spectral_features.py --output results/spectral_features.pt
fi

# ── Hβ features (for T2 LineTransfer) ──────────────────────────────────────────────
if [[ -f "results/spectral_features_hbeta.pt" ]]; then
    echo ">>> SKIP Hβ features (already present)"
else
    echo ">>> Extracting Hβ features  $(date)"
    python compute_hbeta_features.py --output results/spectral_features_hbeta.pt
fi

echo ""
echo "=============================================================="
echo "PHASE 1 done  —  $(date)"
echo "=============================================================="
