#!/bin/bash
#SBATCH --job-name=bess-02a-downstream
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/02a_downstream_%j.out
#SBATCH --error=logs/02a_downstream_%j.err

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2a — CPU downstream tasks
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs every downstream evaluation that does NOT need a GPU :
#
#   T1   probe_features.py           × 3 encoder seeds   (Ridge CV probes)
#   T2a  classify_profiles.py                            (5-class profile, exploratory)
#   T2   t2_cross_line_probing.py                        (Hβ -> Hα cross-line)
#   T3a  t3_pca_ridge_temporal_baseline.py               (PCA+Ridge forecaster)
#   T4   clustering_analysis.py                          (KMeans + HDBSCAN + UMAP, exploratory)
#
# Requires Phase 0 (embeddings) and Phase 1 (features).
# W&B: project anonym-bess-26, groups t1_multiseed_v2 / t3_pca_ridge_v2
#                  (suffix v2 = morphology hardening 2026/05, encoders unchanged).
# Duration : ~30 min – 1 h on 8 CPU cores.
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

echo "=============================================================="
echo "PHASE 2a — CPU downstream tasks"
echo "Date : $(date)  |  Node : $(hostname)"
echo "=============================================================="

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT/downstream"

eval "$(conda shell.bash hook)"
conda activate envglobal

mkdir -p "$REPO_ROOT/scripts/logs" results figures

# ── T1 : linear probes × 3 encoder seeds ────────────────────────────────────
echo ""
echo "── T1 : linear probes (Ridge CV) × 3 encoder seeds ──────────"
ENCODER_SEEDS=(42 123 456)
CV_SEED=0
for ENC in "${ENCODER_SEEDS[@]}"; do
    OUT="results/probe_features_results_seed${ENC}.json"
    if [[ -f "$OUT" ]]; then
        echo ">>> SKIP T1 encoder seed=${ENC} (${OUT} exists)"
        continue
    fi
    echo ">>> T1 encoder seed=${ENC}  $(date)"
    python probe_features.py \
        --embeddings "$REPO_ROOT/data/embeddings_halpha_seed${ENC}/star_data.pt" \
        --seed "$CV_SEED" \
        --output_suffix "_seed${ENC}" \
        --wandb_project anonym-bess-26 \
        --wandb_entity anon-entity \
        --wandb_group t1_multiseed_v2 \
        --wandb_name "t1_probe_enc${ENC}_$(date +%Y%m%d_%H%M%S)" \
        --wandb_tags "encoder_seed${ENC}" "cv_seed${CV_SEED}" "t1" "v2" "morphology_v2"
done
# Canonical alias : probe_features_results.json -> encoder seed 42
ln -sf "probe_features_results_seed42.json" "results/probe_features_results.json"

# ── T2a : 5-class profile classification (exploratory) ──────────────────────
echo ""
echo "── T2a : profile classification (exploratory) ───────────────"
if [[ -f "results/classify_profiles.json" ]]; then
    echo ">>> SKIP classify_profiles"
else
    python classify_profiles.py \
        --embeddings "$REPO_ROOT/data/embeddings_halpha/star_data.pt"
fi

# ── T2 : cross-line Hβ -> Hα probing ────────────────────────────────────────
echo ""
echo "── T2 : cross-line Hβ → Hα probing ──────────────────────────"
if [[ -f "results/probe_t2_hbeta_to_halpha.json" ]]; then
    echo ">>> SKIP t2_cross_line_probing"
else
    python t2_cross_line_probing.py \
        --hbeta_features results/spectral_features_hbeta.pt \
        --halpha_features results/spectral_features.pt \
        --output results/probe_t2_hbeta_to_halpha.json \
        --output_fig figures/t2_hbeta_to_halpha.png
fi


# ── T3a : PCA + Ridge temporal baseline ─────────────────────────────────────
echo ""
echo "── T3a : PCA + Ridge temporal baseline ──────────────────────"
if [[ -f "results/t3_pca_ridge_temporal_baseline.json" ]]; then
    echo ">>> SKIP t3_pca_ridge_temporal_baseline"
else
    python t3_pca_ridge_temporal_baseline.py \
        --wandb_project anonym-bess-26 \
        --wandb_entity anon-entity \
        --wandb_group t3_pca_ridge_v2 \
        --wandb_tags "t3" "pca_ridge" "v2" "morphology_v2"
fi

# ── T4 : clustering analysis (exploratory) ──────────────────────────────────
echo ""
echo "── T4 : clustering & UMAP (exploratory) ─────────────────────"
if [[ -f "results/clustering_analysis.json" ]]; then
    echo ">>> SKIP clustering_analysis"
else
    python clustering_analysis.py || echo "   (clustering_analysis returned non-zero ; non-blocking for paper tables)"
fi

echo ""
echo "=============================================================="
echo "PHASE 2a done  —  $(date)"
echo "=============================================================="
