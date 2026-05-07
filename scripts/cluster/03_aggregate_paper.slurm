#!/bin/bash
#SBATCH --job-name=bess-03-aggregate
#SBATCH --partition=shortrun
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/03_aggregate_%j.out
#SBATCH --error=logs/03_aggregate_%j.err

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Statistical aggregation & paper artefacts
# ══════════════════════════════════════════════════════════════════════════════
#
# Takes every JSON produced by Phases 0–2 and produces the paper tables,
# bootstrap confidence intervals, and summary figures.
#
# Outputs :
#   - stats/results/t1_multiseed_summary.json   (T1 across 3 encoder seeds)
#   - stats/results/t3_bootstrap_cis.json       (T3 bootstrap CI95)
#   - stats/results/coverage_audit.json         (spectral line coverage %)
#   - paper/tables/table_stage1.tex
#   - paper/tables/table_t1_probes.tex
#   - paper/tables/table_t3_baselines.tex
#   - paper/tables/ci_results.json + table_ci_t1.tex
#   - paper/tables/summary.json                 (machine-readable)
#
# Requires Phases 0 + 1 + 2a + 2b.
# Duration : < 5 min total.
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

echo "=============================================================="
echo "PHASE 3 — Aggregation & paper artefacts"
echo "Date : $(date)"
echo "=============================================================="

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

eval "$(conda shell.bash hook)"
conda activate envglobal

mkdir -p scripts/logs stats/results paper/tables figures

# ── T1 multi-seed aggregation (mean ± std over 3 encoder seeds) ─────────────
echo ""
echo ">>> T1 multi-seed summary"
python stats/aggregate_t1_multiseed.py \
    --seeds 42 123 456 \
    --wandb_project anonym-bess-26 \
    --wandb_entity anon-entity \
    --wandb_group t1_multiseed_agg_v2

# ── T1 Gaussian CI95 (from per-fold probe results) ──────────────────────────
echo ""
echo ">>> T1 Gaussian CI95"
python stats/bootstrap_ci.py || echo "  ! bootstrap_ci.py failed (non-blocking)"

# ── T3 bootstrap CI95 ───────────────────────────────────────────────────────
echo ""
echo ">>> T3 bootstrap CI95"
python stats/bootstrap_ci_t3.py \
    --pca_ridge_results downstream/results/t3_pca_ridge_temporal_baseline.json \
    --ts_fm_results     downstream/results/t3_ts_fm_baselines.json \
    --output            stats/results/t3_bootstrap_cis.json \
    --wandb_project     anonym-bess-26 \
    --wandb_entity anon-entity \
    --wandb_group       t3_bootstrap_v2

# ── Spectral coverage audit (uses the cached metadata parquet) ──────────────
echo ""
echo ">>> Coverage audit"
if [[ -f "paper/tables/metadata_slim.parquet" ]]; then
    python stats/coverage_audit.py \
        --metadata paper/tables/metadata_slim.parquet \
        --output   stats/results/coverage_audit.json \
        --wandb_project anonym-bess-26 \
        --wandb_entity anon-entity \
        --wandb_group   coverage_audit_v2
else
    echo "  ! paper/tables/metadata_slim.parquet missing"
    echo "    regenerate with : python stats/extract_metadata_slim.py"
fi

# ── Final LaTeX tables + summary.json ───────────────────────────────────────
echo ""
echo ">>> Paper tables (LaTeX)"
python stats/aggregate_results.py

# ── Paper figures ───────────────────────────────────────────────────────────
echo ""
echo ">>> Paper figures"
python stats/figure_teaser.py          || echo "  ! figure_teaser (non-blocking)"
python stats/figure_quality_profile.py || echo "  ! figure_quality_profile (non-blocking)"
python stats/figure_multiseed.py       || echo "  ! figure_multiseed (non-blocking)"

echo ""
echo "=============================================================="
echo "PHASE 3 done  —  $(date)"
echo "Paper artefacts in : paper/tables/   (LaTeX + summary.json)"
echo "=============================================================="
