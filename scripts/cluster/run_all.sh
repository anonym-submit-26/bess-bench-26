#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_all.sh — End-to-end pipeline launcher for BESS-Bench v1.0
# ══════════════════════════════════════════════════════════════════════════════
#
# Submits every phase to SLURM with dependencies so the cluster runs them in
# the correct order without human intervention :
#
#   Phase 0 ─┐
#            ├─► Phase 2a (CPU downstream, needs embeddings + features)
#   Phase 1 ─┤
#            ├─► Phase 2b (GPU TS-FM baselines, needs features)
#            │
#            └─► Phase 3  (aggregation, needs 0 + 1 + 2a + 2b)
#
# Usage :
#   cd bess_bench/scripts
#   bash run_all.sh
#
# All logs go to bess_bench/scripts/logs/  (one file per phase + job id).
# All results are written next to the scripts they come from
# (stage1_encoder/runs/, reference_results/embeddings/, reference_results/downstream/,
#  reference_results/stats/, paper/tables/).
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

echo "=============================================================="
echo "BESS-Bench v1.0 — full pipeline launcher"
echo "Date : $(date)"
echo "=============================================================="

# ── Phase 0a : PARALLEL multi-seed training (3 × GPU, 1 each) ───────────────
# We submit 3 independent train jobs so they can run on 3 different nodes
# simultaneously (cluster has free a6000/v100/2080 GPUs).
JOB_T42=$(sbatch  --parsable --export=ALL,SEED=42  --job-name=bess-00a-s42  00a_train_seed.slurm)
JOB_T123=$(sbatch --parsable --export=ALL,SEED=123 --job-name=bess-00a-s123 00a_train_seed.slurm)
JOB_T456=$(sbatch --parsable --export=ALL,SEED=456 --job-name=bess-00a-s456 00a_train_seed.slurm)
echo "submitted  Phase 0a (train seed 42,  GPU)         → $JOB_T42"
echo "submitted  Phase 0a (train seed 123, GPU)         → $JOB_T123"
echo "submitted  Phase 0a (train seed 456, GPU)         → $JOB_T456"

# ── Phase 0 : finalize (eval + embeddings + symlink + aggregate + HF export)
# Depends on all 3 training seeds. It will SKIP trains (best.pt exists) and
# only run the per-seed eval + embeddings + multiseed aggregate + HF export.
JOB0=$(sbatch --parsable --dependency=afterok:${JOB_T42}:${JOB_T123}:${JOB_T456} 00_pretrain_encoder.slurm)
echo "submitted  Phase 0  (finalize encoders, waits on 0a) → $JOB0"

# ── Phase 1 : feature extraction (CPU), parallel to Phase 0a/0 ──────────────
JOB1=$(sbatch --parsable 01_extract_features.slurm)
echo "submitted  Phase 1  (feature extraction, CPU)     → $JOB1"

# ── Phase 2a : CPU downstream, waits on 0 + 1 ───────────────────────────────
JOB2A=$(sbatch --parsable --dependency=afterok:${JOB0}:${JOB1} 02a_downstream_cpu.slurm)
echo "submitted  Phase 2a (downstream CPU, waits on 0+1) → $JOB2A"

# ── Phase 2b : GPU TS-FM, waits on 0 + 1 ────────────────────────────────────
# Needs both feature CSVs (Phase 1) AND aggregated embeddings (Phase 0
# finalize, which symlinks data/embeddings_halpha/ -> seed42 master).
JOB2B=$(sbatch --parsable --dependency=afterok:${JOB0}:${JOB1} 02b_ts_fm_gpu.slurm)
echo "submitted  Phase 2b (TS-FM zero-shot GPU, waits on 0+1) → $JOB2B"

# ── Phase 3 : aggregation, waits on 0 + 1 + 2a + 2b ─────────────────────────
JOB3=$(sbatch --parsable --dependency=afterok:${JOB0}:${JOB1}:${JOB2A}:${JOB2B} 03_aggregate_paper.slurm)
echo "submitted  Phase 3  (aggregation + paper tables)   → $JOB3"

echo ""
echo "=============================================================="
echo "All jobs submitted. Monitor with :"
echo "   sjobs"
echo "   squeue -u \$USER"
echo ""
echo "Expected duration :  ~2 h wall-clock  (3× A6000/A100 in parallel)"
echo "Final artefacts :"
echo "   paper/tables/*.tex                (paper LaTeX tables)"
echo "   paper/tables/summary.json         (machine-readable)"
echo "   stage1_encoder/hf_export/…        (encoder HF release)"
echo "   reference_results/stats/*.json    (bootstrap CIs, coverage)"
echo ""
echo "To abort :  scancel $JOB_T42 $JOB_T123 $JOB_T456 $JOB0 $JOB1 $JOB2A $JOB2B $JOB3"
echo "=============================================================="
