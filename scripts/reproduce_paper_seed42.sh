#!/usr/bin/env bash
# ==============================================================================
# reproduce_paper_seed42.sh
# ──────────────────────────────────────────────────────────────────────────────
# Reproduce all paper results that depend on the canonical encoder (seed 42)
# starting from PUBLIC anonymous artefacts on the HuggingFace Hub:
#
#   model   : https://huggingface.co/anonym-submit-26/bemae-halpha-v1
#   dataset : https://huggingface.co/datasets/anonym-submit-26/bess-bench-26
#
# All output is written under  results_reproduced/  (this repository, never
# anywhere else). After completion, run scripts/compare_with_reference.py to
# compare against the deposited reference_results/.
#
# What this reproduces (everything that uses only seed 42, not all encoder ablations):
#     T1  SpecProbe       (probe_features_results_seed42.json)
#     T2  LineTransfer    (probe_t2_hbeta_to_halpha.json)
#     T3  EWForecast      (t3_pca_ridge_temporal_baseline.json)
#     T3  ts-FM baselines (t3_ts_fm_baselines.json)  [GPU recommended]
#     SNR ablation (appendix)  (snr_ablation_seed{42,123,456}.json)
#
# What this DOES NOT re-run (multi-seed / 24 ablation checkpoints):
#     stage1 multi-seed table, T1 ±std, ablations table, SNR multi-seed.
#     For these, see scripts/cluster/run_all.sh (full pipeline on a SLURM cluster).
#     Reference numbers are deposited in reference_results/ for inspection.
#
# Determinism:
#   - device = cuda (matches the reference run on RTX 2080 Ti) -> bit-identical
#     reproduction of paper numbers when CUDA + bf16 are available
#   - device = cpu (fallback)  -> deterministic fp32 path; matches paper at
#     |Δ|<5e-4 on TS-FM MAE and <1e-2 on probe R² (see compare report)
#   - WANDB_MODE=disabled -> no telemetry
#   - PYTHONHASHSEED, OMP/MKL pinned for reproducibility
#
# Usage:
#     cd anon_bess_files
#     bash scripts/reproduce_paper_seed42.sh                 # GPU (default)
#     DEVICE=cpu bash scripts/reproduce_paper_seed42.sh      # CPU fallback
#     SKIP_TSFM=1 bash scripts/reproduce_paper_seed42.sh     # skip TS-FM
#
# Total runtime: ~20-30 min on RTX 2080 Ti (full pipeline incl. TS-FM).
# ==============================================================================

set -euo pipefail

# ---- trap: print which line failed -----------------------------------------
_on_error() {
    local exit_code=$?
    local line_no=$1
    echo ""
    echo "========================================================"
    echo "ERROR: script failed at line ${line_no} (exit code ${exit_code})"
    echo "Check the log above for the Python traceback."
    echo "========================================================"
}
trap '_on_error $LINENO' ERR

# ---- repo root (assumes this script lives in <repo>/scripts/) ---------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

# ---- determinism / anonymity / no telemetry ----------------------------------
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export WANDB_MODE=disabled
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=0
export TOKENIZERS_PARALLELISM=false

# DEVICE: default to cuda (matches reference run); fallback to cpu if no GPU
if [[ -z "${DEVICE:-}" ]]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        DEVICE=cuda
    else
        DEVICE=cpu
        echo "[INFO] CUDA not available; falling back to DEVICE=cpu"
    fi
fi
SKIP_TSFM="${SKIP_TSFM:-0}"      # 1 -> skip Chronos / TimesFM

# ---- output layout -----------------------------------------------------------
OUT="$REPO/results_reproduced"
mkdir -p \
    "$OUT/assets/hf_model" \
    "$OUT/embeddings" \
    "$OUT/features" \
    "$OUT/downstream" \
    "$OUT/figures" \
    "$OUT/logs"

LOG="$OUT/logs/reproduce.log"
exec > >(tee -a "$LOG") 2>&1

echo "======================================================================="
echo "BESS-Bench reproduction (seed 42)"
echo "Date     : $(date)"
echo "Repo     : $REPO"
echo "Output   : $OUT"
echo "Device   : $DEVICE"
echo "Threads  : OMP=$OMP_NUM_THREADS  MKL=$MKL_NUM_THREADS"
echo "Skip TSFM: $SKIP_TSFM"
echo "======================================================================="

# ---- ensure huggingface_hub is available -------------------------------------
python -c "import huggingface_hub" 2>/dev/null || {
    echo ">>> Installing huggingface_hub..."
    pip install --quiet huggingface_hub
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- Download the public anonymous encoder checkpoint
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "STEP 1 / 6 -- Downloading encoder from HuggingFace Hub"
echo "======================================================================"
CKPT="$OUT/assets/hf_model/pytorch_model.bin"
if [[ -s "$CKPT" && -s "$OUT/assets/hf_model/config.json" ]]; then
    echo "    already present -> $CKPT  ($(du -h "$CKPT" | cut -f1))"
else
    python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="anonym-submit-26/bemae-halpha-v1",
    local_dir="$OUT/assets/hf_model",
    allow_patterns=["pytorch_model.bin", "config.json"],
)
print("downloaded to:", p)
PY
fi
ls -la "$OUT/assets/hf_model/"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- Extract embeddings z ∈ ℝ¹²⁸ from the encoder
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "STEP 2 / 6 -- Extracting embeddings (device=$DEVICE)"
echo "======================================================================"
EMB_FILE="$OUT/embeddings/star_data.pt"
if [[ -s "$EMB_FILE" ]]; then
    echo "    already present -> $EMB_FILE"
else
    python stage1_encoder/precompute_embeddings.py \
        --checkpoint "$CKPT" \
        --output_dir "$OUT/embeddings" \
        --device "$DEVICE" \
        --batch_size 128 \
        --num_workers 2
fi
ls -la "$OUT/embeddings/"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- Compute classical spectral features (FWHM, EW, V/R, …)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "STEP 3 / 6 -- Classical spectral features (Halpha + Hbeta)"
echo "======================================================================"
SPEC="$OUT/features/spectral_features.pt"
HBETA="$OUT/features/spectral_features_hbeta.pt"
if [[ -s "$SPEC" ]]; then
    echo "    already present -> $SPEC"
else
    python downstream/compute_spectral_features.py --output "$SPEC"
fi
if [[ -s "$HBETA" ]]; then
    echo "    already present -> $HBETA"
else
    python downstream/compute_hbeta_features.py --output "$HBETA"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 -- T1 SpecProbe (linear probes z -> 6 features), seed 42
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "STEP 4 / 6 -- T1 SpecProbe (seed 42)"
echo "======================================================================"
python downstream/probe_features.py \
    --embeddings "$EMB_FILE" \
    --features   "$SPEC" \
    --seed 42 \
    --output_suffix "_seed42" \
    --output_dir "$OUT/downstream" \
    --figures_dir "$OUT/figures" \
    --wandb_mode disabled

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 -- T2 LineTransfer (Hbeta features -> Halpha features)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "STEP 5 / 6 -- T2 LineTransfer"
echo "======================================================================"
python downstream/t2_cross_line_probing.py \
    --hbeta_features "$HBETA" \
    --halpha_features "$SPEC" \
    --output "$OUT/downstream/probe_t2_hbeta_to_halpha.json" \
    --output_fig "$OUT/figures/t2_hbeta_to_halpha.png"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 -- T3 EWForecast: PCA+Ridge baseline + (optional) TS foundation models
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "STEP 6 / 6 -- T3 EWForecast"
echo "======================================================================"
python downstream/t3_pca_ridge_temporal_baseline.py \
    --embeddings "$EMB_FILE" \
    --features   "$SPEC" \
    --output_dir "$OUT/downstream" \
    --seed 42 \
    --wandb_mode disabled

echo ""
echo "    --- z-Ridge on encoder embeddings (context=5) ---"
echo "        required by figure_t3_horizon.py (Fig. 6)"
python downstream/t3_z_ridge_temporal_baseline.py \
    --context    5 \
    --embeddings "$EMB_FILE" \
    --features   "$SPEC" \
    --output_dir "$OUT/downstream"

if [[ "$SKIP_TSFM" == "1" ]]; then
    echo ""
    echo "    [skipped] Chronos-Bolt + TimesFM-2.0  (SKIP_TSFM=1)"
else
    python downstream/t3_ts_fm_baselines.py \
        --embeddings "$EMB_FILE" \
        --features   "$SPEC" \
        --output_dir "$OUT/downstream" \
        --device "$DEVICE" \
        --wandb_mode disabled
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 -- SpecProbe per-SNR-bin diagnostic (Appendix, 3 probe seeds)
# ─────────────────────────────────────────────────────────────────────────────
# The encoder is fixed (seed 42 checkpoint shipped on the Hub). The seed
# argument here only randomises the 5-fold CV partitioning of the probe.
# We therefore reproduce the three appendix files from the SAME encoder.
echo ""
echo "======================================================================"
echo "STEP 7 / 7 -- SpecProbe per-SNR-bin diagnostic (probe seeds 42, 123, 456)"
echo "======================================================================"
for s in 42 123 456; do
    python downstream/snr_ablation.py \
        --embeddings "$EMB_FILE" \
        --features   "$SPEC" \
        --seed "$s" \
        --output "$OUT/downstream/snr_ablation_seed${s}.json"
done

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================="
echo "REPRODUCTION COMPLETE"
echo "======================================================================="
echo "Output written to:  $OUT"
echo ""

# ---- regenerate paper figures -----------------------------------------
# All figures are written under  $OUT/figures/  (results_reproduced/figures/).
# The committed canonical copies under  paper/figures/  are NEVER overwritten.
echo "======================================================================="
echo "Regenerating paper figures into $OUT/figures/"
echo "======================================================================="
mkdir -p "$OUT/figures"

# Fig. T3-horizon: depends on freshly produced t3_*.json from $OUT/downstream/.
echo "----- Fig. T3-horizon (from reproduced t3_* JSONs) -----"
python stats/figure_t3_horizon.py \
    --input-dir "$OUT/downstream" \
    --output    "$OUT/figures/fig_t3_horizon.pdf"

# Fig. 1 (teaser gallery) and Fig. 3 (embedding PCA): consume freshly produced
# embeddings + features from $OUT (true seed-42 reproduction).
export BESS_FIG_DIR="$OUT/figures"
export BESS_FEATURES_PATH="$SPEC"
export BESS_EMB_PATH="$EMB_FILE"

echo "----- Fig. 1  (teaser gallery, 5 morphological classes) -----"
python paper/figure_scripts/fig1_teaser_gallery.py

echo "----- Fig. 3  (embedding PCA coloured by morphology) -----"
python paper/figure_scripts/fig3_embedding_pca.py

# Figs. 4, 5, 6: re-render committed multi-seed numbers (reference_results/stats/).
# Multi-seed inputs (t1_multiseed_summary.json, t3_bootstrap_cis.json) are
# produced by Path B only; path A reuses the deposited JSONs to verify that
# the plotting pipeline itself is deterministic.
echo "----- Fig. 4  (Stage-1 probes) -----"
python paper/figure_scripts/fig4_stage1_probes.py

echo "----- Fig. 5  (T1 SpecProbe per-feature R^2) -----"
python paper/figure_scripts/fig5_t1_per_feature.py

echo "----- Fig. 6  (T3 forecast bootstrap) -----"
python paper/figure_scripts/fig6_t3_forecast.py

unset BESS_FIG_DIR BESS_FEATURES_PATH BESS_EMB_PATH

# ---- regenerate paper tables ------------------------------------------
# Same convention: written under  $OUT/tables/ , committed copies under
# paper/tables/  are NEVER overwritten.
echo "======================================================================="
echo "Regenerating paper tables into $OUT/tables/"
echo "======================================================================="
mkdir -p "$OUT/tables"

export BESS_TABLES_DIR="$OUT/tables"
export BESS_DOWNSTREAM_DIR="$OUT/downstream"

# Aggregate Stage-1 multi-seed (reads committed reference_results/stage1_encoder/),
# T1 probes (reads $OUT/downstream/probe_features_results_seed42.json), and
# T3 baselines (reads $OUT/downstream/t3_*.json).
python stats/aggregate_results.py

# T1 95% CI from CV folds.
python stats/bootstrap_ci.py

unset BESS_TABLES_DIR BESS_DOWNSTREAM_DIR

# Note on Fig. 2 (dataset quality profile):
#   Requires  paper/tables/metadata_slim.parquet  built from the full HF
#   snapshot via  stats/extract_metadata_slim.py  (~2 GB download, ~1 min).
#   Skipped from the quick reproduction; the canonical PDF/PNG ships under
#   paper/figures/ for inspection. To regenerate locally:
#       python stats/extract_metadata_slim.py
#       python paper/figure_scripts/fig2_dataset_quality.py
echo ""

find "$OUT" -type f \( -name "*.json" -o -name "*.pt" -o -name "*.pdf" -o -name "*.png" \) 2>/dev/null \
    | sort | sed 's|'"$REPO"'/|    |'
echo ""

# ---- automatic comparison against reference_results/ ----------------------
SKIP_COMPARE="${SKIP_COMPARE:-0}"
if [[ "$SKIP_COMPARE" != "1" && -d "$REPO/reference_results" ]]; then
    echo "======================================================================="
    echo "AUTOMATIC COMPARISON AGAINST reference_results/"
    echo "======================================================================="
    python scripts/compare_with_reference.py \
        --tol-strict 1e-3 \
        --tol-gpu    1e-2 \
        || echo "[INFO] compare_with_reference.py reported deviations; see report above and REPRODUCE.md."
    echo "======================================================================="
else
    echo "To compare against reference numbers, run:"
    echo "    python scripts/compare_with_reference.py"
    echo "======================================================================="
fi
