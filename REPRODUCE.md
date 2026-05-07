# Reproducing BESS-Bench results

This document describes how to reproduce every number reported in the paper,
**from a fresh clone**, with two complementary paths:

| Path | Hardware | Time | Coverage |
|---|---|---|---|
| **A. Quick verification** (recommended for reviewers) | 1 GPU (RTX 2080 Ti or equivalent), or CPU fallback | ~25 min on GPU / ~3 h on CPU | seed 42 only -- verifies all headline numbers |
| **B. Full pipeline** | 1 GPU + 8 CPU on a SLURM cluster | ~3 h wall-clock, ~22 GPU-h total | 3 seeds, ablations, bootstrap CIs, SNR auxiliary table |

The two paths share the same code base; path A is a single-script wrapper
that pins `seed=42`, uses GPU when available (matching the reference run),
and skips the multi-seed / ablation grid.

> **Note on the SLURM scripts.** The files under `scripts/cluster/` activate
> a conda environment named `envglobal`. Adapt the `conda activate envglobal`
> line (and any `module load` commands above it) to match your local cluster
> setup; the actual Python dependencies are listed in `requirements.txt`.

---

## Path A -- Quick verification (single command)

### 1. Clone and create environment

```bash
git clone https://github.com/anonym-submit-26/bess-bench-26.git && cd script
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is self-contained and pins all needed packages:
`torch>=2.4`, `datasets>=4.0`, `huggingface_hub>=0.27`, `scikit-learn`,
`numpy`, `scipy`, `astropy`, `matplotlib`, `chronos-forecasting>=2.2.0`,
`timesfm>=1.3.0`, `wandb`. Tested with Python 3.10 and 3.11.

> **Note.** `chronos-forecasting` and `timesfm` are only required for
> EWForecast TS-FM baselines (step 6 below). Skip them if you only need
> SpecProbe/LineTransfer/PCA-Ridge.

### 2. Run the reproduction script

```bash
bash scripts/reproduce_paper_seed42.sh
```

Device selection is automatic: GPU (CUDA, bf16 for TS-FM inference) when
available -- this matches the reference run -- otherwise CPU (deterministic
fp32 fallback). Override explicitly with `DEVICE=cpu bash scripts/reproduce_paper_seed42.sh`
or `DEVICE=cuda ...`.

The script:

1. downloads the encoder checkpoint `anonym-submit-26/bemae-halpha-v1`
   (3 MB) and the BESS-Bench dataset `anonym-submit-26/bess-bench-26`
   (~5 GB cached locally, first run only);
2. recomputes encoder embeddings on the selected device;
3. extracts H-alpha and H-beta classical features;
4. runs SpecProbe (seed 42), LineTransfer (H-beta -> H-alpha), EWForecast
   PCA+Ridge, and EWForecast TS-FM zero-shot (Chronos-Bolt-S/B + TimesFM-2.0);
5. **automatically calls `compare_with_reference.py`** at the end and prints
   a per-metric report.

All outputs land under `results_reproduced/`:

```
results_reproduced/
├── assets/hf_model/               # downloaded BeMAE checkpoint
├── embeddings/star_data.pt        # CPU-recomputed encoder outputs
├── features/spectral_features.pt  # Hα features (EW, FWHM, V/R, …)
├── features/spectral_features_hbeta.pt
├── downstream/
│   ├── probe_features_results_seed42.json
│   ├── probe_t2_hbeta_to_halpha.json
│   ├── t3_pca_ridge_temporal_baseline.json
│   └── t3_ts_fm_baselines.json
├── compare_report.txt             # diff vs reference_results/
└── logs/reproduce.log
```

### 3. Read the reproducibility report

The report (also written to `results_reproduced/compare_report.txt`) flags
each metric:

| Flag | Tolerance | Meaning |
|---|---|---|
| `STRICT` | $\|\Delta\| < 10^{-3}$ | bit-near-identical reproduction |
| `GPU-OK` | $\|\Delta\| < 10^{-2}$ | acceptable numerical noise (e.g.\ bf16 vs fp32, RidgeCV alpha-grid quantisation) |
| (above tolerance) | $\|\Delta\| > 10^{-2}$ | flagged for inspection |

The reference results in `reference_results/` were produced on the original
training machine (RTX 2080 Ti, CUDA 12, bf16 for TS-FM inference). Expected
profile of the verification (audited on `seed 42`, `141` metrics):

| Task | # metrics | STRICT | GPU-OK | above tol |
|---|---:|---:|---:|---:|
| T1 SpecProbe | 72 | 37 | 33 | 2 |
| T2 LineTransfer | 36 | 36 | 0 | 0 |
| T3 EWForecast PCA+Ridge | 24 | 24 | 0 | 0 |
| T3 EWForecast TS-FM | 9 | 9 | 0 | 0 |
| **Total** | **141** | **106 (75%)** | **33 (23%)** | **2 (1.4%)** |

The two metrics that exceed `tol_gpu` are both on the same combination
`vr_ratio` x `z (128D)`:

- `vr_ratio / z (128D) / r2_mean`  -- ref `0.3831` vs reproduced `0.3667` ($\|\Delta\| = 1.65 \cdot 10^{-2}$)
- `vr_ratio / z (128D) / r2_std`   -- ref `0.0861` vs reproduced `0.0374` ($\|\Delta\| = 4.87 \cdot 10^{-2}$)

This is **not** a code/checkpoint drift but a documented `RidgeCV` instability
on this specific cell: `vr_ratio` is only defined on the ~22% of spectra that
show a double peak, with intrinsic $R^2 \approx 0.38$ (noisy target); the inner
leave-one-out CV inside `RidgeCV` selects $\alpha$ from a 20-point geometric
grid, and a $10^{-5}$-level perturbation of the embedding (precision boundary
between reference run and reproduction) flips the chosen $\alpha$ to a
neighbouring value, which shifts the out-of-fold $R^2$ by a few centi-units.
All other `vr_ratio` representations (PCA(10/32/50), raw-flux-128D,
handcrafted-5D-LOO) are STRICT or GPU-OK with $\|\Delta\| < 7 \cdot 10^{-3}$,
confirming the locality of the effect. **No reported number in the paper is
affected.**

### 4. Determinism settings

The script pins everything that can be pinned:

```bash
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=disabled
```

When `DEVICE=cpu`, all linear algebra runs in deterministic fp32. When
`DEVICE=cuda`, only the TS-FM inference path uses bf16 (matching the
reference run); BeMAE encoder remains in fp32.

---

## Path B -- Full pipeline (multi-seed, ablations, GPU)

### 1. SLURM submission (recommended on a cluster)

```bash
cd scripts/cluster
bash run_all.sh
```

This submits 5 SLURM jobs chained by `--dependency=afterok`:

| # | Phase | Resources | Duration | What it does |
|---|------|-----------|----------|--------------|
| 0 | `00_pretrain_encoder.slurm`  | 1 GPU, 8 CPU, 32 GB | ~1h30/seed | Pre-trains BeMAE-H-alpha for seeds 42/123/456, evaluates, exports seed 42 to HuggingFace format. |
| 1 | `01_extract_features.slurm`  | 8 CPU, 32 GB        | ~1h     | Extracts H-alpha and H-beta spectral features. Runs in parallel with phase 0. |
| 2a | `02a_downstream_cpu.slurm`  | 8 CPU, 32 GB        | ~45 min | SpecProbe x 3 seeds, profile classification, LineTransfer, EWForecast PCA+Ridge, SNR auxiliary table. Waits on 0 + 1. |
| 2b | `02b_ts_fm_gpu.slurm`       | 1 GPU, 4 CPU, 32 GB | ~30 min | EWForecast zero-shot Chronos-Bolt + TimesFM-2.0. Waits on 1. |
| 3 | `03_aggregate_paper.slurm`   | 4 CPU, 16 GB        | ~5 min  | Multi-seed aggregation, bootstrap CI95, coverage audit, LaTeX tables. Waits on all. |

Total wall-clock approx 3 h end-to-end; total compute approx 22 GPU-h + 1.5 CPU-h.

Logs land in `scripts/cluster/logs/<phase>_<jobid>.{out,err}`.

### 2. Without SLURM (single workstation, 1 GPU)

Every `.slurm` script is a plain Bash script -- the `#SBATCH` lines are just
comments. Run sequentially:

```bash
cd scripts/cluster && mkdir -p logs
bash 00_pretrain_encoder.slurm
bash 01_extract_features.slurm
bash 02a_downstream_cpu.slurm
bash 02b_ts_fm_gpu.slurm
bash 03_aggregate_paper.slurm
```

> **Why ship the cluster scripts?** They document, exactly and at the
> command level, the sequence used to produce the canonical numbers in
> `reference_results/`. Path A (`reproduce_paper_seed42.sh`) is a
> single-seed subset of the same pipeline; the cluster scripts let a
> reviewer audit that no hidden divergence exists between path A and the
> reference run, and let a downstream user re-train the encoder or extend
> the ablation grid.

### 3. Re-running a single phase

Phases are idempotent (each script skips outputs that already exist). To
force a re-run, delete the relevant output JSON/PT files and resubmit.

---

## Reference results layout

The directory `reference_results/` ships the canonical paper numbers for
diff-based verification:

```
reference_results/
├── stage1_encoder/
│   ├── multiseed_summary.json                          # Table 2
│   └── Halpha_all_seed{42,123,456}/eval_results.json
├── downstream/
│   ├── probe_features_results_seed{42,123,456}.json    # T1 SpecProbe
│   ├── probe_features_results_ablation_*.json          # 27 ablation runs
│   ├── snr_ablation_seed{42,123,456}.json              # robustness
│   ├── probe_t2_hbeta_to_halpha.json                   # T2 LineTransfer
│   ├── t3_pca_ridge_temporal_baseline.json             # T3 EWForecast classical
│   ├── t3_ts_fm_baselines.json                         # T3 EWForecast TS-FM
│   ├── spectral_features_stats.json
│   └── spectral_features_hbeta.manifest.json
└── stats/
    ├── t1_multiseed_summary.json                       # T1 across seeds
    ├── ablations_summary.json                          # ablation grid
    ├── t3_bootstrap_cis.json                           # CI95 EWForecast
    ├── coverage_audit.json                             # spectral coverage
    └── authoritative_numbers.json                      # paper master file
```

All files are produced by phases 0–3 of path B; path A regenerates the
seed-42 subset and `compare_with_reference.py` flags any divergence.

---

## Collection pipeline (not reproduced)

The harvesting code that downloads raw FITS from the BeSS SSA endpoint is
**not** redistributed — it depends on an external VO service and
re-downloading 339 k spectra takes ≈ 30 h. The full collection protocol is
described in the paper (`Section: Data collection`) and in
`dataset/DATASHEET.md`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: chronos` / `timesfm` | `pip install chronos-forecasting timesfm` (only needed for EWForecast TS-FM) |
| `OOM` during encoder pre-training | Reduce `batch_size` in `configs/baseline.yaml` (default 256, halve for <8 GB GPUs) |
| TimesFM hangs on CPU | Expected — single TimesFM zero-shot run takes ~10 min on CPU. Reduce `--batch_size 16` if you have <8 GB RAM. |
| `compare_with_reference.py` flags a deviation on `vr_ratio/z (128D)` | Known and documented above (RidgeCV alpha-selection instability on a noisy low-$R^2$ target). Does not affect any paper number. |
| Wandb prompts for login | The script sets `WANDB_MODE=disabled`; if a script ignores it, run `wandb disabled` once. |
