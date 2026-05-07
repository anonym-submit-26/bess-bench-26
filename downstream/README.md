# Downstream — BESS-Bench v1.0 evaluation

Evaluation tasks on the `z ∈ ℝ¹²⁸` embeddings produced by BeMAE
(pre-trained spectral encoder, Hα ±50 Å window, 26 937 spectra at
parquet-row level).

Official protocol: see `EVALUATION_PROTOCOL.md` at the repository root.

---

## Structure

```
downstream/
├── compute_spectral_features.py      # Prerequisite — Hα features (EW, FWHM, V/R, Δv, depth, type)
├── compute_hbeta_features.py         # Prerequisite — normalised Hβ flux (T2 only)
│
│  ── Benchmarked tasks (v1.0) ─
├── probe_features.py                 # T1 SpecProbe   — Ridge CV z -> 6 Hα features
├── t2_cross_line_probing.py          # T2 LineTransfer — Ridge Hβ -> Hα features
├── t3_pca_ridge_temporal_baseline.py # T3 EWForecast — PCA(K)+Ridge baseline
├── t3_ts_fm_baselines.py             # T3 EWForecast — Chronos-Bolt + TimesFM (zero-shot)
├── t3_z_ridge_temporal_baseline.py   # T3 EWForecast — z+Ridge temporal
│
│  ── Validations and ablations ─
├── validate_morphology_audit.py      # Cross-check classify_profiles - compute_spectral_features labels
├── snr_ablation.py                   # SNR-bin robustness study (paper Appendix A.8)
│
│  ── Exploratory analyses (not benchmarked in v1.0) ─
├── classify_profiles.py              # KNN/MLP z -> morphological type (exploratory)
└── clustering_analysis.py            # K-Means/HDBSCAN on z (exploratory)
```

---

## Prerequisites

1. **Pre-computed embeddings** (3 seeds): `../data/embeddings_halpha_seed{42,123,456}/star_data.pt`
   → produced by `stage1_encoder/precompute_embeddings.py` after training.
2. **Dataset**: `anonym-submit-26/bess-bench-26` (HuggingFace, config `default`).
3. **Environment**: `.venv/` at the repository root — `source .venv/bin/activate`.

---

## Execution

The downstream scripts are driven by SLURM from `scripts/cluster/`:

```bash
# Full pipeline from the repository root:
bash scripts/cluster/run_all.sh

# Phase 1 only (features):
sbatch scripts/cluster/01_extract_features.slurm

# Phase 2 (tasks, depends on phases 0 + 1):
sbatch scripts/cluster/02a_downstream_cpu.slurm   # T1 / T2 / T3-PCA
sbatch scripts/cluster/02b_ts_fm_gpu.slurm        # T3 zero-shot TS-FM (GPU)

# Manual execution (development):
python compute_spectral_features.py       # ~30 min (loads HF dataset)
python compute_hbeta_features.py          # ~5 min
python probe_features.py                  # T1 ~2 min
python t2_cross_line_probing.py           # T2 ~2 min
python t3_pca_ridge_temporal_baseline.py  # T3 baseline ~3 min
python t3_ts_fm_baselines.py              # T3 TS-FM ~15 min (GPU)
python t3_z_ridge_temporal_baseline.py    # T3 z+Ridge ~2 min
```

---

## Benchmarked tasks (v1.0)

### T1 — SpecProbe (`probe_features.py`)
RidgeCV (5-fold GroupKFold on `star_id`, encoder seeds 42/123/456):
`z -> {EW, FWHM, V/R, Δv, central_depth, peak_intensity}`.
Baselines: PCA(10) + Ridge and PCA(50) + Ridge on raw normalised flux.
Metric: mean R² ± std across seeds, Gaussian CI95 over the 5 folds.

### T2 — LineTransfer (`t2_cross_line_probing.py`)
RidgeCV: Hβ flux (or PCA of Hβ flux) -> Hα features.
2 525 (star, MJD) pairs with co-observed Hα and Hβ.
Question: which Hα features can be predicted from Hβ alone?

### T3 — EWForecast (`t3_*_temporal_baseline.py`, `t3_ts_fm_baselines.py`)
One-step prediction of EW(Hα) from a windowed context of L ∈ {5, 10}
past observations. Temporal split at MJD 59215.0 (train < cutoff,
test ≥ cutoff). Metric: `ratio_MAE = MAE(model) / MAE(persistence)`,
< 1.0 means improvement over persistence.
Baselines: PCA(K)+Ridge (K ∈ {10, 50}), z+Ridge, Chronos-Bolt
(Small/Base, zero-shot), TimesFM-2.0 (zero-shot).

---

## Exploratory analyses (not benchmarked in v1.0)

### Morphology (`classify_profiles.py`)
KNN / MLP: `repr -> morphological type` (absorption / single / double /
double_asym / shell). Compares z vs hand-crafted features (6D) vs PCA.
**Exploratory**: not part of the v1.0 protocol (`EVALUATION_PROTOCOL.md`).

### Clustering (`clustering_analysis.py`)
K-Means + HDBSCAN on z / features / PCA. Metrics: Silhouette, NMI, ARI
vs `profile_type`; UMAP plus temporal trajectories. **Exploratory**: not
part of the v1.0 protocol.
