# Stage 1 Hα — Specialised encoder (Phase A)

MAE encoder specialised on a 100 Å window around Hα (6562.8 Å).

## Motivation

The full-spectrum encoder (stage1/) dilutes the Hα signal in the continuum
reconstruction (Hα = 1.5% of the 4096 bins → ignored by the MAE loss).
This encoder focuses on the 128 physically most relevant bins.

## Architecture

| Parameter | Full encoder | Hα encoder |
|-----------|-------------|------------|
| Input | 4096 bins (~2000 Å) | **128 bins** (100 Å) |
| Patch size | 16 px (overlap 8) | **8 px** (overlap 4) |
| Patches | ~511 | **31** |
| d_model | 256 | **128** |
| Layers | 6 | **4** |
| Params | ~5.3M | **~912K** |
| Train time | ~2h | **~30 min** |

## Usage

### 🚀 Recommended: multi-seed submission in a single `sbatch`

```bash
cd bess_bench/stage1_encoder
mkdir -p logs
sbatch run_multiseed.slurm
```

This single script (see `run_multiseed.slurm`):
1. Trains the 3 official seeds (42, 123, 456) **sequentially on the same GPU node**
   → inter-seed variance = true algorithmic variance (no hardware effect).
2. Evaluates each checkpoint (`evaluate.py`).
3. Aggregates metrics in `runs/multiseed_summary.json`
   (mean ± std over the 3 seeds).
4. Exports the seed-42 encoder in HuggingFace Hub format to
   `hf_export/bemae-halpha-v1/` (ready for `huggingface-cli upload`).

Budget: ~1h30 of A100 GPU + ~15 min eval + export. Idempotent: a seed
already trained (`best.pt` present) is skipped.

### Running a single seed manually

```bash
python train.py --seed 42 --run_name Halpha_all_seed42
python evaluate.py --checkpoint runs/Halpha_all_seed42/best.pt
```

### Local (quick test, CPU)

```bash
python train.py --epochs 5 --device cpu --batch_size 16 --run_name test_local
```

### HuggingFace Hub export (after training)

```bash
python export_hf.py \
    --checkpoint runs/Halpha_all_seed42/best.pt \
    --output_dir hf_export/bemae-halpha-v1 \
    --model_id anonym-submit-26/bemae-halpha-v1

huggingface-cli upload anonym-submit-26/bemae-halpha-v1 hf_export/bemae-halpha-v1
```

The export directory is self-contained: `pytorch_model.bin` (encoder only,
~3 MB), `config.json`, `model.py` + `config.py`, `README.md` (model card)
and `example_usage.py`.

## Files

- `config.py` — Configuration (architecture + training)
- `dataset.py` — Data pipeline: Hα crop ±50 Å → 128 interpolated bins
- `model.py` — SpectralEncoderHalpha + MAEDecoder (no GRL)
- `train.py` — Training loop with early stopping
- `evaluate.py` — 4 diagnostic tests + UMAP
- `export_hf.py` — Encoder export → HuggingFace Hub snapshot
- `run_multiseed.slurm` — SLURM: 3 seeds + eval + export in one job

## Reproducibility (v1.0 benchmark)

- **HF Dataset**: `anonym-submit-26/bess-bench-26`, subset `Halpha_all`
  (a revision tag will be frozen at the time of the public anonymised release).
- **Frozen hyperparams**: see `config.py` (80 epochs, mask_ratio=0.60,
  lr=1e-4, batch=256, patch_size=8 overlap 4 → 31 patches).
- **Hold-out test stars**: `GAM CAS, PLEIONE, ZET TAU, V442 AND, PI AQR,
  DEL SCO, 28 CYG, OME CMA` (fixed in `config.py::TrainConfig.test_stars`).
- **Seeds**: **3 official seeds** for the paper → 42, 123, 456.
  Each seed produces an independent run; the `R² CV(z→EW)` metric
  is aggregated as mean ± std in `paper/tables/summary.json`.
- **Reference hardware**: encoder pre-training was originally run on
  an NVIDIA A100 80 GB (~30 min/seed, ~1.5 GPU-h total for the
  3 seeds). The TS-FM zero-shot inference and the reproducer script
  `scripts/reproduce_paper_seed42.sh` were validated on an
  RTX 2080 Ti (bf16). Both produce numerically equivalent baselines
  (see `REPRODUCE.md`).
- **Published checkpoint**: seed 42 (`runs/Halpha_all/best.pt`) is the
  checkpoint used to generate `data/embeddings_halpha/star_data.pt`
  consumed by T1/T2 and `downstream/t3_ts_fm_baselines.py`.

### Running the 3 official seeds

**Recommended method** — a single `sbatch` does everything (train + eval + HF export):

```bash
cd bess_bench/stage1_encoder
mkdir -p logs
sbatch run_multiseed.slurm
```

**Manual method** (equivalent, without SLURM):

```bash
cd bess_bench/stage1_encoder
for s in 42 123 456 ; do
  python train.py --seed $s --run_name Halpha_all_seed${s}
  python evaluate.py --checkpoint runs/Halpha_all_seed${s}/best.pt
done
python export_hf.py \
    --checkpoint runs/Halpha_all_seed42/best.pt \
    --output_dir hf_export/bemae-halpha-v1 \
    --model_id anonym-submit-26/bemae-halpha-v1
```

Headline numbers are aggregated by `run_multiseed.slurm` in
`runs/multiseed_summary.json` (mean ± std over the 3 seeds).
