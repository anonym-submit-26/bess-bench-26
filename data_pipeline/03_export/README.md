# 03_export — HuggingFace Hub export

Pipeline that exports the BeSS foundation dataset to the HuggingFace Hub.

## Prerequisites

1. **HuggingFace token** with write access:
   - Create one at https://huggingface.co/settings/tokens
   - Make it available either through the environment:
     ```
     export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
     export HF_REPO_ID=<user>/bess-bench-26
     ```
     or via a `.env` file at the project root.

2. **Python packages** (top-level `requirements.txt`):
   `datasets`, `huggingface_hub`, `astropy`, `duckdb`, `pyarrow`.

3. **Data sources**:
   - DuckDB: `bess_data.duckdb` (339k spectra, 83 columns)
   - FITS: directory pointed to by `BESS_FITS_DIR` (~15 GB)

## Scripts

| Step | Script                  | Description                            | Approx. duration |
|------|-------------------------|----------------------------------------|------------------|
| 1    | `01_extract_spectra.py` | FITS → Parquet shards                  | ~30-60 min       |
| 2    | `02_build_dataset.py`   | Metadata + spectra → DatasetDict       | ~15-30 min       |
| 3    | `03_push_to_hub.py`     | Push to HuggingFace Hub                | ~30-60 min       |
| —    | `check_step.py`         | Per-step verification helper           | <1 min           |

## Execution

### Option A — SLURM (recommended)

```bash
# 1. Configure the HF token
export HF_TOKEN=hf_xxx
export HF_REPO_ID=<user>/bess-bench-26

# 2. Edit the repo ID inside the SLURM script
vi export_to_hf.sh

# 3. Submit
cd data_pipeline/03_export
sbatch export_to_hf.sh

# 4. Follow
tail -f ../../logs/export_hf_<jobid>.out
```

### Option B — Manual, step by step

```bash
cd data_pipeline/03_export
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"

# Step 1 — FITS → Parquet
python 01_extract_spectra.py --workers 8

# Step 2 — Assemble DatasetDict
python 02_build_dataset.py

# Step 3 — Push (private by default)
python 03_push_to_hub.py --repo-id "<user>/bess-bench-26"
```

### Option C — Quick test (100 spectra)

```bash
python 01_extract_spectra.py --limit 100 --workers 4
python 02_build_dataset.py
python 03_push_to_hub.py --repo-id "<user>/bess-bench-test" --dry-run
```

## Output layout

```
03_export/
├── 01_extract_spectra.py
├── 02_build_dataset.py
├── 03_push_to_hub.py
├── check_step.py
├── export_to_hf.sh
├── README.md
├── parquet_spectra/          ← produced by step 1
│   ├── shard_000000.parquet
│   ├── shard_000001.parquet
│   └── ...
└── hf_dataset/               ← produced by step 2
    ├── train/
    ├── validation/
    └── test/
```

## Dataset schema

35 columns total:

- **Spectral arrays**: `wavelength`, `flux`, `flux_error` (Sequence[float32])
- **Identity**: `spectrum_id`, `star_name`, `star_name_raw`, `ra`, `dec`
- **Spectral**: `lambda_min`, `lambda_max`, `n_pixels`, `spectral_resolution`
- **Quality**: `snr`, `snr_continuum`, `snr_quality`
- **Time**: `observation_date`, `mjd`, `exposure_time`
- **Instrument**: `spectrograph`, `telescope`, `observer_type`, `is_echelle_order`
- **Geography**: `site_latitude`, `site_longitude`, `helio_velocity`

## Splits

- **train**: ~70% of the stars
- **validation**: ~15% of the stars
- **test**: ~15% of the stars

Splits are computed **by star** (not by spectrum) to avoid leakage.

## Notes

- The dataset is created as **private** by default.
- To make it public: `python 03_push_to_hub.py --public --repo-id ...`
- 81.6% of the entries are individual echelle orders (see `is_echelle_order`).
