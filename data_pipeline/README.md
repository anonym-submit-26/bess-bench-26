# Data Pipeline

Transforms the 339,115 raw BeSS spectra into a clean, generic
HuggingFace dataset that can be reused by any downstream model.

The DuckDB database stays the raw source. The HuggingFace dataset is
the clean intermediate artefact. Model-specific preparation
(tokenisation, normalisation, resampling, ...) is performed downstream
as lightweight `dataset.map()` operations.

## Structure

```
data_pipeline/
├── 01_quality/                              # Module 1 — Quality & filtering
│   ├── compute_snr.py                       # SNR estimation (DER_SNR + continuum)
│   └── rejected_spectra_snr.json            # 29 corrupted spectra
├── 02_cleanup/                              # Module 2 — Metadata cleanup
│   ├── 01_analyze_instruments.py            # Parse 667 instrument names → JSON
│   ├── 02_apply_instrument_cleanup.py       # Lookup → 5 DuckDB columns
│   ├── instrument_lookup.json               # Mapping table (667 entries)
│   ├── 03_validate_coordinates.py           # Coordinate normalisation + 3 clean columns
│   ├── 04_validate_mjd.py                   # MJD validation + temporal_quality flag
│   ├── 05_classify_observer_type.py         # Pro/amateur classification
│   ├── 06_normalize_star_names.py           # Deduplication 4,755 → 1,468 stars
│   ├── star_name_lookup.json                # Star name mapping (1,468 entries)
│   └── 07_flag_echelle_orders.py            # Echelle flag (81.6% echelle)
├── 03_export/                               # Module 3 — HuggingFace export
│   ├── 01_extract_spectra.py                # FITS → sharded Parquet (parallel)
│   ├── 02_build_dataset.py                  # Build DatasetDict (35 cols, splits)
│   ├── 03_push_to_hub.py                    # Push to HuggingFace Hub
│   ├── check_step.py                        # Per-step verification helper
│   ├── export_to_hf.sh                      # SLURM orchestrator (3 steps)
│   └── README.md                            # Export documentation
└── README.md                                # This file
```

## Prerequisites

- DuckDB database: `bess_data.duckdb` (83 columns, 339k spectra)
- FITS files: directory pointed to by `BESS_FITS_DIR` (~15 GB, 339k files)
- Python environment with `duckdb`, `astropy`, `numpy`, `scipy`,
  `datasets`, `huggingface_hub` (see top-level `requirements.txt`)
- HuggingFace token in environment variable `HF_TOKEN`

## Data flow (DuckDB → Dataset)

| Field           | DuckDB                            | Layer-1 action                                        |
|-----------------|-----------------------------------|-------------------------------------------------------|
| SNR             | `snr_estimated`, `snr_continuum`  | Quality threshold, `snr_quality` flag                 |
| Date            | `observation_date`                | Convert to valid MJD + `temporal_quality`             |
| Instrument      | 667 raw names                     | Lookup → `spectrograph_clean` (~50 groups)            |
| Telescope       | 314 raw names                     | Lookup → `telescope_clean` (~100 groups)              |
| Observer        | `observer_type`                   | amateur / professional / unknown                      |
| Star names      | 4,755 raw names                   | Dedup → `star_name_clean` (1,468 unique)              |
| Echelle         | `is_echelle_order`                | Flag (81.6% echelle, 18.4% single-order)              |
| Lat/Lon         | 99.8% present                     | Validation + longitude normalisation                  |
| Exposure        | `fits_exptime`                    | Pass-through                                          |
| Resolution      | 91.5% `fits_bss_itrp`             | Pass-through                                          |
| Heliocentric v. | `fits_bss_vhel`                   | Pass-through                                          |
| Flux            | FITS files                        | Native arrays (Parquet)                               |

**Rule** — Layer 1 *cleans* (consistent names, valid values, numeric
formats) but does not *transform* (no [-1,1] normalisation, no encoding,
no resampling).

## Output dataset

One row per spectrum, 35 columns:

| Column                          | Type            | Description                              |
|---------------------------------|-----------------|------------------------------------------|
| `spectrum_id`                   | string          | DuckDB ID (MD5)                          |
| `wavelength`                    | list[float32]   | Native wavelength grid (Å)               |
| `flux`                          | list[float32]   | Calibrated flux                          |
| `flux_error`                    | list[float32]   | Flux error (nullable)                    |
| `star_name`                     | string          | Canonical name (deduplicated)            |
| `star_name_raw`                 | string          | Original BeSS name                       |
| `ra`, `dec`                     | float64         | J2000 coordinates (degrees)              |
| `observation_date`              | string          | ISO-8601 date                            |
| `mjd`                           | float64         | Modified Julian Date                     |
| `exposure_time`                 | float64         | Exposure time (seconds)                  |
| `snr`                           | float64         | DER_SNR                                  |
| `snr_continuum`                 | float64         | SNR over continuum windows               |
| `snr_quality`                   | string          | excellent/good/medium/low/very_low       |
| `spectrograph`                  | string          | Cleaned spectrograph (lookup)            |
| `telescope`                     | string          | Cleaned telescope (lookup)               |
| `observer_type`                 | string          | amateur / professional / unknown         |
| `is_echelle_order`              | bool            | True if individual echelle order         |
| `site_latitude`                 | float64         | Observer latitude (degrees, rounded)     |
| `site_longitude`                | float64         | Observer longitude (degrees, rounded)    |
| `spectral_resolution`           | float64         | Theoretical instrumental R               |
| `spectral_resolution_measured`  | float64         | Measured R                               |
| `lambda_min`, `lambda_max`      | float64         | Spectral coverage (Å)                    |
| `helio_velocity`                | float64         | Heliocentric velocity (km/s)             |
| `n_pixels`                      | int32           | Number of spectral pixels                |

**Splits** — train (70%) / validation (15%) / test (15%), split **by star**
(not by spectrum) to avoid temporal leakage. Fixed seed (42).

## Execution order

```bash
# 1. Quality
python 01_quality/compute_snr.py

# 2. Metadata cleanup
python 02_cleanup/01_analyze_instruments.py
python 02_cleanup/02_apply_instrument_cleanup.py
python 02_cleanup/03_validate_coordinates.py
python 02_cleanup/04_validate_mjd.py
python 02_cleanup/05_classify_observer_type.py
python 02_cleanup/06_normalize_star_names.py
python 02_cleanup/07_flag_echelle_orders.py

# 3. HuggingFace export
python 03_export/01_extract_spectra.py        # FITS → Parquet (~30-60 min)
python 03_export/02_build_dataset.py          # Assemble + splits (~15-30 min)
python 03_export/03_push_to_hub.py \
    --repo-id "<user>/bess-bench-26"

# Or via SLURM:
sbatch 03_export/export_to_hf.sh
```
