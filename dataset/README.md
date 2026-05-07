---
license: cc-by-4.0
language:
- en
pretty_name: BESS-Bench
size_categories:
- 100K<n<1M
task_categories:
- feature-extraction
- time-series-forecasting
- other
task_ids:
- multivariate-time-series-forecasting
tags:
- astronomy
- astrophysics
- spectroscopy
- be-stars
- stellar-spectra
- time-series
- representation-learning
- benchmark
- citizen-science
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# BESS-Bench: Long-baseline, mixed-quality Be-star spectra for ML time-series research

> **339 115 spectral rows · 37 624 physical observations · 1 468 Be stars · 35 years · 91.6 % of observations from amateur observers**
> A benchmark for representation learning and temporal modelling on heterogeneous
> stellar spectroscopic time series.
>
> **Three granularity levels** (see §Dataset composition): parquet *rows*
> (atom of the file), *physical observations* (one observing session ≡
> one `(star_name, mjd)` group; an échelle observation spans a median of
> 26 orders), and *unique stars* (split unit). The amateur share is
> $91.6\,\%$ at the observation level and $81.7\,\%$ at the row level
> (échelle orders multiply professional row counts).

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## TL;DR

BESS-Bench is an ML-ready, per-star split release of the BeSS community Be-star
spectral archive, together with:

- **Full-range optical spectra** (1 150 – 10 442 Å native coverage). The
  fraction of spectra that *strictly* cover each commonly studied Be-star
  line at ±50 Å is reported at two granularities — **per physical
  observation** (Hα 77.6 %, Hβ 35.7 %, He I 5876
  36.0 %, Na D 36.0 %, Hγ 32.5 %, Fe II 5169 35.6 %, Hδ 19.4 %, O I 7772 4.0 %)
  and **per parquet row** (row-level coverage used inside tasks: Hα 7.9 %,
  Hβ 0.8 %, …). Both tables are reproduced in §*Line coverage* below.
- A pre-trained Hα-window spectral encoder (803 K parameters, $R^2 = 0.93$
  on held-out stars).
- **Three benchmarked downstream tasks** (**SpecProbe** feature regression,
  **LineTransfer** Hβ → Hα generalisation, **EWForecast** short-horizon
  EW(Hα) forecasting), with fixed protocols and bootstrap / multi-seed
  confidence intervals. The three tasks focus on Hα and Hβ because
  these are the only two lines with row-level coverage large enough
  to support meaningful CI95 bootstraps in v1.0.
- A transparent comparison on EWForecast: a bundled causal temporal
  transformer fails to beat persistence (ratio_mae 1.166 over four
  seeds, *all* seeds > 1.0), while a simple PCA+Ridge temporal
  baseline does (ratio_mae **0.942**, bootstrap CI95 [0.927, 0.957]),
  matching modern zero-shot time-series foundation models.

Full methodology, datasheet and reproducibility instructions are in the
accompanying paper and in [DATASHEET.md](./dataset/DATASHEET.md). Result-level
JSON artefacts (per-task summaries, bootstrap CIs, coverage audits)
live in the companion code repository; this card quotes their content
for self-containedness.

## Quick start

```python
from datasets import load_dataset

# Full dataset (9.5 GB, 21 shards)
ds = load_dataset("anonym-submit-26/bess-bench-26", split="train")

# Or a 500 MB inspection sample
ds = load_dataset("anonym-submit-26/bess-bench-26", split="train", streaming=True)

# Apply the official per-star split
import pandas as pd
splits = pd.read_csv("splits.csv")          # bundled in this repo
train_stars = set(splits.query("split == 'train'").star_name)
train = ds.filter(lambda x: x["star_name"] in train_stars, num_proc=8)
```

Reproduction of every paper table from frozen checkpoints is documented
in `REPRODUCE.md` of the companion code repository.

## Dataset composition

### Three granularity levels

BESS-Bench is distributed as a parquet table, and we distinguish three
nested units of measurement:

| Unit                       | Count   | Definition                                      |
|----------------------------|---------|-------------------------------------------------|
| Parquet rows               | 339 115 | One row = one 1D spectrum or one échelle order  |
| **Physical observations**  |  37 624 | One observing session ≡ one `(star_name, mjd)`  |
| Unique stars (split unit)  |   1 468 | Per-star split, no leakage                      |

The ~9× inflation from physical observations to parquet rows comes from
**9 917 échelle sessions** each stored as 19–40 individual orders
(median 26 orders / session) — an intentional choice that preserves the
native resolution (R ≈ 10 000 – 17 000). The 27 707 remaining
observations are single-range spectra (one row = one observation).

| Stream          | Rows    | Physical obs. | Unique stars | Median R     |
|-----------------|---------|---------------|--------------|--------------|
| Single-range    |  62 476 |  27 707       |  1 354       | 9 000        |
| Échelle orders  | 276 639 |   9 917       |    915       | 11 000       |
| **Total**       | 339 115 |  37 624       |  1 468       | —            |

### Summary metrics

| Metric                         | Value                               |
|--------------------------------|-------------------------------------|
| Parquet rows                   | 339 115                             |
| Physical observations          |  37 624                             |
| Unique stars                   |   1 468                             |
| Temporal coverage              | 1990-03 → 2025-12 (35 years)        |
| Wavelength coverage            | 1 150 – 10 442 Å                    |
| Median SNR                     | 189                                 |
| Amateur fraction (rows)        | 81.7 %                              |
| Amateur fraction (obs.)        | 91.6 %                              |
| Spectrographs / telescopes     | 26 / 62                             |
| Licence                        | CC-BY-4.0                        |

Filter échelle orders with the `is_echelle_order` flag. v1.0 downstream
tasks group by `(star_name, mjd, instrument_setup)` so that an échelle
observation contributes once, not 26 times.

### Wavelength frame convention

The `wavelength` column is shipped **as in the BeSS FITS** (no shift
applied), so users can reproduce any rest frame they wish. For ≈ 97 %
of spectra the released grid is therefore **topocentric**. To reach
the heliocentric frame, apply

    λ_helio = λ_obs · (1 − bss_rqvh / c)         (c = 299 792.458 km/s)

`helio_velocity` (= FITS `BSS_VHEL`) is the correction *already
applied* by the observer (0 for 97 % of spectra) and `bss_rqvh`
(= FITS `BSS_RQVH`) is the residual correction *still to apply*
(populated for 99.8 % of spectra, ±20 km s⁻¹, σ ≈ 10 km s⁻¹).

The bundled BESS-Bench baselines (encoder pre-training, SpecProbe/LineTransfer/EWForecast)
**apply this correction internally** before the 128-bin Hα/Hβ
crop so that features and embeddings live in a common heliocentric
frame compatible with external surveys (SDSS, LAMOST, APOGEE).

### Line coverage — two complementary tables

BESS-Bench ships the native wavelength coverage of each observation (not
a cropped Hα window). Two coverage tables are computed on the v1.0
snapshot (audit script and full JSON in the companion code repository).

**Table A — coverage per physical observation** *(astrophysical view:
of the 37 624 observing sessions, which contain this line at ±50 Å?
Échelle rows are aggregated via λ<sub>min</sub>=min, λ<sub>max</sub>=max
over their orders.)*

| Line        | λ₀ (Å)  | Coverage | Observations covered (out of 37 624) |
|-------------|--------:|---------:|-------------------------------------:|
| Hα          |  6562.8 | 77.63 %  | 29 207                               |
| Na D        |  5893.0 | 35.98 %  | 13 536                               |
| He I 5876   |  5876.0 | 35.97 %  | 13 532                               |
| Hβ          |  4861.3 | 35.66 %  | 13 416                               |
| Fe II 5169  |  5169.0 | 35.55 %  | 13 376                               |
| Hγ          |  4340.5 | 32.54 %  | 12 242                               |
| Hδ          |  4101.7 | 19.45 %  |  7 317                               |
| O I 7772    |  7772.0 |  4.05 %  |  1 523                               |

**Table B — coverage per parquet row** *(per-row coverage, used when
tasks iterate over individual parquet rows rather than over observing
sessions)*

| Line        | λ₀ (Å)  | Coverage | Rows covered (out of 339 115) |
|-------------|--------:|---------:|------------------------------:|
| Hα          |  6562.8 |  7.94 %  |  26 937                       |
| Na D        |  5893.0 |  3.37 %  |  11 413                       |
| He I 5876   |  5876.0 |  1.94 %  |   6 570                       |
| Hβ          |  4861.3 |  0.78 %  |   2 652                       |
| Fe II 5169  |  5169.0 |  0.71 %  |   2 421                       |
| Hγ          |  4340.5 |  0.69 %  |   2 337                       |
| Hδ          |  4101.7 |  0.66 %  |   2 246                       |
| O I 7772    |  7772.0 |  0.41 %  |   1 387                       |

The 10× gap between the two tables is explained entirely by the échelle
decomposition: each échelle row covers a narrow sub-range by
construction, so a single observing session that covers Hα contributes
26 rows in Table B but only 1 observation in Table A. Wavelength
distribution across rows: λ<sub>min</sub> p5/p50/p95 = 3 980 / 5 277 /
7 006 Å; λ<sub>max</sub> p5/p50/p95 = 4 113 / 5 413 / 7 351 Å.

v1.0 baselined tasks (SpecProbe, LineTransfer, EWForecast) use the
Hα and Hβ windows because they combine high observation-level
coverage (77.6 % / 35.7 %) with enough row-level density (26 937 /
2 652 spectra) for stable CVs. He I, Na D and Fe II 5169 each reach
~36 % observation-level coverage and are released for community
experimentation, but no v1.0 leaderboard task targets them.

### Splits

Split **by star** (not by spectrum) to prevent temporal leakage:

| Split       | Stars | Spectra  | Fraction |
|-------------|-------|----------|----------|
| train       | 1 024 | 241 057  | ≈71 %    |
| validation  | 219   | 44 383   | ≈13 %    |
| test        | 225   | 53 675   | ≈16 %    |

Assignment is deterministic via
`md5("bess_bench_v1::" + star_name)[:8] mod 100` ↦
{0–69: train, 70–84: val, 85–99: test}. The result is bundled as
`splits.csv`; the generating script is shipped in the companion code
repository for full reproducibility.

## Schema

| Column                          | Type           | Description                                  |
|---------------------------------|----------------|----------------------------------------------|
| `wavelength`                    | list[float32]  | Native wavelength grid (Å)                   |
| `flux`                          | list[float32]  | Calibrated flux                              |
| `flux_error`                    | list[float32]? | Flux uncertainty (null for a substantial fraction of amateur FITS; see DATASHEET §3) |
| `spectrum_id`                   | string         | MD5 (star_name, mjd, instrument_setup)       |
| `star_name`                     | string         | Canonical de-duplicated name (1 468 unique)  |
| `star_name_raw`                 | string         | Original BeSS object name                    |
| `ra`, `dec`                     | float64        | J2000 (degrees)                              |
| `lambda_min`, `lambda_max`      | float64        | Spectral coverage (Å)                        |
| `n_pixels`                      | int32          | Number of spectral pixels                    |
| `spectral_resolution`           | float64        | Theoretical R                                |
| `spectral_resolution_measured`  | float64?       | Measured R (null when unavailable)           |
| `snr`                           | float64        | DER_SNR                                      |
| `snr_continuum`                 | float64        | Continuum-window SNR                         |
| `snr_quality`                   | string         | excellent/good/medium/low/very_low/unknown   |
| `observation_date`              | string         | ISO-8601 date                                |
| `mjd`                           | float64        | Modified Julian Date                         |
| `exposure_time`                 | float64        | Seconds                                      |
| `temporal_quality`              | string         | good / borderline / unknown                  |
| `spectrograph`                  | string         | Normalised                                   |
| `telescope`                     | string         | Normalised                                   |
| `detector`                      | string         | Normalised                                   |
| `instrument_setup`              | string         | Canonical (spectrograph, telescope) pair     |
| `instrument_confidence`         | string         | high / medium / low                          |
| `observer_type`                 | string         | amateur / professional / unknown             |
| `site_latitude`                 | float64?       | Degrees, rounded to 0.1° (≈11 km)              |
| `site_longitude`                | float64?       | Degrees, rounded to 0.1° (≈11 km)              |
| `site_elevation`                | float64?       | Metres, rounded to 100 m                     |
| `helio_velocity`                | float64?       | Heliocentric correction *already applied* by the observer (= FITS `BSS_VHEL`, km s⁻¹). 0 for ≈97 % of spectra. |
| `bss_rqvh`                      | float64?       | Residual heliocentric correction *to apply* to reach the heliocentric frame (= FITS `BSS_RQVH`, km s⁻¹). ±20 km s⁻¹, 1σ≈10 km s⁻¹, populated for 99.8 % of spectra. |
| `telluric_corrected`            | string         | Raw FITS header string (free-form, often `None`) |
| `normalized`                    | string         | Raw FITS header string (free-form, often `None`) |
| `is_echelle_order`              | bool           | True when the row is one order of an echelle |
| `fits_format`                   | string         | Original FITS flavour                        |

## Benchmark protocol

BESS-Bench v1.0 releases **three quantitative tasks** (SpecProbe,
LineTransfer, EWForecast). Every task reports reproducible mean ± std
and/or bootstrap CI95 over seeds {42, 123, 456} or 10 000 bootstrap
resamples, under the rule that a learned-model claim of improvement
requires the corresponding CI to exclude the baseline value.

### SpecProbe — Hα single-line spectral-feature regression (Ridge probe, 3-seed)

Frozen 128-D embeddings from the released Hα-window encoder → Ridge
regression on Hα spectral features: EW, FWHM, central depth, Δv,
vr_ratio, peak intensity. 5-fold GroupKFold on `star_id` over 26 858
scored Hα spectra (Δv / vr_ratio defined on the 10 804 cleanly
double-peaked profiles), repeated across encoder seeds {42, 123, 456}. Baselines: Ridge on
PCA(10) and PCA(50) of the raw 128-D normalised Hα flux. Reported
numbers (mean ± std over 3 seeds; full per-fold artefacts in the
companion code repository):

| feature          | z (128D, encoder)   | PCA(10)            | PCA(50)            |
|------------------|---------------------|--------------------|--------------------|
| `ew`             | 0.9951 ± 0.001      | 0.9998 ± 0.000     | 1.0000 ± 0.000     |
| `fwhm`           | **0.5704 ± 0.036**  | 0.1329 ± 0.000     | 0.1730 ± 0.000     |
| `central_depth`  | **0.9250 ± 0.006**  | 0.1062 ± 0.000     | 0.1627 ± 0.000     |
| `delta_v`        | **0.5679 ± 0.020**  | 0.2479 ± 0.000     | 0.3077 ± 0.000     |
| `vr_ratio`       | **0.4010 ± 0.030**  | 0.2838 ± 0.000     | 0.2868 ± 0.000     |
| `peak_intensity` | 0.9960 ± 0.001      | 0.9640 ± 0.000     | 0.9664 ± 0.000     |

The encoder adds clear value on FWHM (kinematics), central depth
(opacity), Δv and V/R ratio (kinematics); EW and peak_intensity are
already linearly decodable from the raw flux so the encoder's
marginal value is expected to be small.

### LineTransfer — Cross-line probing (Hβ → Hα features)

Same Hα labels as SpecProbe, input is the Hβ ±50 Å window (128 bins, locally
normalised), restricted to the 2 525 (star, MJD) pairs where both
lines are observed. Demonstrates whether a second Balmer line carries
predictive signal for Hα features. Best result (full per-feature
breakdown in the companion code repository):

| Target          | Best Hβ representation | R² ± CV std |
|-----------------|------------------------|-------------|
| `ew` (Hα)       | Hβ PCA(10)             | 0.692 ± 0.029 |
| other features  | all representations    | ≈ 0 (no transferrable signal) |

### EWForecast — Short-horizon EW(Hα) forecasting (bootstrap CI95)

One-step-ahead forecast of EW(Hα) at the next observation epoch given
past sequences. Primary metric: `ratio_mae = MAE(model) / MAE(persistence)`,
computed on the v1.0 temporal test set (cutoff MJD = 59215). **A
published claim of beating persistence requires the CI95 to exclude
1.0 from above** (for learned temporal transformers: all four seeds'
ratios must be < 1.0; for deterministic baselines: bootstrap CI95
on per-prediction residuals, n = 10 000, must exclude 1.0). Full
bootstrap artefacts in the companion code repository.

| Method                          | point ratio | CI95 / per-seed spread             | beats persistence? |
|---------------------------------|------------:|------------------------------------|:------------------:|
| Transformer Δz+Δt, 4 seeds      | 1.166       | [1.095, 1.279] (spread of 4 seeds) | **no (all seeds > 1.0)** |
| PCA(10)+Ridge, ctx=5            | **0.942**   | bootstrap CI95 [0.927, 0.957]      | **yes**            |
| PCA(10)+Ridge, ctx=10           | 0.956       | bootstrap CI95 [0.937, 0.976]      | **yes**            |
| Chronos-Bolt Base (zero-shot)   | 0.968       | bootstrap CI95 [0.955, 0.980]      | **yes**            |
| TimesFM-2.0-500M (zero-shot)    | 0.969       | bootstrap CI95 [0.949, 0.989]      | **yes**            |
| PCA(50)+Ridge, ctx=5            | 0.983       | bootstrap CI95 [0.966, 1.002]      | inconclusive       |
| PCA(50)+Ridge, ctx=10           | 1.040       | bootstrap CI95 [1.016, 1.065]      | no                 |

The headline finding of the temporal protocol: the bundled learned
transformer is *worse* than persistence on every seed, while a simple
PCA+Ridge temporal baseline (best 0.942) and modern zero-shot TS-FMs
(Chronos-Bolt Base 0.968, TimesFM-2.0 0.969) beat persistence with
comfortable margins.

## Citation

```bibtex
@misc{bess_bench_2026,
  title        = {BESS-Bench: Long-baseline Be-star spectra for ML time-series research},
  author       = {Anonymous, for the NeurIPS 2026 D\&B Track review},
  year         = {2026},
  note         = {See paper for final authorship list upon acceptance.},
  howpublished = {\url{https://huggingface.co/datasets/anonym-submit-26/bess-bench-26}},
}
```

When using BESS-Bench, please also cite the underlying BeSS database:

```bibtex
@article{neiner2011bess,
  title   = {The BeSS database},
  author  = {Neiner, C. and others},
  journal = {Astronomical Journal},
  volume  = {142},
  pages   = {149},
  year    = {2011},
  doi     = {10.1088/0004-6256/142/5/149}
}
```

## Licence

CC-BY-4.0. Attribution must credit the BeSS database (Neiner et al.
2011) and the BeSS consortium (LESIA / Observatoire de Paris).
Commercial use is permitted under the same attribution conditions.

## Acknowledgements

We thank the BeSS community of amateur and professional observers for
35 years of sustained spectroscopic monitoring. BESS-Bench packages
and structures these public observations for ML use, and credits the
original observers via the BeSS database.
