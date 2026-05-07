# Datasheet — BESS-Bench

This datasheet follows the template of Gebru et al. (2021), "Datasheets for Datasets",
with additional fields recommended by the NeurIPS 2026 Datasets & Benchmarks Track
(Croissant RAI extension).

Dataset identifier: **BESS-Bench v1.0** · Release date: **May 2026**
License: **CC-BY-4.0** (permissive reuse with attribution to the BeSS
database and the BeSS consortium at LESIA / Observatoire de Paris).

---

## 1. Motivation

### For what purpose was the dataset created?
BESS-Bench was created to enable reproducible machine-learning research on
**long-baseline, mixed-quality stellar spectral time series**. It targets three
gaps in existing astronomical ML benchmarks:

1. Most public spectral datasets (SDSS, LAMOST, Gaia RVS) are essentially
   single-epoch and exclude amateur data by design. BESS-Bench is the first
   homogenised, ML-ready release of the BeSS collaborative archive, which
   spans 35 years and mixes amateur (91.6 % of physical observations,
   81.7 % of parquet rows) and professional observations.
2. The astronomical community actively debates whether heterogeneous amateur
   time series can support learned representations. BESS-Bench provides the
   first benchmark to study this question at scale.
3. Temporal modelling of stellar spectra is reported in several recent papers
   but evaluation protocols, baselines and splits are inconsistent. BESS-Bench
   ships fixed per-star splits, classical baselines (persistence, PCA+Ridge)
   and a benchmark protocol to curb unreported variance.

### Who created the dataset and on behalf of which entity?
The dataset was curated by the authors of the accompanying paper (identity
withheld for double-blind review). The underlying spectra were contributed
by the BeSS (Be Star Spectra) community, a collaboration between amateur
and professional observers coordinated by LESIA/Observatoire de Paris.

### Who funded the creation of the dataset?
The curation work (harvesting, cleaning, release engineering) was funded by
the authors' institutional compute budget. The original spectra were
collected by volunteer observers and professional facilities at their own
cost.

---

## 2. Composition

### What do the instances represent?
Each instance is **one stellar spectrum** (wavelength, flux, optional flux
error) together with observation metadata (date, instrument, observer type,
SNR, site coordinates, etc.). A spectrum is a one-dimensional array sampled
on a native wavelength grid that is specific to each observation.
BESS-Bench redistributes the **full native wavelength coverage** reported by
each observer (see "Line coverage" below); v1.0 downstream tasks are
centred on Hα for coverage reasons but multi-line analyses are explicitly
supported by the release.

### Line coverage across the full spectral range
Although v1.0 baselined tasks focus on Hα and Hβ, the release is
deliberately full-range and each spectrum retains its native `wavelength`
array. Coverage is reported at two granularities (audit script and full
JSON in the companion code repository):

**Table A — coverage per physical observation** (grouped by
`(star_name, mjd)`; one échelle session is counted once, with
λ<sub>min</sub>=min and λ<sub>max</sub>=max aggregated across its orders):

| Line        | λ₀ (Å)  | Coverage | Observations (of 37 624) |
|-------------|--------:|---------:|-------------------------:|
| Hα          |  6562.8 | 77.63 %  | 29 207                   |
| Na D        |  5893.0 | 35.98 %  | 13 536                   |
| He I 5876   |  5876.0 | 35.97 %  | 13 532                   |
| Hβ          |  4861.3 | 35.66 %  | 13 416                   |
| Fe II 5169  |  5169.0 | 35.55 %  | 13 376                   |
| Hγ          |  4340.5 | 32.54 %  | 12 242                   |
| Hδ          |  4101.7 | 19.45 %  |  7 317                   |
| O I 7772    |  7772.0 |  4.05 %  |  1 523                   |

**Table B — coverage per parquet row** (per-row unit, used when tasks
iterate over individual parquet rows):

| Line        | λ₀ (Å)  | Coverage | Rows (of 339 115) |
|-------------|--------:|---------:|------------------:|
| Hα          |  6562.8 |  7.94 %  | 26 937            |
| Na D        |  5893.0 |  3.37 %  | 11 413            |
| He I 5876   |  5876.0 |  1.94 %  |  6 570            |
| Hβ          |  4861.3 |  0.78 %  |  2 652            |
| Fe II 5169  |  5169.0 |  0.71 %  |  2 421            |
| Hγ          |  4340.5 |  0.69 %  |  2 337            |
| Hδ          |  4101.7 |  0.66 %  |  2 246            |
| O I 7772    |  7772.0 |  0.41 %  |  1 387            |

The 10× gap between Table A and Table B is structural: 276 639 rows
(81.6 %) are individual échelle orders that each span a narrow
sub-range, so a single observing session covering Hα contributes
26 rows (median) in Table B but a single observation in Table A.
Wavelength distributions at the row level: λ<sub>min</sub> p5/p50/p95 =
3 980 / 5 277 / 7 006 Å; λ<sub>max</sub> p5/p50/p95 = 4 113 / 5 413 / 7 351 Å.

A quantitative cross-line task (**LineTransfer**, Hβ → Hα) is
baselined on the 13 416 Hβ-covering observations (2 652 rows). Lines
with high observation-level coverage but low row-level coverage
(He I, Na D, Fe II, at ~36 % of observations each) are released
as-is but not benchmarked in v1.0.

### How many instances are there?
BESS-Bench has three nested units of analysis:

- **339 115 parquet rows** (file-level atom). One row = one 1D spectrum
  or one échelle order.
- **37 624 physical observations** (astrophysical unit). One observation
  = one `(star_name, mjd)` group ≡ one observing session. 27 707 are
  single-range spectra, 9 917 are échelle sessions each spanning a
  median of 26 orders (p5/p95 = 19 / 40 orders).
- **1 468 unique Be stars** (canonical names, de-duplicated from 4 755
  raw identifiers via spatial + textual cross-matching). This is the
  split unit (per-star train/val/test).
- Temporal coverage: **1990-03 → 2025-12** (35 years); median cadence
  per star 5 days, with heavy-tailed gaps.

The v1.0 downstream tasks group by `(star_name, mjd, instrument_setup)`
so échelle sessions contribute as single events, not as 26 parallel
ones.

### Observer types and instrument families
BESS-Bench is deliberately a **community-science dataset**: the vast
majority of observations come from volunteer amateur astronomers using
off-the-shelf small-telescope spectrographs (Shelyak eShel, Alpy 600,
LISA, Lhires III). Professional observations — mostly from small
research facilities contributing to BeSS — act as a minority reference
stream.

**Observer type** (from the `observer_type` metadata column,
computed by the parser from FITS-header affiliation strings):

| observer_type  | Parquet rows | %       | Physical observations | %       |
|----------------|-------------:|--------:|----------------------:|--------:|
| amateur        |      276 954 | 81.67 % |                34 456 | 91.58 % |
| professional   |       60 169 | 17.74 % |                 1 744 |  4.63 % |
| unknown        |        1 992 |  0.59 % |                 1 424 |  3.78 % |
| **Total**      |  **339 115** | 100 %   |            **37 624** | 100 %   |

The row-vs-observation gap is again driven by \u00e9chelle usage: most
professional observations in BeSS are \u00e9chelle (~34 orders per session
on average) while amateur observations are a mix of \u00e9chelle and
single-range instruments.

**Instrument families** (inferred from `is_echelle_order` and the
`spectral_resolution` column, since BeSS does not publish a single
normalised `instrument_model` field \u2014 see *Is any information missing?*):

| Family                              | Rows    | Physical obs. | Unique stars | Typical hardware            |
|-------------------------------------|--------:|--------------:|-------------:|-----------------------------|
| \u00c9chelle (R \u2248 10 000 \u2013 50 000)       | 276 639 |         9 917 |          915 | Shelyak eShel, prof. \u00e9chelles|
| Single-range, R \u2265 8 000             |  51 516 |        16 749 |        n.a.  | High-res single              |
| Single-range, 3 000 \u2264 R < 8 000     |   5 127 |         5 125 |        n.a.  | Lhires III high-res          |
| Single-range, R unknown             |   3 340 |         3 340 |        n.a.  | Header missing field         |
| Single-range, R < 1 000             |   1 858 |         1 858 |        n.a.  | LISA, Alpy 600               |
| Single-range, 1 000 \u2264 R < 3 000     |     635 |           635 |        n.a.  | Lhires III low-/mid-res      |

(Reproducible via the coverage-audit script in the companion code
repository, field `observer_breakdown.by_resolution_family` of the
output JSON.)

The instrument family labels are **proxies**, not ground truth: they
are derived from the `spectral_resolution` header value, which may be
theoretical (from the instrument manual) rather than measured. The
`spectral_resolution_measured` column, when populated, should be
preferred. We do not attempt a canonical `instrument_model` field in
v1.0 because the free-text header values in BeSS mix instrument,
telescope and observer conventions in non-trivial ways.

### Does the dataset contain all possible instances?
No. BESS-Bench is a curated snapshot of the BeSS database harvested through
its public Virtual-Observatory interface. Spectra failing one of the
automated quality filters (SNR < 10, coordinates invalid, star unresolved,
corrupted FITS) were excluded. Stars with fewer than two usable spectra
were retained for baseline sanity checks but flagged in `star_list.csv`.

### What data does each instance consist of?
Each row contains 34 columns split into five groups (see the Dataset Card
for full schema):

- **Arrays**: `wavelength`, `flux`, `flux_error`
- **Identity**: `spectrum_id`, `star_name`, `star_name_raw`, `ra`, `dec`
- **Spectral metadata**: `lambda_min`, `lambda_max`, `n_pixels`,
  `spectral_resolution`, `spectral_resolution_measured`
- **Quality**: `snr`, `snr_continuum`, `snr_quality`
- **Temporal**: `observation_date`, `mjd`, `exposure_time`,
  `temporal_quality`
- **Instrument & context**: `spectrograph`, `telescope`, `detector`,
  `instrument_setup`, `instrument_confidence`, `observer_type`,
  `site_latitude`, `site_longitude`, `site_elevation`, `helio_velocity`,
  `bss_rqvh`, `telluric_corrected`, `normalized`, `is_echelle_order`,
  `fits_format`

**Wavelength frame convention.** The `wavelength` array is released
**as in the BeSS FITS** (no shift applied), so users can reproduce any
rest frame they wish. BeSS stores two related keywords:

- `helio_velocity` (= FITS `BSS_VHEL`, km s⁻¹) is the heliocentric
  correction *already applied* by the observer to the FITS wavelength
  solution. It is `0` for **97.2 %** of spectra (BeSS guidelines
  recommend shipping uncorrected wavelengths).
- `bss_rqvh` (= FITS `BSS_RQVH`, km s⁻¹) is the residual heliocentric
  correction *that still needs to be applied* to move the spectrum
  into the heliocentric frame. It is non-zero for **99.8 %** of
  spectra (range ±20 km s⁻¹, 1σ ≈ 10 km s⁻¹).

In the vast majority of cases (≈ 97 %), the released wavelength grid
is therefore **topocentric**. To shift to the heliocentric frame,
apply

    λ_helio = λ_obs · (1 − bss_rqvh / c)

with *c* = 299 792.458 km s⁻¹. For the 0.2 % of spectra with
missing `bss_rqvh` the correction is skipped (treated as already
heliocentric).

All BESS-Bench v1.0 reference baselines (encoder pre-training, SpecProbe/LineTransfer/EWForecast
tasks) **apply this correction internally** before the 128-bin Hα/Hβ
crop and interpolation, so that features and embeddings live in a
common heliocentric frame compatible with external surveys (SDSS,
LAMOST, APOGEE). The exact code path is shared between
[`stage1_encoder/dataset.py`](stage1_encoder/dataset.py) and
[`downstream/compute_spectral_features.py`](downstream/compute_spectral_features.py).

### Is there a label or target associated with each instance?
There is no single canonical target. BESS-Bench defines **three
benchmarked quantitative tasks** with task-specific labels computed
on-the-fly from the spectra:

- **SpecProbe — Hα single-line spectral-feature regression**. Ridge probes
  (5-fold CV, 3 encoder seeds) on frozen 128-D encoder embeddings vs.
  PCA(10)/PCA(50) of the raw normalised Hα flux. Targets: EW, FWHM,
  central depth, Δv, vr_ratio, peak intensity. EW, FWHM, central
  depth and peak intensity are defined for all 26 858 scored Hα
  spectra; Δv and vr_ratio are defined only on the ~10 800 cleanly
  double-peaked profiles (NaN otherwise — see Appendix
  *External validation of spectral features* of the companion paper
  for the exact rules).
- **LineTransfer — Cross-line probing (Hβ → Hα features)**. 2 525 (star, MJD)
  pairs covering both lines. Ridge on Hβ flux / Hβ PCA(10)/PCA(50)
  → Hα features. Demonstrates that the full-range release carries
  predictive signal outside the Hα window.
- **EWForecast — Short-horizon EW(Hα) forecasting**. One-step-ahead forecast
  under a persistence baseline; 4 seeds for the bundled learned
  transformer and bootstrap CI95 (10 000 resamples) on per-prediction
  residuals for the PCA+Ridge / zero-shot TS-FM baselines.

Labels and splits are released alongside the dataset in
`dataset/star_list.csv` and `dataset/splits.csv`. All label-computation
scripts are version-controlled in the companion code repository.

### Is any information missing?
Yes. Missingness is deliberately documented per column and is a research
signal:

- `flux_error` is null for a substantial fraction of the dataset (amateur
  FITS often lack per-pixel uncertainties). Users must handle missing
  uncertainties explicitly.
- `spectral_resolution_measured` is null when the FITS header does not
  report slit width and the resolution cannot be recovered from metadata.
- `site_latitude`/`site_longitude`/`site_elevation` are null for
  ~9% of spectra when the observer did not provide geographic metadata.
- `spectral_resolution` (theoretical) falls back to a per-spectrograph
  median if the header value is missing.

### Are relationships between instances made explicit?
Yes. Two keys support relational use:

- `star_name` links all spectra of a given star. This is the canonical
  grouping used for the temporal tasks and for the split assignment.
- `spectrum_id` (MD5 of (star_name, mjd, instrument_setup)) is unique per
  exposure. Échelle orders of the same exposure share the same MJD and
  `instrument_setup` but have distinct `spectrum_id`s.

### Are there recommended data splits?
Yes. BESS-Bench releases a **fixed split by star** (not by spectrum) to
prevent temporal leakage:

- train: 1 024 stars / 241 057 spectra (≈71%)
- validation: 219 stars / 44 383 spectra (≈13%)
- test: 225 stars / 53 675 spectra (≈16%)

Split assignment is deterministic: `md5("bess_bench_v1::" + star_name)[:8]`
mod 100 ↦ {0–69: train, 70–84: val, 85–99: test}. The resulting
`dataset/splits.csv` file is bit-identical across runs; the generating
script ships in the companion code repository.

### Are there any errors, sources of noise, or redundancies in the dataset?
Yes, and part of the scientific interest of the benchmark is to study them:

- **Heterogeneous calibration**: 26 spectrographs and 62 telescopes
  contribute, with varying wavelength solutions, flat-field quality,
  continuum normalisation strategies and telluric correction.
- **Échelle redundancy**: 81.6% of raw rows are individual échelle orders.
  They are independent one-dimensional arrays but share the same exposure.
  The `is_echelle_order` flag makes this explicit; downstream tasks group
  by (star_name, mjd, instrument_setup).
- **SNR distribution is bimodal**: a long tail of low-SNR amateur spectra
  coexists with a clean professional core.
- **Name ambiguity**: 4 755 raw object names collapsed to 1 468 unique
  stars after SIMBAD cross-matching; residual 0.3% manual corrections
  are documented in `pipeline/03_cleanup/`.

### Is the dataset self-contained?
Yes. All spectra and metadata are bundled in the HuggingFace dataset. The
benchmark code (models, baselines, evaluation scripts, pretrained
checkpoints, frozen embeddings) is mirrored in this repository and does
**not** rely on any closed-source asset or time-gated API.

### Does the dataset contain data that might be considered confidential or sensitive?
No. BESS is a public astronomical archive; every published contributor
consented to open sharing of their spectra. The dataset contains no human
subject data in any health, legal, or demographic sense.

However, `site_latitude`, `site_longitude` and `site_elevation` are
derived from observer-reported geographic metadata and could, in
principle, de-anonymise individual amateur observers at the country/town
level. We mitigate this risk by:

1. Not releasing observer names, addresses, or any FITS `OBSERVER`
   keyword.
2. Rounding coordinates to 0.1° (~11 km) and elevation to 100 m in
   the published dataset.
3. Providing a documented opt-out procedure (see §8).

No minors, no biometric data, no personally identifiable information
beyond the coarse coordinates described above.

---

## 3. Collection process

### How was the data acquired?
Spectra were obtained by the BeSS community using heterogeneous
spectrographs on amateur and professional telescopes, then uploaded to the
BeSS central archive at LESIA. **We did not collect or own the original
observations.** Our contribution is to (i) harvest the public
Virtual-Observatory interface of the archive, (ii) clean the metadata,
(iii) standardise the schema, and (iv) release an ML-ready package.

### What mechanisms were used to collect the data?
The raw spectra were harvested via the SSA (Simple Spectrum Access) VO
protocol and via bulk TAR downloads authorised by the archive. Harvesting
scripts are omitted from this release for size and mirror-politeness
reasons but the full procedure is described in the paper
(Methodology / Data Acquisition) and in `pipeline/README.md`. The
cleaning, quality-control and export pipeline **is** included
(`pipeline/02_quality`, `pipeline/03_cleanup`, `pipeline/04_export`).

### If the dataset is a sample, what was the sampling strategy?
The dataset is not a sample: it includes every BeSS spectrum whose FITS
file passed automated quality gates at the harvest cutoff date
(January 2026). Excluded spectra are enumerated in a held-out manifest
for auditing.

### Who was involved in the collection process?
- Original observers: ~400 amateur astronomers and ~30 professional teams
  (see BeSS acknowledgements).
- Archive maintainers: LESIA staff (Neiner et al. 2011 and subsequent
  maintenance).
- Curation: authors of this paper.

### Over what timeframe was the data collected?
- Spectra: 1990-03-11 → 2025-12-30 (35 years).
- Curation and release engineering: 2024-09 → 2026-04.

### Were any ethical review processes conducted?
No formal IRB review was conducted because the dataset does not contain
human subject data. An internal review assessed risks to amateur
contributors (geographic privacy) and informed the coordinate rounding
decision.

---

## 4. Preprocessing / cleaning / labelling

### Was any preprocessing / cleaning done?
Yes, extensively. The pipeline comprises three stages:

1. **Quality gate** (`pipeline/02_quality/`): SNR estimation via DER_SNR
   (Stoehr 2008), continuum-window SNR estimation, sigma-clipping of
   manifest corruptions, MJD ↔ ISO-date cross-validation.
2. **Cleanup** (`pipeline/03_cleanup/`): regex-based spectrograph and
   telescope normalisation, coordinate validation, observer-type inference
   from spectrograph class, star name spatial+textual deduplication,
   échelle order flagging.
3. **Export** (`pipeline/04_export/`): DuckDB → HuggingFace `Dataset`,
   shard writing, schema typing, dataset card rendering.

Every step is deterministic and every script is version-controlled.
Note that the **wavelength grid is exported untouched** (topocentric
for ≈97 % of spectra); the heliocentric correction is applied
*on-the-fly* by the encoder pre-training and downstream task code
using the `bss_rqvh` metadata column (see *Wavelength frame
convention* above). This preserves the option for users to choose a
different rest frame (barycentric, LSR, stellar systemic) without
having to undo a baked-in shift.

### Was the raw data saved in addition to the cleaned data?
The raw FITS headers remain on the BeSS archive and are publicly
retrievable by any user with a BeSS account. We do not redistribute raw
FITS to avoid mirroring the archive; interested researchers can recreate
the raw ingest step from our `pipeline/02_quality/` scripts.

### Is the cleaning software available?
Yes: `pipeline/02_quality`, `pipeline/03_cleanup`, `pipeline/04_export`
in this repository. Tested on Python 3.10+ with the versions listed in
`requirements.txt`.

---

## 5. Uses

### Has the dataset been used for any tasks already?
Yes, in the accompanying paper:

- Self-supervised Hα-window representation learning (masked
  auto-encoding, $R^2 = 0.93$ on held-out stars).
- Probing of spectral features from frozen embeddings (SpecProbe, all six
  features).
- Cross-line probing Hβ → Hα features (LineTransfer, 2 525 paired spectra).
- Short-horizon temporal EW(Hα) forecasting against persistence,
  PCA+Ridge, and zero-shot time-series foundation models
  (Chronos-Bolt Small/Base, TimesFM-2.0-500M). **PCA(10)+Ridge
  (ctx=5 and ctx=10), Chronos-Bolt Base and TimesFM-2.0 beat
  persistence at bootstrap CI95 level**, while Chronos-Bolt Small
  and PCA(50)+Ridge ctx=5 remain inconclusive (CI95 straddles 1.0)
  and PCA(50)+Ridge ctx=10 fails (CI95 strictly above 1.0); the
  bundled learned transformer also fails on every seed — a
  calibration useful to the community.

### What other tasks could the dataset be used for?
Radial-velocity estimation, disk dynamics, shell episode detection,
binarity detection, low-resolution spectral super-resolution, cross-instrument
normalisation, calibration transfer between amateur and professional
spectrographs, uncertainty modelling under heteroscedastic labels.

### Is there anything about the composition of the dataset that could impact future uses?
Yes:

- The Be-star population is overrepresented at declination δ > -30°
  because the active BeSS contributor base is concentrated in the
  northern hemisphere. Models should not be extrapolated to southern
  sources without recalibration.
- Amateur coverage is skewed toward brighter (V < 8) stars. Faint-star
  performance is not characterised.
- The temporal cadence is quasi-nightly for a small set of very active
  Be stars (e.g. ζ Tau, γ Cas, δ Sco) and months-apart for most of the
  catalogue. This imbalance must be accounted for in any temporal
  modelling claim.
- All labels are derived from the spectra themselves. No external
  photometric or interferometric labels are bundled.

### Are there tasks for which the dataset should not be used?
- The dataset **must not** be used to train or benchmark stellar classification
  for spectral types outside the late-O / early-B range, for which it
  is not representative.
- It **must not** be used to attempt re-identification of individual
  amateur observers from site coordinates.

---

## 6. Distribution

### Will the dataset be distributed to third parties?
Yes. Public HuggingFace dataset release (URL withheld for double-blind
review; a time-locked, anonymous mirror is provided for reviewers).

### How will it be distributed?
HuggingFace Hub, in the Parquet + `datasets` standard layout
(21 shards, 9.5 GB total). A small inspection sample (~500 MB,
~7% of the stars stratified by split) can be reproducibly built from the
full release with `python stats/build_sample.py --target-mb 500`,
following the NeurIPS D&B Track guidance on lightweight inspection.

### When will it be distributed?
Target: upon paper acceptance. An anonymous reviewer-only mirror is
accessible during the review period.

### Will the dataset be distributed under a copyright or other IP license?
CC-BY-4.0, with attribution to the BeSS database (Neiner et al. 2011)
and the BeSS consortium at LESIA / Observatoire de Paris.

### Have any third parties imposed IP-based or other restrictions?
No. BeSS spectra are already distributed publicly under a policy
compatible with CC-BY-4.0. Our redistribution does not introduce
additional restrictions.

### Do any export controls or regulatory restrictions apply?
No.

---

## 7. Maintenance

### Who will support / host / maintain the dataset?
The authors will host the dataset on HuggingFace Hub and maintain the
`bess-bench` GitHub repository under the institutional organisation that
accepts the paper. Contact information will be added upon un-blinding.

### How can the owner be contacted?
A double-blind-compatible contact email will be disclosed upon
acceptance.

### Is there an erratum?
An `ERRATA.md` file will be opened upon release to track any bug,
relabelling or coordinate correction.

### Will the dataset be updated?
Yes. A v1.x minor release is planned once per year to integrate newly
harvested spectra and any bug fixes. Versions are pinned by Git tag and
HuggingFace revision. Evaluation scripts are stable across v1.x.

### If others want to contribute, is there a mechanism for them to do so?
Yes. GitHub pull requests to the `bess-bench` repository (post-acceptance)
will be reviewed. New gold labels, new downstream tasks and new baseline
scripts are encouraged; changes to the core pipeline or to existing splits
will be rejected to preserve benchmark comparability.

---

## 8. Responsible AI considerations (Croissant RAI fields)

### Data collection consent
Original observers uploaded their spectra voluntarily to a public archive
with open-distribution terms. We do not redistribute any observer name or
FITS `OBSERVER` keyword. Observers who request removal of their spectra
may contact the BeSS archive (for the original data) or the authors (for
any trace in the published dataset). Requests are honoured within 30 days
in the next minor release.

### Personal information
No PII is released. Geographic coordinates are rounded to 0.1° (~11 km)
and elevation to 100 m.

### Sensitive attributes
None.

### Known biases
- Geographic: northern-hemisphere dominance.
- Magnitude: bias toward bright (V < 8) Be stars.
- Instrument: heavy representation of a handful of popular amateur
  spectrographs (LHIRES III, eShel).
- Target selection: famous variable Be stars are heavily oversampled.

These biases are quantified in the paper (Supplementary Table S2).

### Risks of misuse
Re-identification of individual observers from site coordinates is the
primary residual risk. Mitigation: coordinate rounding, no observer names,
documented opt-out path.

### Human labour
All labels released in this version were produced by the authors
(automatic computation for SpecProbe/LineTransfer/EWForecast from spectral measurements). No
manual classification or crowdworker effort is involved in v1.0.

### Safety and security review
The dataset contains no code, model or payload that could be used to
attack systems. No executable content is embedded in the Parquet shards.

### Environmental footprint
The full training of the released $803$\,k-parameter Hα encoder takes
≈0.5 GPU-hour on a single NVIDIA A100 80 GB (80 epochs, mask-ratio 0.6);
with three seeds (42/123/456) the full encoder budget is ≈1.5 GPU-hours.
(The full BeMAE auto-encoder, including the decoder used only at
pre-training time, has ≈912\,k parameters.) The EWForecast TS-FM
baselines are **zero-shot** (no training): Chronos-Bolt
Small (48 M), Chronos-Bolt Base (205 M) and TimesFM-2.0-500 M together
take ≈10 GPU-minutes of inference on a single A100. PCA+Ridge is
CPU-only (~1 CPU-minute). Total release training budget: ~2 GPU-hours.
Harvest and preprocessing are CPU-bound (~6 CPU-days for a full
re-ingest).
