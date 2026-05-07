# BESS-Bench — Evaluation Protocol (v1.0)

This document is the **authoritative specification** for every number
reported on BESS-Bench. Any reproduction of results must follow it
*to the letter*. Numbers obtained with a different protocol are not
comparable.

The protocol is deliberately conservative: the dataset defines exactly
**three tasks** — SpecProbe, LineTransfer, and EWForecast — each
with a single primary metric and fixed reference baselines reported
under a unified statistical protocol (three random seeds, bootstrap CI₉₅).

---

## 0. Global rules (apply to every task)

1. **Data source.** Load the dataset exclusively from the released
   Hugging Face snapshot
   (`datasets.load_dataset("anonym-submit-26/bess-bench-26")`,
   `revision=<frozen tag>`). Do not mix in private scrapes of BeSS.
2. **Splits.** Use the `splits.csv` bundled with the release verbatim.
   Split columns: `split ∈ {train, val, test}`. Splits are defined
   *by star* for SpecProbe/LineTransfer and *by MJD cutoff* for
   EWForecast.
3. **Test contamination.** No test spectrum, no test star, no test
   epoch may appear during pre-training, fine-tuning, hyper-parameter
   selection, or early stopping.
4. **Seed.** Every randomised component (train sampler, init, dropout,
   fold assignments, bootstrap) must use an explicit integer seed.
   The split assignment is fixed (seed `42`). Multi-seed runs use
   `{42, 123, 456}` unless stated otherwise.
5. **Reporting.** Every reported result must serialise to a JSON file
   that validates against the schema in §4 (enforced by
   `stats/aggregate_results.py::summary_schema`). Missing tasks are
   acceptable; silently changing a metric definition is not.
6. **Compute budget disclosure.** Report GPU-hours, GPU model, peak
   VRAM, and wall-clock time per task. We do not cap compute but we
   make it visible.

---

## 1. SpecProbe — Hα single-line spectral-feature regression (Ridge probe)

### 1.1. Task
Given a frozen embedding `z ∈ ℝᵈ` of a single spectrum, predict six
spectral features of the Hα line computed by
`downstream/compute_spectral_features.py`:

| Feature          | Unit | Coverage  | Notes                                   |
|------------------|------|-----------|-----------------------------------------|
| `ew`             | Å    | 26 858 sp | Pseudo-equivalent width                 |
| `fwhm`           | Å    | 26 858 sp | Full width at half maximum              |
| `central_depth`  | —    | 26 858 sp | 1 − min(normalised flux)                |
| `delta_v`        | km/s | 10 804 sp | Doppler shift, double-peaked only       |
| `vr_ratio`       | —    | 10 804 sp | V/R peak ratio, double-peaked only      |
| `peak_intensity` | —    | 26 858 sp | Max(normalised flux) − 1                |

### 1.2. Protocol
- **Probe.** `sklearn.linear_model.RidgeCV` (α grid: 20 log-spaced
  values in `[1e-2, 1e3]`) fitted on standardised `z`. No feature
  engineering on top of `z`.
- **Evaluation.** 5-fold `GroupKFold` on `star_id` (no star shared
  between train and test of any fold).
- **Multi-seed.** Train the encoder with seeds `{42, 123, 456}` and
  report `R²_mean ± R²_std` over seeds (the encoder seed is the unit
  of variability; CV folds are pooled within a seed).
- **Primary metric.** `R²_mean` per feature. Secondary: `R²_std` over
  seeds.
- **Reference baselines** (bundled):
  - `PCA(10)` of normalised flux.
  - `PCA(50)` of normalised flux.
- **Headline aggregated test.** Bootstrap CI₉₅ of the ratio
  `R²(z) / R²(PCA(10))` aggregated over the three shape features
  (`fwhm`, `central_depth`, `delta_v`) by seed-level resampling
  (n = 3 seeds, 10 000 replicates, 2.5/97.5 percentiles). Aggregating
  rather than reporting three independent comparisons controls the
  family-wise error rate (≈14 % at α = 0.05).
- **Per-feature significance rule.** A submission *strictly beats*
  PCA(10) on feature `f` when the Gaussian 95 % CIs `μ ± 1.96·σ/√k`
  (k=5 folds) are **disjoint** (see
  `stats/bootstrap_ci.py::t1_probe_ci`). Overlapping CIs count as a
  tie.

### 1.3. Reporting
- One `R²_mean ± CI95` per feature per method.
- Emit `ci_results.json["specprobe"]` as defined by `stats/bootstrap_ci.py`.

---

## 2. LineTransfer — Cross-line Hβ → Hα probing

### 2.1. Task
For every same-night `(star, MJD)` pair that covers both Hα and Hβ within
±50 Å (n = 2 525 pairs over 1 094 stars), predict the six Hα features
(§1.1) from the **Hβ window flux** alone — testing whether the encoder
generalises beyond the line it was supervised on.

### 2.2. Protocol
- **Input.** 128-bin normalised flux on the Hβ ±50 Å window emitted by
  `downstream/compute_hbeta_features.py`.
- **Probe.** `sklearn.linear_model.Ridge(alpha=1.0)` from Hβ flux to
  each Hα target (one Ridge per target).
- **Evaluation.** 5-fold `GroupKFold` on `star_id`, same multi-seed
  setting as SpecProbe.
- **Primary metric.** `R²_mean(ew)` (most physically meaningful: Hα
  and Hβ EWs are correlated through the underlying disk emission,
  v1.0 reaches ≈0.69 on PCA(10) of Hβ). Secondary: per-feature
  `R²_mean ± std` for `{fwhm, central_depth, delta_v, vr_ratio,
  peak_intensity}` (line-shape features do not transfer reliably,
  R² ≲ 0.05 in v1.0).
- **Reference baseline.** PCA(10) on Hβ flux + Ridge to Hα targets,
  same protocol.

### 2.3. Reporting
Emit `summary.json["linetransfer"]` matching
`reference_results/downstream/probe_t2_hbeta_to_halpha.json` schema.

---

## 3. EWForecast — One-step EW(Hα) forecasting

This is the **headline temporal protocol** of the benchmark.

### 3.1. Task
For each of the **194 stars with ≥ 30 Hα epochs** that pass the SNR ≥ 10
quality gate, form rolling context windows of length `L ∈ {5, 10}` past
epochs ending at `t_k` and predict `EW(t_{k+1})` at the next available
epoch. Time unit: days. The reference implementation enforces a temporal
split at `MJD = 59215.0` (train < cutoff, test ≥ cutoff) so that no test
epoch is ever used in PCA fitting or Ridge fitting.

### 3.2. Protocol
- **Persistence baseline** (canonical):
  `ŷ_{k+1} = ew(t_k)`. Used as the denominator of the primary metric.
- **Reference learned baseline.** `PCA+Ridge` over context windows —
  `downstream/t3_pca_ridge_temporal_baseline.py`. Four configurations:
  `{PCA(10), PCA(50)} × {ctx=5, ctx=10}`. Best v1.0:
  PCA(10)·ctx=5 → ratio 0.942.
- **Reference zero-shot baselines.** Chronos-Bolt-base (205 M) and
  TimesFM-2.0-500 M run zero-shot —
  `downstream/t3_ts_fm_baselines.py`.
- **Primary metric.**
  `ratio_mae = mae_model / mae_persistence`
  on the pooled test set (all stars, all horizons).
  Smaller is better; `ratio_mae < 1` is the minimum bar to claim any
  improvement over persistence.
- **Bootstrap CI95.** Non-parametric bootstrap on the per-epoch absolute
  errors (n = 10 000 resamples, 2.5/97.5 percentiles); see
  `stats/bootstrap_ci_t3.py`.
- **Claim rule.** A published claim of beating persistence requires
  the bootstrap CI95 to lie **strictly below 1** (`upper < 1`). A point estimate below 1 with a CI crossing 1 is a
  tie.

### 3.3. Secondary diagnostics
Emit `by_horizon` R² and MAE for the buckets
`{<7d, 7–30d, 30–90d, 90–365d, >365d}` (matches
`t3_pca_ridge_temporal_baseline.json::by_horizon`). v1.0 finding:
baselines beat persistence on `Δt ∈ [7, 365] d` and collapse to
ratio ≥ 1 beyond 365 d.

### 3.4. Reporting
One row per configuration plus a bootstrap summary. Emit
`summary.json["t3"]` as defined by
`stats/aggregate_results.py::table_t3_baselines`.

---

## 4. What is **not** part of the benchmark (and why)

- **Wilcoxon / McNemar / permutation tests.** The reference evaluation
  scripts do not serialise raw per-spectrum predictions; paired tests
  are planned for a future version once `--save-predictions` ships.
- **Change-point detection.** Out of scope for v1.0: stable expert
  annotations of outburst / disk-loss events are not yet released.
- **Phase classification.** A fourth task (phase classification on
  Hα morphological classes) is **deferred to v1.1**: labels are not
  released and no v1.0 baseline is scored.
- **Zero-shot / in-context evaluation of foundation LMs.** Out of
  scope for v1.0 because SpecProbe and LineTransfer are pure probing
  tasks; this can change in later revisions.
- **Pre-training metrics** (reconstruction loss, contrastive score).
  Pre-training is not a task; v1.0 reports SpecProbe, LineTransfer and
  EWForecast.

---

## 5. Versioning

This document is `v1.0`. Any breaking change to a metric, split, or
significance rule bumps the major version and invalidates prior
results. Patch versions (`v1.0.x`) may clarify wording but never
change numbers.
