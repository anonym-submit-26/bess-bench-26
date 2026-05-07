# stats/ — Statistical aggregation and confidence intervals

Scripts that aggregate the raw downstream JSON outputs, compute confidence
intervals, and emit the final tables and figures used in the paper.

## Pipeline

```bash
# 1. Raw aggregation (multi-seed mean ± std, ablation grid) → summary JSONs
python stats/aggregate_results.py
python stats/aggregate_ablations.py

# 2. Confidence intervals
python stats/bootstrap_ci.py        # T1 Gaussian CV CI95 + headline ratio bootstrap
python stats/bootstrap_ci_t3.py     # T3 EWForecast bootstrap CI95 (n = 10 000)

# 3. Paper figures
python stats/figure_multiseed.py    # T1 / ablation bar charts
python stats/figure_t3_horizon.py   # ratio_mae stratified by Δt
```

## Key output files

| Path | Content |
|---|---|
| `reference_results/stats/t1_multiseed_summary.json` | T1 SpecProbe per-feature R² over seeds {42, 123, 456} |
| `reference_results/stats/ablations_summary.json`    | T1 with axis ablations (mask_ratio / n_layers / d_model / patch_size) and headline 4.24× ratio |
| `reference_results/stats/t3_bootstrap_cis.json`     | T3 EWForecast bootstrap CI95 for PCA(K)+Ridge configs and TS-FMs |
| `paper/tables/*.tex`                      | LaTeX tables consumed by the paper |
| `figures/*.pdf`                           | Paper figures |

## Methodological choices

### T1 SpecProbe — Gaussian CV CI95
Each seed reports R² per feature averaged over 5 GroupKFold folds
(`star_id` as group). Cross-seed mean ± std is the headline; per-feature
Gaussian CI95 = `mean ± 1.96 · std / √k` (k = 5 folds) is used in
the paper to mark a method as *strictly* better than another when the
two CI95 are disjoint.

### T1 headline aggregated ratio
The headline test is the bootstrap CI95 of
`R²(z) / R²(PCA(10))` aggregated over the three non-trivial shape
features (`fwhm`, `central_depth`, `delta_v`), seed-level resampling
(n = 3 seeds, 10 000 replicates, 2.5/97.5 percentiles). v1.0 result:
**4.24× [4.12, 4.33]**.

### T3 EWForecast — bootstrap CI95
Non-parametric bootstrap on the per-prediction absolute errors
(n = 10 000 resamples). A learned baseline strictly beats persistence
when the CI95 lies entirely below 1. v1.0 best result: PCA(10)+Ridge
ctx=5 → ratio_mae 0.942 with CI95 [0.927, 0.957].

### Tests deliberately not performed
- **Paired Wilcoxon / McNemar / permutation tests.** The reference
  evaluation scripts now serialise per-prediction `y_true / y_pred`
  arrays in the T3 outputs, but the paper's claim rule is built on
  the bootstrap CI95 already; paired non-parametric tests are
  planned as additional diagnostics for v1.1.
- **Phase classification (T4).** Deferred to v1.1 (labels not yet
  finalised; see `EVALUATION_PROTOCOL.md` §4).

## Reproducibility

All scripts are deterministic (random seeds fixed in each script).
Outputs are bit-identical across runs on the same input JSONs.
