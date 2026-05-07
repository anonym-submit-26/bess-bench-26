---
license: cc-by-4.0
language:
  - en
size_categories:
  - 1K<n<10K
task_categories:
  - feature-extraction
  - time-series-forecasting
tags:
  - astronomy
  - spectroscopy
  - be-stars
  - h-alpha
  - sample
---

# BESS-Bench v1.0 — Reviewer sample (~1%)

This is a **stratified sample** of the full BESS-Bench v1.0 dataset
[`anonym-submit-26/bess-bench-26`](https://huggingface.co/datasets/anonym-submit-26/bess-bench-26),
provided so that NeurIPS 2026 reviewers can quickly inspect data quality
without downloading the 9.5 GB full release.

## How the sample was built

- **Source.** The full release loaded as a single `train` split
  (339,115 spectra × 35 columns).
- **Star-level splits.** The frozen star-grouped partition
  (`splits.csv`, 1,024 train / 219 val / 225 test stars) is used to
  preserve the three-way structure.
- **Sampling rule (seed 42).** For each of the three splits, draw
  **50 stars uniformly at random**; for every selected star, retain at
  most **20 spectra**. Union over the three splits.
- **Result.** ~3,000 spectra (≈1 % of the corpus), spanning 150 stars,
  with the original `train` / `validation` / `test` membership preserved
  in an additional `split` column.

The schema, dtypes, column semantics and metadata conventions are
**identical** to the full release; only the rows are subsampled. No
spectrum is altered.

## Reproducing the sample

The build script is in the anonymous code repository at
`anon_bess_files/scripts/build_review_sample.py`. With the canonical
HuggingFace dataset loaded locally:

```bash
python anon_bess_files/scripts/build_review_sample.py
```

## Differences with the full release

| Property | Full release | This sample |
|---|---|---|
| Repo | `anonym-submit-26/bess-bench-26` | `anonym-submit-26/bess-bench-26-sample` |
| Spectra | 339,115 | ~3,000 |
| Stars | 1,468 | 150 (50 per split) |
| Per-star cap | none | 20 spectra |
| `split` column | absent (split via `splits.csv`) | present |
| Schema | 35 columns | 35 + `split` |

## License

CC-BY-4.0, same as the full release. See the parent dataset card for
attribution and citation guidance.
