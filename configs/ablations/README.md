# BeMAE-Hα ablations (v1.0)

Each YAML below specifies **one variant** of the BeMAE-Hα encoder.
The fields are CLI overrides applied verbatim to `stage1_encoder/train.py`
and `stage1_encoder/precompute_embeddings.py` (the latter inherits everything
from the checkpoint, so no override is needed for embeddings).

Default ("baseline") values (see `stage1_encoder/config.py`):
- `mask_ratio: 0.60`
- `n_layers: 4`
- `d_model: 128`
- `patch_size: 8` (with `patch_overlap=4`, i.e. step=4 → 31 patches)
- `batch_size: 256`
- `epochs: 80`
- `lr: 1e-4`

We ablate **4 axes × 2 off-baseline values × 3 seeds = 24 trainings**:

| Axis        | Variants            | Files                                     |
|-------------|---------------------|-------------------------------------------|
| `mask_ratio`| {0.30, 0.75}        | `mask_ratio_030.yaml`, `mask_ratio_075.yaml` |
| `n_layers`  | {2, 6}              | `n_layers_2.yaml`, `n_layers_6.yaml`         |
| `d_model`   | {64, 256}           | `d_model_64.yaml`, `d_model_256.yaml`        |
| `patch_size`| {4, 16}             | `patch_size_4.yaml`, `patch_size_16.yaml`    |

The baseline (mask_ratio=0.60, n_layers=4, d_model=128, patch_size=8) reuses
the existing 3 seeds in `stage1_encoder/runs/Halpha_all_seed{42,123,456}/`,
so we never re-train it. Total Phase B training budget = 24 GPU-runs
(~30 min/run on A100 80GB ≈ 12 GPU-hours).

For probing, only `{fwhm, central_depth, delta_v}` are evaluated (the three
non-trivial shape features that drive the headline gain). EW and peak_intensity
are saturated for all encoders; vr_ratio is reported for the headline table
but not part of the ablation sensitivity analysis.

See `scripts/cluster/00b_ablations_pretrain.slurm`, `scripts/cluster/02c_ablations_probe.slurm`
and `stats/aggregate_ablations.py`.
