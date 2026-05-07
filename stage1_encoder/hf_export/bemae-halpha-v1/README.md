---
library_name: pytorch
license: cc-by-4.0
tags:
- astronomy
- spectroscopy
- be-stars
- self-supervised
- masked-autoencoder
- time-series
datasets:
- anonym-submit-26/bess-bench-26
---

# BeMAE-Hα — Masked Autoencoder Encoder for Be-Star Hα Spectra

`anonym-submit-26/bemae-halpha-v1` is the **pretrained encoder** of the BeMAE-Hα model
introduced in *"BESS-Bench: Benchmarking Spectral Representations for
Be-Star Variability"* (NeurIPS 2026 D&B Track, under review).

It is a small (≈803\,k params) Transformer encoder that maps a 128-bin spectrum
cropped around the Hα line (6562.8 ± 50 Å) to a 128-dimensional embedding
`z_halpha`. The encoder was trained with a Masked Autoencoder (MAE) objective
on the Hα slice of BESS-Bench (≈26.9\,k spectra over 1 468 Be stars,
1990–2025).

## Quick start

```python
from huggingface_hub import snapshot_download
import sys, json, torch

ckpt = snapshot_download("anonym-submit-26/bemae-halpha-v1")
sys.path.insert(0, ckpt)
from model import SpectralEncoderHalpha, ModelConfig  # noqa: E402

with open(f"{ckpt}/config.json") as f:
    cfg_dict = json.load(f)["model_config"]
cfg = ModelConfig(**cfg_dict)

encoder = SpectralEncoderHalpha(cfg)
encoder.load_state_dict(torch.load(f"{ckpt}/pytorch_model.bin", map_location="cpu"))
encoder.eval()

# flux, wavelengths, validity : torch.Tensor of shape [B, 128]
# Spectra must be cropped to Hα ± 50 Å (6512.8–6612.8 Å) and normalized
# to the pseudo-continuum at the window edges (see `example_usage.py`).
with torch.no_grad():
    z_halpha, *_ = encoder(flux, wavelengths, validity, mask=None)
# z_halpha : Tensor of shape [B, 128]
```

## Intended use

`z_halpha` is a general-purpose embedding of the Hα profile. It has been
evaluated on the three downstream tasks of BESS-Bench v1.0 — SpecProbe
(Hα feature regression), LineTransfer (Hβ → Hα generalisation) and
EWForecast (short-horizon EW(Hα) forecasting); see the paper for details.

Typical uses :
- Clustering and retrieval of similar Hα profiles across the BeSS database.
- Building lightweight regressors on top of `z_halpha` for physical
  parameters (EW, V/R, velocity).
- Time-series features for short-horizon EW(Hα) forecasting (cf. EWForecast).

## Training details

| Item | Value |
|---|---|
| Architecture | Transformer MAE encoder |
| Input | 128 bins, Hα ± 50 Å (6512.8–6612.8 Å) |
| Patches | 8 px, step 4 → 31 patches |
| Embedding dim | 128 |
| Transformer layers / heads | 4 / 4 |
| Parameters | ≈803\,k (encoder only; full BeMAE auto-encoder ≈912\,k) |
| Pretraining objective | Masked Autoencoder, mask ratio 0.60, 3 contiguous blocks |
| Optimizer | AdamW, lr=0.0001, wd=0.05, warmup 5 epochs |
| Epochs | 80 (with early stopping) |
| Batch size | 256 |
| Dataset | `anonym-submit-26/bess-bench-26`, Hα slice (≈26.9\,k spectra) |
| Seed (this checkpoint) | **42** |
| Best val MAE loss | 0.335420 |
| Hardware | 1× NVIDIA A100 80 GB, ~30 min per seed |

## Reproducibility

The v1.0 benchmark reports aggregated results over **3 seeds** (42, 123, 456).
This specific checkpoint corresponds to **seed 42**, which is the
"official" reference run used for all qualitative plots in the paper.
Multi-seed aggregates are available in the companion repository.

To retrain from scratch (~1h30 on an A100) :

```bash
git clone https://github.com/anonym-submit-26/bess-bench-26
cd bess-bench/stage1_encoder
sbatch run_multiseed.slurm   # trains seeds 42, 123, 456 sequentially
```

## Citation

```bibtex
@misc{bess_bench_2026bess,
  title        = {BESS-Bench: Benchmarking Spectral Representations for Be-Star Variability},
  author       = {Anonymous and others},
  year         = {2026},
  note         = {NeurIPS 2026 Datasets and Benchmarks Track}
}
```

## Limitations

- Trained exclusively on BeSS amateur-grade spectra; transfer to professional
  instruments (ESPRESSO, CARMENES) has not been evaluated.
- The input window is fixed at Hα ± 50 Å; wider profiles may be truncated.
- `z_halpha` is not a physical parameter vector; linear probes are required
  for interpretable regression (see SpecProbe and LineTransfer in the paper).
