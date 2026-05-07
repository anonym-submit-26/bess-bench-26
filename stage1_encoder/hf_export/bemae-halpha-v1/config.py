"""
config.py — Configuration for the specialised Hα encoder (Phase A).

Reduced architecture compared to the full-spectrum encoder:
  - Input : 128 bins (crop ±50 Å autour de Hα 6562.8 Å)
  - Patches : 8 pixels, overlap 4 → 31 patches
  - Transformer: 4 layers, d=128, 4 heads
  - Embedding CLS : z_halpha ∈ ℝ^128

Objective : encode the Be physics (EW, V/R, Hα profile) that the
full-spectrum encoder dilutes in the continuum reconstruction.

No GRL: negative result confirmed with the full encoder,
and here we operate on a 100 Å window -> no continuum to contaminate.
"""

from dataclasses import dataclass, field
from typing import Optional


# ── Physical constants ─────────────────────────────────────────────────────

HALPHA_CENTER = 6562.8       # Wavelength of Hα center (Å)
HALPHA_HALF_WINDOW = 50.0    # Half crop window (Å) -> ±50 Å
N_BINS = 128                 # Number of bins after interpolation


@dataclass
class ModelConfig:
    """
    Architecture of the Hα encoder — compact version.

    With 128 bins and patch_size=8, step=4 :
      N_patches = (128 - 8) / 4 + 1 = 31 patches

    The model is ~10× smaller than the full encoder (300K vs 5.3M params).
    Each bin = 0.78 Å, which is sufficient to resolve Hα profiles
    (FWHM typical ~5-15 Å for Be stars).
    """
    # ── Encoder ──
    d_model: int = 128           # Internal dimension (vs 256 for full)
    n_layers: int = 4            # Transformer layers (vs 6)
    n_heads: int = 4             # Attention heads (vs 8)
    d_ff: int = 512              # FFN internal (vs 1024)
    patch_size: int = 8          # Pixels per patch (vs 16)
    patch_overlap: int = 4       # Overlap between patches (vs 8)
    max_seq_len: int = N_BINS    # 128 bins (vs 4096)
    dropout: float = 0.1

    # ── MAE Decoder ──
    d_decoder: int = 64          # Lightweight decoder (vs 128)
    n_decoder_layers: int = 2
    n_decoder_heads: int = 4

    # ── No adversarial discriminator ──
    # GRL removed: negative result confirmed + window too short
    # for significant instrumental contamination.


@dataclass
class TrainConfig:
    """Training configuration for the Hα encoder."""

    # ── Data ──
    dataset_name: str = "anonym-submit-26/bess-bench-26"
    subset: str = "Halpha_all"    # All spectra covering Hα ±50 Å
    batch_size: int = 256         # Larger batch (small input → more memory available)
    num_workers: int = 4
    max_length: int = N_BINS      # 128 bins

    # ── Crop Hα ──
    halpha_center: float = HALPHA_CENTER
    halpha_half_window: float = HALPHA_HALF_WINDOW
    n_bins: int = N_BINS
    include_echelle: bool = True  # Include echelle orders covering Hα

    # ── Optimisation ──
    epochs: int = 80              # More epochs (more homogeneous data)
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # ── MAE ──
    mask_ratio: float = 0.60     # 60% masking (31 patches → ~19 masked)
    n_mask_blocks: int = 3       # 3 contiguous blocks (short sequence)

    # ── Normalisation ──
    # Local normalisation by pseudo-continuum at window edges
    # (no global normalisation: already on a short window)
    norm_mode: str = "local_continuum"
    n_edge_pixels: int = 10      # Pixels at the edges to estimate the continuum

    # ── Checkpoints ──
    save_every: int = 10
    output_dir: str = "runs"
    run_name: Optional[str] = None

    # ── Logging ──
    # The entire benchmark v1.0 logs into a single W&B project: anonym-bess-26.
    # Sub-phases are distinguished via wandb groups/tags (see train.py and SLURM).
    wandb_project: str = "anonym-bess-26"
    wandb_entity: Optional[str] = None

    # ── Divers ──
    seed: int = 42
    device: str = "cuda"

    # ── Star split ──
    # Same test stars as the full encoder for comparability
    test_stars: list = field(default_factory=lambda: [
        "GAM CAS", "PLEIONE", "ZET TAU",
        "V442 AND", "PI AQR", "DEL SCO",
        "28 CYG", "OME CMA",
    ])

    # ── Filtre SNR minimal (None = pas de filtre) ──
    min_snr: Optional[float] = None
