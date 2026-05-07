"""
model.py — Specialised Hα encoder (Stage 1 Hα) with MAE.

Compact version of SpectralEncoder adapted for spectra cropped
to 128 bins around Hα (6512.8–6612.8 Å).

Architecture:
  Hα spectrum [128] → Patches [31, 8] → Projection [31, 128]
  → + Wavelength PE → 60% masking → [CLS] + Visible [~13, 128]
  → 4-layer Transformer → CLS embedding z_halpha [128]
  → MAE Decoder [reconstruction]

Key differences from the full encoder (stage1/model.py):
  - Input : 128 bins (vs 4096)
  - Patches : 8 px, overlap 4, step 4 → 31 patches (vs 511)
  - d_model : 128 (vs 256)
  - n_layers : 4 (vs 6)
  - No GRL/discriminator (negative result confirmed)
  - ~300K params (vs ~5.3M)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ══════════════════════════════════════════════════════════════════════════════
# CONTIGUOUS MASKING
# ══════════════════════════════════════════════════════════════════════════════

def contiguous_masking(n_patches: int, mask_ratio: float = 0.60,
                       n_blocks: int = 3) -> np.ndarray:
    """
    Generate a contiguous-block mask for MAE.

    Adapted for 31 patches (vs 511 for the full encoder):
      - 60% masking → ~19 masked patches, ~12 visible
      - 3 contiguous blocks (vs 4) because the sequence is short

    With 12 visible patches + 1 CLS = 13 tokens for the encoder.
    This is sufficient for a 4-layer Transformer.
    """
    n_masked = int(n_patches * mask_ratio)

    if n_masked == 0 or n_patches < n_blocks:
        return np.zeros(n_patches, dtype=bool)

    mask = np.zeros(n_patches, dtype=bool)
    block_size = max(1, n_masked // n_blocks)

    possible_starts = np.arange(0, max(1, n_patches - block_size))

    if len(possible_starts) < n_blocks:
        starts = possible_starts
    else:
        starts = np.sort(
            np.random.choice(possible_starts, size=n_blocks, replace=False)
        )

    for s in starts:
        end = min(s + block_size, n_patches)
        mask[s:end] = True

    # Fill if necessary
    current = mask.sum()
    if current < n_masked:
        unmasked = np.where(~mask)[0]
        extra = min(n_masked - current, len(unmasked))
        if extra > 0:
            chosen = np.random.choice(unmasked, size=extra, replace=False)
            mask[chosen] = True

    return mask


# ══════════════════════════════════════════════════════════════════════════════
# ENCODAGE POSITIONNEL PAR LONGUEUR D'ONDE
# ══════════════════════════════════════════════════════════════════════════════

class WavelengthPE(nn.Module):
    """
    Sinusoidal positional encoding based on wavelength (Å).

    Identique au full encoder : PE(λ) = sin/cos(λ / 10000 × div_term).

    Even over a 100 Å window, the relative position is physically
    significative : les ailes bleue (λ < 6562.8) et rouge (λ > 6562.8)
    of Hα have different physical meanings (V/R ratio, asymmetries).
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, lambda_means: torch.Tensor) -> torch.Tensor:
        """
        lambda_means: [B, N_patches] — λ moyen de chaque patch (en Å).
        Retourne: [B, N_patches, d_model].
        """
        pos = lambda_means.unsqueeze(-1) / 10000.0
        pe = torch.zeros(*lambda_means.shape, self.d_model,
                         device=lambda_means.device)
        pe[..., 0::2] = torch.sin(pos * self.div_term)
        pe[..., 1::2] = torch.cos(pos * self.div_term)
        return pe


# ══════════════════════════════════════════════════════════════════════════════
# MAE DECODER
# ══════════════════════════════════════════════════════════════════════════════

class MAEDecoder(nn.Module):
    """
    Lightweight MAE decoder for reconstructing masked Hα patches.

    Smaller than the full encoder decoder:
      - d_decoder = 64 (vs 128)
      - n_layers = 2
      - head projette vers patch_size = 8 (vs 16)
    """

    def __init__(self, d_encoder: int, d_decoder: int, n_layers: int,
                 n_heads: int, patch_size: int):
        super().__init__()
        self.d_decoder = d_decoder

        # Projection encoder → decoder
        self.proj = nn.Linear(d_encoder, d_decoder)

        # Mask token appris
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_decoder) * 0.02)

        # Decoder mini-Transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_decoder, nhead=n_heads,
            dim_feedforward=d_decoder * 4,
            activation="gelu", batch_first=True, dropout=0.1,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

        # Prediction head: d_decoder → patch_size
        self.head = nn.Linear(d_decoder, patch_size)

    def forward(self, encoded_visible, visible_pe, full_pe, mask):
        """
        Reconstruct masked patches.

        Inputs:
          encoded_visible : [B, N_vis, d_encoder]
          visible_pe      : [B, N_vis, d_decoder]
          full_pe         : [B, N_all, d_decoder]
          mask            : [B, N_all] — True = masked

        Output:
          [B, N_all, patch_size]
        """
        B, N_all = mask.shape

        vis = self.proj(encoded_visible)

        full_seq = self.mask_token.expand(B, N_all, -1).clone()
        vis_positions = (~mask)
        full_seq[vis_positions] = vis.reshape(-1, self.d_decoder)

        full_seq = full_seq + full_pe

        decoded = self.decoder(full_seq)
        return self.head(decoded)


# ══════════════════════════════════════════════════════════════════════════════
# ENCODEUR SPECTRAL Hα
# ══════════════════════════════════════════════════════════════════════════════

class SpectralEncoderHalpha(nn.Module):
    """
    MAE Transformer encoder for cropped Hα spectra (128 bins).

    Pipeline interne :
      1. Patchify    : split into 31 patches of 8 pixels (step=4)
      2. Projection  : chaque patch [8] → token [128]
      3. PE          : positional encoding based on λ
      4. Masquage    : ne garde que ~40% des patches (12 visibles)
      5. [CLS]       : global summary token
      6. Transformer : 4 couches d'auto-attention
      7. Sortie      : z_halpha = LayerNorm(CLS) ∈ ℝ^128

    Avec 128 bins au lieu de 4096 :
      - 31 patches au lieu de 511
      - 12 visibles au lieu de 153
      - ~10× faster to run
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.patch_overlap = cfg.patch_overlap
        self.d_model = cfg.d_model
        self.step = cfg.patch_size - cfg.patch_overlap  # = 4

        # Projection patch → token
        self.patch_proj = nn.Linear(cfg.patch_size, cfg.d_model)

        # Encodage positionnel
        self.wave_pe = WavelengthPE(cfg.d_model)

        # Token CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers
        )

        # Normalisation finale
        self.norm = nn.LayerNorm(cfg.d_model)

    def patchify(self, flux, wavelengths, validity):
        """
        Split the 128-bin spectrum into 31 patches of 8 pixels.

        Avec step=4 : N = (128 - 8) / 4 + 1 = 31 patches
        """
        patches = flux.unfold(-1, self.patch_size, self.step)
        lam_patches = wavelengths.unfold(-1, self.patch_size, self.step)
        val_patches = validity.unfold(-1, self.patch_size, self.step)

        lambda_means = lam_patches.mean(-1)
        patch_valid = val_patches.mean(-1)

        return patches, lambda_means, patch_valid

    def forward(self, flux, wavelengths, validity, mask=None):
        """
        Forward pass.

        Inputs:
          flux        : [B, 128]
          wavelengths : [B, 128]
          validity    : [B, 128]
          mask        : [B, 31] optional

        Outputs:
          z            : [B, d_model]       — CLS embedding (z_halpha)
          encoded      : [B, N_vis, d_model] — encoded visible tokens
          patches      : [B, 31, 8]          — all patches
          lambda_means : [B, 31]             — mean λ per patch
          mask         : [B, 31]             — mask used
          wpe          : [B, 31, d_model]    — positional encoding
        """
        B = flux.shape[0]

        # Patchify
        patches, lambda_means, patch_valid = self.patchify(
            flux, wavelengths, validity
        )
        N = patches.shape[1]

        # Projection + PE
        tokens = self.patch_proj(patches)
        wpe = self.wave_pe(lambda_means)
        tokens = tokens + wpe

        # Masquage MAE
        if mask is None:
            mask = torch.zeros(B, N, dtype=torch.bool, device=flux.device)

        visible_mask = ~mask
        n_visible = visible_mask[0].sum().item()

        visible_tokens = tokens[visible_mask].view(B, n_visible, self.d_model)
        visible_pe = wpe[visible_mask].view(B, n_visible, self.d_model)

        # CLS + tokens visibles
        cls = self.cls_token.expand(B, -1, -1)
        input_tokens = torch.cat([cls, visible_tokens], dim=1)

        # Masque d'attention
        visible_valid = patch_valid[visible_mask].view(B, n_visible)
        attn_pad = torch.cat([
            torch.ones(B, 1, device=flux.device),
            (visible_valid > 0.1).float()
        ], dim=1)
        src_key_padding_mask = (attn_pad == 0)

        # Transformer
        encoded = self.encoder(
            input_tokens, src_key_padding_mask=src_key_padding_mask
        )

        # Embedding CLS
        z = self.norm(encoded[:, 0])

        return z, encoded[:, 1:], patches, lambda_means, mask, wpe


# ══════════════════════════════════════════════════════════════════════════════
# FULL MODEL — STAGE 1 Hα
# ══════════════════════════════════════════════════════════════════════════════

class Stage1HalphaModel(nn.Module):
    """
    Assembly: Hα Encoder + MAE Decoder.

    No instrument discriminator (GRL removed).
    The only loss is the MAE reconstruction of masked patches.
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()

        self.encoder = SpectralEncoderHalpha(model_cfg)

        self.mae_decoder = MAEDecoder(
            d_encoder=model_cfg.d_model,
            d_decoder=model_cfg.d_decoder,
            n_layers=model_cfg.n_decoder_layers,
            n_heads=model_cfg.n_decoder_heads,
            patch_size=model_cfg.patch_size,
        )

        # PE projection for the decoder
        self.pe_proj = nn.Linear(model_cfg.d_model, model_cfg.d_decoder)

    def forward(self, flux, wavelengths, validity, mask):
        """
        Forward pass: encoder → MAE reconstruction.

        Inputs:
          flux        : [B, 128]
          wavelengths : [B, 128]
          validity    : [B, 128]
          mask        : [B, 31] — MAE mask

        Returns:
          z             : [B, 128]     — CLS embedding (z_halpha)
          mae_loss      : scalar       — MSE reconstruction of masked patches
          reconstructed : [B, 31, 8]   — reconstructed patches
          patches       : [B, 31, 8]   — original patches (target)
          mask          : [B, 31]      — mask used
        """
        # Encode
        z, encoded_vis, patches, lambda_means, mask, wpe = self.encoder(
            flux, wavelengths, validity, mask
        )

        B, N, P = patches.shape

        # PE for the decoder
        full_pe = self.pe_proj(wpe)
        vis_mask = ~mask
        n_vis = vis_mask[0].sum().item()
        vis_pe = full_pe[vis_mask].view(B, n_vis, -1)

        # MAE decoding
        reconstructed = self.mae_decoder(encoded_vis, vis_pe, full_pe, mask)

        # MAE loss: MSE on masked patches only
        target = patches
        mae_loss = ((reconstructed - target) ** 2)
        mask_expanded = mask.unsqueeze(-1).expand_as(mae_loss)
        n_masked_total = mask_expanded.sum()

        if n_masked_total > 0:
            mae_loss = (mae_loss * mask_expanded.float()).sum() / n_masked_total
        else:
            mae_loss = mae_loss.mean()

        return {
            "z": z,
            "mae_loss": mae_loss,
            "reconstructed": reconstructed,
            "patches": patches,
            "mask": mask,
        }

    def get_embeddings(self, flux, wavelengths, validity):
        """
        Inference mode: CLS embeddings without masking.
        Retourne z_halpha ∈ ℝ^128.
        """
        z, _, _, _, _, _ = self.encoder(flux, wavelengths, validity, mask=None)
        return z
