"""
example_usage.py — Minimal inference with BeMAE-Hα.

This script loads a checkpoint from HuggingFace Hub, retrieves a spectrum
de test depuis le dataset BeSS-foundation, et calcule son embedding z_halpha.
"""
import json
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download

MODEL_ID = "anonym-submit-26/bemae-halpha-v1"

# 1. Download the snapshot
ckpt_dir = snapshot_download(MODEL_ID)
sys.path.insert(0, ckpt_dir)

from model import SpectralEncoderHalpha, ModelConfig  # noqa: E402

# 2. Charger la config et les poids
with open(f"{ckpt_dir}/config.json") as f:
    meta = json.load(f)
cfg = ModelConfig(**meta["model_config"])

encoder = SpectralEncoderHalpha(cfg)
state = torch.load(f"{ckpt_dir}/pytorch_model.bin", map_location="cpu")
encoder.load_state_dict(state)
encoder.eval()

# 3. Prepare a dummy spectrum (Hα-centred, 128 bins, pseudo-continuum at 1.0)
B = 4
flux = torch.ones(B, 128) + 0.3 * torch.randn(B, 128) * 0.01
wavelengths = torch.linspace(6512.8, 6612.8, 128).unsqueeze(0).expand(B, -1)
validity = torch.ones(B, 128)

# 4. Inference
with torch.no_grad():
    z_halpha, *_ = encoder(flux, wavelengths, validity, mask=None)

print(f"z_halpha shape : {tuple(z_halpha.shape)}")
print(f"z_halpha[0, :8] : {z_halpha[0, :8].numpy()}")
