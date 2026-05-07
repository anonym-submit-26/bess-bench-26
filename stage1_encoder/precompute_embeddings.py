"""
precompute_embeddings.py — Pre-computation of z_halpha embeddings for downstream
tasks (T1, T2, T3).

Loads a BeMAE-Hα checkpoint, iterates over all spectra in the
`anonym-submit-26/bess-bench-26` dataset that cover Hα ±50 Å, and saves a
PyTorch dict `star_data.pt` indexed by star name:

    {
      "GAM CAS": {
        "mjds":       Tensor [n_obs]     — sorted MJDs
        "embeddings": Tensor [n_obs, 128] — z_halpha
        "snrs":       Tensor [n_obs]
        "ews":        Tensor [n_obs]
      },
      ...
    }

This file is consumed by:
  - downstream/probe_features.py       (T1: linear Hα single-line probes)
  - downstream/probe_t2_hbeta_to_halpha.py  (T2 : cross-line Hβ→Hα)
  - downstream/t3_pca_ridge_temporal_baseline.py  (T3 : PCA+Ridge horizon)
  - downstream/t3_ts_fm_baselines.py   (T3 : Chronos/TimesFM zero-shot)

Usage :
    cd bess_bench/stage1_encoder
    python precompute_embeddings.py --checkpoint runs/Halpha_all_seed42/best.pt
    # → writes ../data/embeddings_halpha/star_data.pt

To produce one file per seed (used by `run_multiseed.slurm`):
    python precompute_embeddings.py \
        --checkpoint runs/Halpha_all_seed123/best.pt \
        --output_dir ../data/embeddings_halpha_seed123
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from dataset import HalphaSpectralDataset, filter_halpha_spectra
from model import Stage1HalphaModel


@torch.no_grad()
def precompute(checkpoint_path: str, output_dir: str, device: str = "cuda",
               batch_size: int = 256, num_workers: int = 4,
               include_echelle: bool = True):
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Charger config + poids ─────────────────────────────────────────────
    ckpt_path = Path(checkpoint_path)
    cfg_json = ckpt_path.parent / "config.json"

    with open(cfg_json) as f:
        cfg_dict = json.load(f)

    # Accept both the original training config (keys "model"/"train") and the
    # HuggingFace bundle config (keys "model_config"/"training"). Both come
    # from the same training run; only the layout differs.
    model_section = cfg_dict.get("model", cfg_dict.get("model_config", {}))
    train_section = cfg_dict.get("train", cfg_dict.get("training", {}))
    # halpha_half_window may live at the input level in the HF bundle.
    if "halpha_half_window" not in train_section:
        hw = (cfg_dict.get("input", {}) or {}).get("halpha_half_window_angstrom")
        if hw is not None:
            train_section = {**train_section, "halpha_half_window": hw}

    model_cfg = ModelConfig(**{k: v for k, v in model_section.items()
                               if k in ModelConfig.__dataclass_fields__})
    train_cfg = TrainConfig(**{k: v for k, v in train_section.items()
                               if k in TrainConfig.__dataclass_fields__})

    model = Stage1HalphaModel(model_cfg).to(device_obj)
    ckpt = torch.load(ckpt_path, map_location=device_obj, weights_only=False)
    # Three checkpoint layouts are supported:
    #   1. Original training checkpoint: dict with key "model_state_dict" (full
    #      model = encoder + MAE decoder).
    #   2. Plain full-model state_dict (top-level keys like "encoder.*",
    #      "mae_decoder.*").
    #   3. HuggingFace bundle: encoder-only state_dict (no "encoder." prefix,
    #      no decoder keys). This is what is published as
    #      ``anonym-submit-26/bemae-halpha-v1/pytorch_model.bin``.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        val_mae = ckpt.get("val_mae_loss", "?")
    else:
        # Detect HF encoder-only bundle: it has no "mae_decoder." or "pe_proj."
        # keys (those are top-level modules of Stage1HalphaModel).
        keys = list(ckpt.keys())
        has_full_model_keys = any(k.startswith("mae_decoder.") or k.startswith("pe_proj.")
                                  for k in keys)
        is_hf_encoder_only = (len(keys) > 0) and not has_full_model_keys
        if is_hf_encoder_only:
            # Load only the encoder sub-module (downstream tasks do not need
            # the MAE decoder).
            missing, unexpected = model.encoder.load_state_dict(ckpt, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"Unexpected keys when loading HF encoder bundle: {unexpected}"
                )
            print(">>> Loaded HuggingFace encoder-only bundle (decoder skipped)")
        else:
            model.load_state_dict(ckpt)
        epoch = train_section.get("epoch_best", "?")
        val_mae = train_section.get("val_mae_loss", "?")
    model.eval()
    print(f">>> Loaded {ckpt_path} (epoch {epoch}, val_mae={val_mae})")

    # ── Full BeSS dataset, filtered on Hα coverage ──────────────────────────
    print(">>> Loading dataset anonym-submit-26/bess-bench-26 (split=train)")
    raw = load_dataset("anonym-submit-26/bess-bench-26", split="train")
    print(f"    total raw spectra : {len(raw):,}")

    print(f">>> Filtering for Hα coverage (±{train_cfg.halpha_half_window} Å, "
          f"include_echelle={include_echelle})")
    filtered = filter_halpha_spectra(raw, include_echelle=include_echelle)
    print(f"    after filter : {len(filtered):,}")

    ds = HalphaSpectralDataset(filtered, n_bins=train_cfg.n_bins)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        drop_last=False)

    # ── Extraction ─────────────────────────────────────────────────────────
    star_data = defaultdict(lambda: {
        "mjds": [], "embeddings": [], "snrs": [], "ews": [],
    })

    n_processed = 0
    for batch in loader:
        flux = batch["flux"].to(device_obj)
        wl = batch["wavelengths"].to(device_obj)
        val = batch["validity"].to(device_obj)

        z = model.get_embeddings(flux, wl, val)
        z_cpu = z.cpu().numpy()

        stars = batch["star_name"]
        mjds = batch["mjd"].numpy()
        snrs = batch["snr"].numpy()
        ews = batch["ew"].numpy()

        for i in range(len(stars)):
            s = stars[i]
            if not s:
                continue
            ew_i = float(ews[i])
            if not np.isfinite(ew_i):
                ew_i = float("nan")
            else:
                ew_i = float(np.clip(ew_i, -100, 100))
            star_data[s]["mjds"].append(float(mjds[i]))
            star_data[s]["embeddings"].append(z_cpu[i])
            star_data[s]["snrs"].append(float(snrs[i]))
            star_data[s]["ews"].append(ew_i)

        n_processed += len(stars)
        if n_processed % 5000 == 0:
            print(f"    {n_processed:,} processed  |  {len(star_data)} stars")

    print(f">>> Done : {n_processed:,} spectra  |  {len(star_data)} stars")

    # ── Tri MJD + conversion tenseurs ──────────────────────────────────────
    final = {}
    obs_counts = []
    for s, data in star_data.items():
        idx = np.argsort(data["mjds"])
        obs_counts.append(len(idx))
        final[s] = {
            "mjds": torch.tensor([data["mjds"][i] for i in idx], dtype=torch.float32),
            "embeddings": torch.tensor(
                np.stack([data["embeddings"][i] for i in idx]),
                dtype=torch.float32,
            ),
            "snrs": torch.tensor([data["snrs"][i] for i in idx], dtype=torch.float32),
            "ews": torch.tensor([data["ews"][i] for i in idx], dtype=torch.float32),
        }

    torch.save(final, out / "star_data.pt")

    obs = np.array(obs_counts)
    meta = {
        "n_stars": len(final),
        "n_spectra": int(n_processed),
        "d_embedding": int(model_cfg.d_model),
        "checkpoint": Path(ckpt_path).name,
        "seed": (cfg_dict.get("train") or cfg_dict.get("training") or {}).get("seed"),
        "halpha_half_window": train_cfg.halpha_half_window,
        "n_bins": train_cfg.n_bins,
        "include_echelle": include_echelle,
        "obs_stats": {
            "min": int(obs.min()),
            "max": int(obs.max()),
            "median": float(np.median(obs)),
            "mean": float(obs.mean()),
            "total": int(obs.sum()),
        },
        "stars_with_30plus": int((obs >= 30).sum()),
        "stars_with_100plus": int((obs >= 100).sum()),
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f">>> Saved {out / 'star_data.pt'}  ({meta['n_stars']} stars, "
          f"{meta['n_spectra']:,} spectra)")
    print(f"    ≥30 obs : {meta['stars_with_30plus']}   "
          f"≥100 obs : {meta['stars_with_100plus']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="../data/embeddings_halpha")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_echelle", action="store_true", default=False)
    args = parser.parse_args()

    precompute(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_echelle=not args.no_echelle,
    )


if __name__ == "__main__":
    main()
