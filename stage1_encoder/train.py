"""
train.py — Training loop for the specialised Hα encoder.

Simplified compared to the full encoder:
  - No GRL (MAE loss only)
  - Shorter input (128 bins → 8px patches → 31 patches)
  - Larger batch (256 because GPU memory is under-utilised)
  - Fast training (~20-30 min vs ~2h for the full encoder)

Usage:
    cd <repo_root>/stage1_encoder
    python train.py
    python train.py --run_name Halpha_clean --min_snr 200 --epochs 100
    python train.py --no_echelle --run_name Halpha_single_order
"""

import argparse
import os
import time
import json
from pathlib import Path

import numpy as np
import torch
import wandb

from config import ModelConfig, TrainConfig
from dataset import prepare_data
from model import Stage1HalphaModel, contiguous_masking


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 Hα — Specialized Spectral Encoder Training"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--min_snr", type=float, default=None,
                        help="Minimum SNR filter (None = no filter)")
    parser.add_argument("--no_echelle", action="store_true", default=False,
                        help="Exclude echelle orders")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="anonym-bess-26")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default="bemae_pretrain_v1",
                        help="W&B group (default: bemae_pretrain_v1 for the encoder pretraining phase)")
    parser.add_argument("--wandb_tags", nargs="*", default=None,
                        help="Additional W&B tags (e.g. seed42, abalation)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=None,
                        help="Override d_model (128 or 256)")
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Override n_layers (default: 4)")
    parser.add_argument("--patch_size", type=int, default=None,
                        help="Override patch_size (default: 8). patch_overlap is set to patch_size//2 unless --patch_overlap is also given.")
    parser.add_argument("--patch_overlap", type=int, default=None,
                        help="Override patch_overlap (default: patch_size//2)")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg, device):
    """
    Train the Hα encoder for one epoch.

    Single loss: MAE (reconstruction of masked patches).
    No adversarial loss.
    """
    model.train()

    total_mae = 0.0
    n_batches = 0

    # Number of patches: (128 - 8) / 4 + 1 = 31
    step = model.encoder.step
    n_patches = (cfg.max_length - model.encoder.patch_size) // step + 1

    for batch in train_loader:
        flux = batch["flux"].to(device)              # [B, 128]
        wavelengths = batch["wavelengths"].to(device)  # [B, 128]
        validity = batch["validity"].to(device)        # [B, 128]

        B = flux.shape[0]

        # Contiguous mask — ONE MASK PER SAMPLE to maximise diversity
        masks = np.stack([
            contiguous_masking(n_patches, cfg.mask_ratio, cfg.n_mask_blocks)
            for _ in range(B)
        ])
        mask = torch.tensor(
            masks, dtype=torch.bool, device=device
        )

        # Forward
        out = model(flux, wavelengths, validity, mask)
        loss = out["mae_loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        total_mae += out["mae_loss"].item()
        n_batches += 1

    return {
        "mae_loss": total_mae / max(n_batches, 1),
        "lr": scheduler.get_last_lr()[0],
    }


@torch.no_grad()
def validate(model, test_loader, cfg, device):
    """Evaluation on test data with a reproducible mask."""
    model.eval()

    total_mae = 0.0
    n_batches = 0

    step = model.encoder.step
    n_patches = (cfg.max_length - model.encoder.patch_size) // step + 1

    # Fixed seed for validation → reproducible results
    val_rng = np.random.RandomState(42)

    for batch in test_loader:
        flux = batch["flux"].to(device)
        wavelengths = batch["wavelengths"].to(device)
        validity = batch["validity"].to(device)
        B = flux.shape[0]

        # Deterministic mask per sample for reproducibility
        old_state = np.random.get_state()
        np.random.set_state(val_rng.get_state())
        masks = np.stack([
            contiguous_masking(n_patches, cfg.mask_ratio, cfg.n_mask_blocks)
            for _ in range(B)
        ])
        val_rng_state = np.random.get_state()
        val_rng = np.random.RandomState(0)
        val_rng.set_state(val_rng_state)
        np.random.set_state(old_state)

        mask = torch.tensor(masks, dtype=torch.bool, device=device)

        out = model(flux, wavelengths, validity, mask)
        total_mae += out["mae_loss"].item()
        n_batches += 1

    return {
        "val_mae_loss": total_mae / max(n_batches, 1),
    }


def main():
    args = parse_args()

    # ── Configuration ──
    train_cfg = TrainConfig(
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        num_workers=args.num_workers,
    )

    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.lr is not None:
        train_cfg.lr = args.lr
    if args.mask_ratio is not None:
        train_cfg.mask_ratio = args.mask_ratio
    if args.min_snr is not None:
        train_cfg.min_snr = args.min_snr
    if args.no_echelle:
        train_cfg.include_echelle = False

    # Nom du run
    if args.run_name is not None:
        train_cfg.run_name = args.run_name
    else:
        parts = ["Halpha"]
        if train_cfg.min_snr is not None:
            parts.append(f"snr{int(train_cfg.min_snr)}")
        if not train_cfg.include_echelle:
            parts.append("noech")
        train_cfg.run_name = "_".join(parts) if len(parts) > 1 else "Halpha_all"

    # Model configuration
    model_cfg = ModelConfig()
    if args.d_model is not None:
        model_cfg.d_model = args.d_model
        model_cfg.d_ff = args.d_model * 4
    if args.n_layers is not None:
        model_cfg.n_layers = args.n_layers
    if args.patch_size is not None:
        model_cfg.patch_size = args.patch_size
        # Default overlap = 50% of patch (unless explicitly overridden)
        if args.patch_overlap is None:
            model_cfg.patch_overlap = max(1, args.patch_size // 2)
    if args.patch_overlap is not None:
        model_cfg.patch_overlap = args.patch_overlap

    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    print("=" * 60)
    print("Preparing Hα data...")
    train_loader, test_loader = prepare_data(train_cfg)

    # ── Model ──
    model = Stage1HalphaModel(model_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,} total, {n_train_params:,} trainable")

    # Verify number of patches
    step = model_cfg.patch_size - model_cfg.patch_overlap
    n_patches = (model_cfg.max_seq_len - model_cfg.patch_size) // step + 1
    n_masked = int(n_patches * train_cfg.mask_ratio)
    n_visible = n_patches - n_masked
    print(f"Patches: {n_patches} total, {n_masked} masked ({train_cfg.mask_ratio:.0%}), "
          f"{n_visible} visible")

    # ── Optimiseur + Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    total_steps = train_cfg.epochs * len(train_loader)
    warmup_steps = train_cfg.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Dossier de sortie ──
    run_dir = Path(train_cfg.output_dir) / train_cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "model": vars(model_cfg),
        "train": {k: v for k, v in vars(train_cfg).items()},
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    # ── Wandb ──
    wandb_tags = list(args.wandb_tags) if args.wandb_tags else []
    # Auto tags to make UI filtering easier
    wandb_tags += [f"seed{train_cfg.seed}", "encoder", "halpha", "v1"]
    wandb.init(
        project=train_cfg.wandb_project,
        entity=train_cfg.wandb_entity,
        name=train_cfg.run_name,
        group=args.wandb_group,
        tags=sorted(set(wandb_tags)),
        config=config_dict,
        dir=str(run_dir),
    )
    wandb.watch(model, log="gradients", log_freq=100)

    # ── Training loop ──
    print("=" * 60)
    print(f"Training: {train_cfg.run_name}")
    print(f"  Epochs: {train_cfg.epochs}, Batch: {train_cfg.batch_size}")
    print(f"  LR: {train_cfg.lr}, Mask ratio: {train_cfg.mask_ratio}")
    print(f"  d_model: {model_cfg.d_model}, n_layers: {model_cfg.n_layers}")
    print(f"  Include echelle: {train_cfg.include_echelle}")
    if train_cfg.min_snr is not None:
        print(f"  Min SNR: {train_cfg.min_snr}")
    print("=" * 60)

    best_val_mae = float("inf")
    patience_counter = 0
    patience = 15  # Early stopping patience

    for epoch in range(train_cfg.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, epoch, train_cfg, device
        )
        val_metrics = validate(model, test_loader, train_cfg, device)

        elapsed = time.time() - t0

        # Logging
        log_dict = {
            "epoch": epoch,
            "train/mae_loss": train_metrics["mae_loss"],
            "train/lr": train_metrics["lr"],
            "val/mae_loss": val_metrics["val_mae_loss"],
            "epoch_time_s": elapsed,
        }
        wandb.log(log_dict)

        print(
            f"[Epoch {epoch+1:3d}/{train_cfg.epochs}] "
            f"MAE={train_metrics['mae_loss']:.6f} "
            f"| Val MAE={val_metrics['val_mae_loss']:.6f} "
            f"| LR={train_metrics['lr']:.2e} "
            f"({elapsed:.1f}s)"
        )

        # Save best model
        if val_metrics["val_mae_loss"] < best_val_mae:
            best_val_mae = val_metrics["val_mae_loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae_loss": best_val_mae,
                "config": config_dict,
            }, run_dir / "best.pt")
            print(f"  → Saved best model (val_mae={best_val_mae:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (patience={patience})")
            break

        # Periodic checkpoint
        if (epoch + 1) % train_cfg.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config_dict,
            }, run_dir / f"checkpoint_ep{epoch+1}.pt")

    # Sauvegarde finale
    torch.save({
        "epoch": train_cfg.epochs - 1,
        "model_state_dict": model.state_dict(),
        "config": config_dict,
    }, run_dir / "final.pt")

    print(f"\nTraining complete. Best val MAE: {best_val_mae:.6f}")
    print(f"Checkpoints in: {run_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
