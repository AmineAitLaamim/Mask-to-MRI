"""
pix2pix training loop.

Implements the step-by-step procedure from the specification:
  1. Sample (mask, real_MRI) pair
  2. Forward pass: fake_MRI = G(mask)
  3. Train discriminator (real + fake)
  4. Train generator (adversarial + L1)
  5. Log losses, save checkpoints + sample grids every N epochs
"""

import os
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .generator import UNetGenerator
from .discriminator import PatchGANDiscriminator
from .losses import GANLoss, discriminator_loss_real, discriminator_loss_fake


# ---------------------------------------------------------------------------
# Learning rate scheduler (original pix2pix: constant then linear decay)
# ---------------------------------------------------------------------------

class LinearLRDecay:
    """
    Learning rate schedule from the original pix2pix paper:
      - Epochs 0..decay_start: constant LR
      - Epochs decay_start..epochs: linearly decay to 0
    """

    def __init__(self, optimizer, lr: float, total_epochs: int, decay_start: int, start_epoch: int = 0):
        self.optimizer = optimizer
        self.lr = lr
        self.total_epochs = total_epochs
        self.decay_start = decay_start
        self.current_epoch = start_epoch

    def step(self):
        self.current_epoch += 1
        if self.current_epoch > self.decay_start:
            # Linear decay: lr goes from initial to 0
            frac = (self.current_epoch - self.decay_start) / (self.total_epochs - self.decay_start)
            new_lr = self.lr * (1.0 - frac)
        else:
            new_lr = self.lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _train_one_batch(
    mask_batch: torch.Tensor,
    real_mri: torch.Tensor,
    generator: UNetGenerator,
    discriminator: PatchGANDiscriminator,
    opt_G: optim.Optimizer,
    opt_D: optim.Optimizer,
    gan_criterion: GANLoss,
    device: torch.device,
    step: int = 0,
    scaler_D: torch.amp.GradScaler | None = None,
    scaler_G: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """
    Perform one training step (one batch) for both D and G.

    Returns dict of loss values for logging.
    """
    mask_batch = mask_batch.to(device)
    real_mri = real_mri.to(device)

    # ----- Step 1: Generate fake MRI -----
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type != "cpu"):
        fake_mri = generator(mask_batch)

    # ----- Step 2: Train Discriminator (every 2 steps) -----
    # Skip training D on odd steps to prevent it from dominating
    if step % 2 == 0:
        opt_D.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type != "cpu"):
            d_real_pred = discriminator(mask_batch, real_mri)
            loss_D_real = discriminator_loss_real(d_real_pred)
            d_fake_pred = discriminator(mask_batch, fake_mri.detach())
            loss_D_fake = discriminator_loss_fake(d_fake_pred)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        if scaler_D is not None:
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()
        else:
            loss_D.backward()
            opt_D.step()
    else:
        loss_D = torch.tensor(0.0, device=device)

    # ----- Step 3: Train Generator -----
    opt_G.zero_grad()

    # Recompute discriminator forward pass for generator's adversarial loss
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type != "cpu"):
        d_fake_pred_g = discriminator(mask_batch, fake_mri)
        loss_G, loss_G_adv, loss_G_L1, loss_perceptual = gan_criterion(d_fake_pred_g, fake_mri, real_mri)

    if scaler_G is not None:
        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()
    else:
        loss_G.backward()
        opt_G.step()

    return {
        "loss_D": loss_D.item(),
        "loss_G": loss_G.item(),
        "loss_G_adv": loss_G_adv.item(),
        "loss_G_L1": loss_G_L1.item(),
        "loss_perceptual": loss_perceptual.item(),
        "loss_D_count": 1 if step % 2 == 0 else 0,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    generator: UNetGenerator,
    discriminator: PatchGANDiscriminator,
    config: dict,
    device: torch.device,
    checkpoint_dir: str = "outputs/checkpoints",
    samples_dir: str = "outputs/samples",
    metrics_dir: str = "outputs/metrics",
    resume_from: str | None = None,
) -> list[dict]:
    """
    Run the full pix2pix training loop.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        generator: U-Net generator
        discriminator: PatchGAN discriminator
        config: Configuration dict (from config.yaml)
        device: torch.device
        checkpoint_dir: Directory to save checkpoints
        samples_dir: Directory to save sample grids
        resume_from: Path to checkpoint file to resume from (optional).
                     If None, will auto-detect latest checkpoint in checkpoint_dir.

    Returns:
        List of per-epoch loss dicts for plotting.
    """
    training_cfg = config["training"]
    epochs = training_cfg["epochs"]
    lr = training_cfg["lr"]
    beta1 = training_cfg["beta1"]
    beta2 = training_cfg.get("beta2", 0.999)
    lambda_l1 = training_cfg["lambda_l1"]
    save_every = training_cfg.get("save_every", 5)

    # Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # Grad Scalers for Mixed Precision
    if device.type == "cuda":
        scaler_D = torch.amp.GradScaler(device.type)
        scaler_G = torch.amp.GradScaler(device.type)
        print("  → Mixed Precision (AMP) enabled for ~2x speedup")
    else:
        scaler_D = None
        scaler_G = None

    # Auto-detect latest checkpoint if resume_from not specified
    if resume_from is None:
        resume_from = find_latest_checkpoint(checkpoint_dir)

    start_epoch = 0
    history = []
    if resume_from and os.path.exists(resume_from):
        start_epoch = load_checkpoint(
            resume_from, generator, discriminator, opt_G, opt_D, 
            scaler_D=scaler_D, scaler_G=scaler_G
        )
        print(f"  → Resuming from checkpoint: {resume_from} (epoch {start_epoch})")
        print()

    # LR scheduler: constant for first half, linear decay for second half
    decay_start = epochs // 2
    lr_scheduler_G = LinearLRDecay(opt_G, lr, epochs, decay_start, start_epoch=start_epoch)
    lr_scheduler_D = LinearLRDecay(opt_D, lr, epochs, decay_start, start_epoch=start_epoch)

    # Loss
    gan_criterion = GANLoss(lambda_l1=lambda_l1).to(device)

    # History
    history = []

    print(f"\nTraining pix2pix: epoch {start_epoch + 1}–{epochs} of {epochs}")
    print(f"  LR={lr}, beta1={beta1}, lambda_L1={lambda_l1}")
    print(f"  Schedule: {decay_start} epochs constant + {decay_start} epochs linear decay")
    print(f"  Checkpoint every {save_every} epochs")
    print(f"  Metrics dir: {metrics_dir}")
    print()

    for epoch in range(start_epoch + 1, epochs + 1):
        # Training phase
        generator.train()
        discriminator.train()

        epoch_losses = {"loss_D": 0.0, "loss_G": 0.0, "loss_G_adv": 0.0, "loss_G_L1": 0.0, "loss_perceptual": 0.0}
        n_batches = 0
        n_d_steps = 0  # Count batches where D was actually trained

        # Global step counter (persistent across epochs) for consistent D skip pattern
        global_step = start_epoch * len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for step, (mask_batch, real_mri) in enumerate(pbar):
            losses = _train_one_batch(
                mask_batch, real_mri,
                generator, discriminator,
                opt_G, opt_D,
                gan_criterion, device, step=global_step + step,
                scaler_D=scaler_D, scaler_G=scaler_G,
            )
            for k, v in losses.items():
                if k == "loss_D_count":
                    n_d_steps += v
                else:
                    epoch_losses[k] += v
            n_batches += 1

            pbar.set_postfix({
                "D": f"{losses['loss_D']:.4f}",
                "G": f"{losses['loss_G']:.4f}",
            })

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        # D loss should be averaged only over steps where D was trained
        if n_d_steps > 0:
            epoch_losses["loss_D"] /= n_d_steps

        # LR step
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        epoch_losses["lr"] = lr_scheduler_G.get_lr()
        epoch_losses["epoch"] = epoch

        history.append(epoch_losses)
        global_step += n_batches

        # Logging
        print(
            f"Epoch {epoch}/{epochs} | "
            f"D: {epoch_losses['loss_D']:.4f} | "
            f"G: {epoch_losses['loss_G']:.4f} | "
            f"G_adv: {epoch_losses['loss_G_adv']:.4f} | "
            f"G_L1: {epoch_losses['loss_G_L1']:.4f} | "
            f"G_perc: {epoch_losses['loss_perceptual']:.4f} | "
            f"LR: {epoch_losses['lr']:.6f}"
        )

        # Save checkpoint + sample grid + loss plot + metrics every N epochs
        if epoch % save_every == 0 or epoch == epochs:
            _save_checkpoint(
                generator, discriminator, opt_G, opt_D, epoch, checkpoint_dir,
                scaler_D=scaler_D, scaler_G=scaler_G
            )
            _save_sample_grid(
                generator, val_loader, epoch, samples_dir, device
            )
            _save_loss_plot(history, os.path.join(samples_dir, "loss_curves.png"))
            _save_metrics(history, os.path.join(metrics_dir, "training_history.json"))

    print("\nTraining complete.")
    return history


# ---------------------------------------------------------------------------
# Live loss plot generation
# ---------------------------------------------------------------------------

def _save_loss_plot(history: list[dict], save_path: str):
    """Generate and save a live loss plot from current history."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import matplotlib.pyplot as plt

    if not history:
        return

    epochs = [h["epoch"] for h in history]
    loss_D = [h["loss_D"] for h in history]
    loss_G = [h["loss_G"] for h in history]
    loss_G_adv = [h["loss_G_adv"] for h in history]
    loss_G_L1 = [h["loss_G_L1"] for h in history]
    loss_perc = [h["loss_perceptual"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, loss_D, "r-", label="Discriminator")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Discriminator Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, loss_G, "b-", label="Generator")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Generator Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, loss_G_adv, "g-", label="Adversarial", alpha=0.7)
    axes[2].plot(epochs, loss_G_L1, "orange", label="L1", alpha=0.7)
    axes[2].plot(epochs, loss_perc, "purple", label="Perceptual", alpha=0.7)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Generator Components")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics saving
# ---------------------------------------------------------------------------

def _save_metrics(history: list[dict], save_path: str):
    """Save training history to a JSON file for persistence."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Convert all values to float for JSON serialization
    serializable = []
    for h in history:
        serializable.append({k: float(v) for k, v in h.items()})
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_metrics(path: str) -> list[dict]:
    """Load training metrics from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def _save_checkpoint(
    generator: UNetGenerator,
    discriminator: PatchGANDiscriminator,
    opt_G: optim.Optimizer,
    opt_D: optim.Optimizer,
    epoch: int,
    checkpoint_dir: str,
    scaler_D: torch.amp.GradScaler | None = None,
    scaler_G: torch.amp.GradScaler | None = None,
):
    """Save model weights and optimizer state."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "opt_G_state_dict": opt_G.state_dict(),
        "opt_D_state_dict": opt_D.state_dict(),
    }
    if scaler_D is not None:
        checkpoint["scaler_D_state_dict"] = scaler_D.state_dict()
    if scaler_G is not None:
        checkpoint["scaler_G_state_dict"] = scaler_G.state_dict()
        
    torch.save(checkpoint, path)
    print(f"  → Saved checkpoint: {path}")


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """
    Find the latest checkpoint in the checkpoint directory.
    Returns the path to the checkpoint file, or None if no checkpoints exist.
    """
    import glob

    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")))
    if not checkpoints:
        return None

    # Return the one with the highest epoch number
    latest = max(checkpoints, key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    return latest


def load_checkpoint(
    path: str,
    generator: UNetGenerator,
    discriminator: PatchGANDiscriminator,
    opt_G: optim.Optimizer | None = None,
    opt_D: optim.Optimizer | None = None,
    scaler_D: torch.amp.GradScaler | None = None,
    scaler_G: torch.amp.GradScaler | None = None,
) -> int:
    """
    Load a checkpoint. Returns the epoch number.
    Optimizers and Scalers are optional (useful for resuming training).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    if opt_G is not None:
        opt_G.load_state_dict(checkpoint["opt_G_state_dict"])
    if opt_D is not None:
        opt_D.load_state_dict(checkpoint["opt_D_state_dict"])
    if scaler_D is not None and "scaler_D_state_dict" in checkpoint:
        scaler_D.load_state_dict(checkpoint["scaler_D_state_dict"])
    if scaler_G is not None and "scaler_G_state_dict" in checkpoint:
        scaler_G.load_state_dict(checkpoint["scaler_G_state_dict"])
    return checkpoint["epoch"]


# ---------------------------------------------------------------------------
# Sample grid generation
# ---------------------------------------------------------------------------

def _save_sample_grid(
    generator: UNetGenerator,
    val_loader: DataLoader,
    epoch: int,
    samples_dir: str,
    device: torch.device,
    n_samples: int = 4,
):
    """Generate a mask | fake | real grid and save as PNG."""
    import numpy as np
    from PIL import Image

    os.makedirs(samples_dir, exist_ok=True)

    generator.eval()

    # Grab n_samples from val loader
    samples = []
    with torch.no_grad():
        for mask, real in val_loader:
            if len(samples) >= n_samples:
                break
            mask = mask.to(device)
            real = real.to(device)
            fake = generator(mask)

            # Denormalize: [-1,1] → [0,255]
            # Mask: (1, H, W) → squeeze
            mask_np = ((mask[0, 0].cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            # Real: (3, H, W) → H×W×3
            real_np = ((real[0].cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            # Fake: (3, H, W) → H×W×3
            fake_np = ((fake[0].cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            samples.append((mask_np, fake_np, real_np))

    # Build grid: 3 columns × n_samples rows
    # Each row: mask | fake | real (side by side)
    rows = []
    for mask_np, fake_np, real_np in samples:
        # Convert mask to 3-channel for display
        mask_3ch = np.stack([mask_np] * 3, axis=-1)
        row = np.concatenate([mask_3ch, fake_np, real_np], axis=1)  # (H, 3*W, 3)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)  # (n_samples*H, 3*W, 3)

    # Save
    img = Image.fromarray(grid)
    path = os.path.join(samples_dir, f"samples_epoch_{epoch}.png")
    img.save(path)
    print(f"  → Saved sample grid: {path}")
