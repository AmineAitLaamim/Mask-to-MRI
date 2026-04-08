"""
DDPM Training loop.

Implements the training procedure for conditional DDPM:
  1. Sample (mask, mri) pair
  2. Sample random timestep t
  3. Add noise: x_t = q_sample(mri, t)
  4. Predict noise: epsilon_pred = model(x_t, t, mask)
  5. Loss: MSE(epsilon_pred, epsilon)
  6. Update model weights
"""

import os
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .unet import ConditionalUNet, create_unet
from .diffusion import DDPM


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) — critical for DDPM sample quality
# ---------------------------------------------------------------------------

class EMA:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy of model parameters that decays towards the live model.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters()}

    def update(self):
        """Update shadow parameters towards current model."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self):
        """Copy shadow parameters into model (for evaluation/sampling)."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model parameters (if needed)."""
        # In practice, we keep using the shadow for sampling
        pass

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict):
        for k, v in state_dict.items():
            self.shadow[k] = v.clone()


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _train_one_batch(
    mask_batch: torch.Tensor,
    mri_batch: torch.Tensor,
    model: ConditionalUNet,
    ddpm: DDPM,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """
    Perform one DDPM training step.

    Returns: loss value
    """
    mask_batch = mask_batch.to(device)
    mri_batch = mri_batch.to(device)
    B = mri_batch.shape[0]

    # Sample random timesteps
    t = torch.randint(0, ddpm.timesteps, (B,), device=device, dtype=torch.long)

    optimizer.zero_grad()

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        loss = ddpm.p_losses(mri_batch, mask_batch, t)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: ConditionalUNet,
    ddpm: DDPM,
    config: dict,
    device: torch.device,
    checkpoint_dir: str = "outputs/checkpoints",
    samples_dir: str = "outputs/samples",
    metrics_dir: str = "outputs/metrics",
    resume_from: str | None = None,
) -> list[dict]:
    """
    Run the full DDPM training loop.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: Conditional U-Net noise predictor
        ddpm: DDPM wrapper (noise schedule + forward/reverse)
        config: Configuration dict (from config.yaml)
        device: torch.device
        checkpoint_dir: Directory to save checkpoints
        samples_dir: Directory to save sample grids
        metrics_dir: Directory to save metrics
        resume_from: Path to checkpoint to resume from

    Returns:
        List of per-epoch loss dicts
    """
    ddpm_cfg = config.get("med_ddpm", {})
    epochs = ddpm_cfg.get("epochs", config["training"]["epochs"])
    lr = ddpm_cfg.get("lr", 2e-5)
    ema_decay = ddpm_cfg.get("ema_decay", 0.995)
    save_every = ddpm_cfg.get("save_every", 5)

    # Optimizer (Adam with DDPM-standard lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # EMA
    ema = EMA(model, decay=ema_decay)

    # Grad Scaler
    if device.type == "cuda":
        scaler = torch.amp.GradScaler(device.type)
        print("  → Mixed Precision (AMP) enabled")
    else:
        scaler = None

    # Resume from checkpoint
    start_epoch = 0
    history = []
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        ema.load_state_dict(checkpoint["ema_state_dict"])
        start_epoch = checkpoint["epoch"]
        history = checkpoint.get("history", [])
        print(f"  → Resumed from checkpoint: {resume_from} (epoch {start_epoch})")

    print(f"\nTraining DDPM: epoch {start_epoch + 1}–{epochs} of {epochs}")
    print(f"  LR={lr}, EMA decay={ema_decay}, Timesteps={ddpm.timesteps}")
    print(f"  Checkpoint every {save_every} epochs")
    print()

    for epoch in range(start_epoch + 1, epochs + 1):
        # Training phase
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for mask_batch, mri_batch in pbar:
            loss = _train_one_batch(
                mask_batch, mri_batch,
                model, ddpm,
                optimizer, device,
                scaler=scaler,
            )
            epoch_loss += loss
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss:.4f}"})

            # EMA update every batch
            ema.update()

        avg_loss = epoch_loss / n_batches

        epoch_record = {
            "epoch": epoch,
            "loss": avg_loss,
            "lr": lr,
        }
        history.append(epoch_record)

        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        # Save checkpoint + samples every N epochs
        if epoch % save_every == 0 or epoch == epochs:
            _save_checkpoint(
                model, optimizer, ema, epoch, history,
                checkpoint_dir, suffix="ddpm"
            )
            _save_sample_grid_ddpm(
                ddpm, model, val_loader, epoch, samples_dir, device,
                suffix="ddpm",
            )
            _save_metrics(history, os.path.join(metrics_dir, "ddpm_training_history.json"))

    print("\nDDPM Training complete.")
    return history


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: ConditionalUNet,
    optimizer: optim.Optimizer,
    ema: EMA,
    epoch: int,
    history: list[dict],
    checkpoint_dir: str,
    suffix: str = "ddpm",
):
    """Save model weights, optimizer, EMA state."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_{suffix}_epoch_{epoch}.pt")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, path)
    print(f"  → Saved DDPM checkpoint: {path}")


def find_latest_ddpm_checkpoint(checkpoint_dir: str, suffix: str = "ddpm") -> str | None:
    """Find the latest DDPM checkpoint."""
    import glob
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, f"checkpoint_{suffix}_epoch_*.pt")))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    return latest


def load_ddpm_checkpoint(
    path: str,
    model: ConditionalUNet,
    optimizer: optim.Optimizer | None = None,
    ema: EMA | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[int, list[dict]]:
    """Load a DDPM checkpoint. Returns (epoch, history)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if ema is not None:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    return checkpoint["epoch"], checkpoint.get("history", [])


# ---------------------------------------------------------------------------
# Sample grid generation
# ---------------------------------------------------------------------------

def _save_sample_grid_ddpm(
    ddpm: DDPM,
    model: ConditionalUNet,
    val_loader: DataLoader,
    epoch: int,
    samples_dir: str,
    device: torch.device,
    n_samples: int = 4,
    suffix: str = "ddpm",
):
    """Generate a mask | fake | real grid and save as PNG."""
    import numpy as np
    from PIL import Image

    os.makedirs(samples_dir, exist_ok=True)

    # Use EMA model for sampling
    model.eval()

    samples = []
    with torch.no_grad():
        for mask, real in val_loader:
            if len(samples) >= n_samples:
                break
            mask = mask.to(device)
            real = real.to(device)

            # Sample from DDPM
            fake = ddpm.sample(mask)

            # Denormalize: [-1,1] → [0,255]
            mask_np = ((mask[0, 0].cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            real_np = ((real[0].cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            fake_np = ((fake[0].cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            samples.append((mask_np, fake_np, real_np))

    # Build grid: 3 columns × n_samples rows
    rows = []
    for mask_np, fake_np, real_np in samples:
        mask_3ch = np.stack([mask_np] * 3, axis=-1)
        row = np.concatenate([mask_3ch, fake_np, real_np], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)

    img = Image.fromarray(grid)
    path = os.path.join(samples_dir, f"{suffix}_samples_epoch_{epoch}.png")
    img.save(path)
    print(f"  → Saved DDPM sample grid: {path}")


# ---------------------------------------------------------------------------
# Metrics saving
# ---------------------------------------------------------------------------

def _save_metrics(history: list[dict], save_path: str):
    """Save training history to JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    serializable = [{k: float(v) for k, v in h.items()} for h in history]
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_ddpm_metrics(path: str) -> list[dict]:
    """Load DDPM training metrics from JSON."""
    with open(path, 'r') as f:
        return json.load(f)
