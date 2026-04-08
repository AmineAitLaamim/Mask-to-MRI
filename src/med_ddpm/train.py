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
    max_grad_norm: float | None = 1.0,
) -> float:
    """
    Perform one DDPM training step.

    Returns: loss value
    """
    # Non-blocking transfers for overlap with compute
    mask_batch = mask_batch.to(device, non_blocking=True)
    mri_batch = mri_batch.to(device, non_blocking=True)
    B = mri_batch.shape[0]

    # Expanded AMP autocast covers the entire forward pass including timestep sampling
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        # Sample random timesteps (inside autocast for full coverage)
        t = torch.randint(0, ddpm.timesteps, (B,), device=device, dtype=torch.long)

        optimizer.zero_grad()

        loss = ddpm.p_losses(mri_batch, mask_batch, t)

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
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
    use_compile: bool = False,
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
        use_compile: If True, apply torch.compile() to the model (PyTorch 2.0+)

    Returns:
        List of per-epoch loss dicts
    """
    ddpm_cfg = config.get("med_ddpm", {})
    epochs = ddpm_cfg.get("epochs", config["training"]["epochs"])
    lr = ddpm_cfg.get("lr", 2e-5)
    ema_decay = ddpm_cfg.get("ema_decay", 0.995)
    save_every = ddpm_cfg.get("save_every", 5)
    max_grad_norm = ddpm_cfg.get("max_grad_norm", 1.0)

    # Enable TF32 for 2-3× speedup on Ampere+ GPUs (RTX 30xx/40xx, A100)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  → TF32 enabled (Ampere+ GPU optimization)")

    # torch.compile (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile"):
        print("  → Applying torch.compile() to model...")
        model = torch.compile(model)

    # Optimizer — fused AdamW if available (PyTorch 2.0+) — ~15% faster optimizer step
    try:
        optimizer = optim.AdamW(model.parameters(), lr=lr, fused=torch.cuda.is_available())
    except TypeError:
        # Fallback for older PyTorch versions without fused argument
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
    print(f"  Grad clipping: max_norm={max_grad_norm}")
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
                max_grad_norm=max_grad_norm,
            )
            epoch_loss += loss
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss:.4f}"})

            # EMA update every batch (skip first epoch — shadow starts from random init)
            if epoch > 1:
                ema.update()

        avg_loss = epoch_loss / n_batches

        epoch_record = {
            "epoch": epoch,
            "loss": avg_loss,
            "lr": lr,
        }

        # Validation loss at checkpoint intervals (not every epoch to save time)
        if epoch % save_every == 0 or epoch == epochs:
            val_loss = _evaluate_val_loss(val_loader, model, ddpm, device)
            epoch_record["val_loss"] = val_loss
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f}")
        else:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        history.append(epoch_record)

        # Save checkpoint + samples + loss plot every N epochs
        if epoch % save_every == 0 or epoch == epochs:
            _save_checkpoint(
                model, optimizer, ema, epoch, history,
                checkpoint_dir, suffix="ddpm"
            )
            _save_sample_grid_ddpm(
                ddpm, model, val_loader, epoch, samples_dir, device,
                suffix="ddpm",
            )
            _save_loss_plot_ddpm(history, os.path.join(samples_dir, "ddpm_loss_curves.png"))
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
    torch.save(checkpoint, path, _use_new_zipfile_serialization=True)
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
# Validation loss evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_val_loss(
    val_loader: DataLoader,
    model: ConditionalUNet,
    ddpm: DDPM,
    device: torch.device,
    max_batches: int = 10,
) -> float:
    """
    Compute validation loss over a subset of the validation loader.

    Args:
        val_loader: Validation DataLoader
        model: U-Net noise predictor
        ddpm: DDPM wrapper
        device: torch.device
        max_batches: Maximum number of batches to evaluate (saves time)

    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0.0
    n_batches = 0

    for mask_batch, mri_batch in val_loader:
        if n_batches >= max_batches:
            break
        mask_batch = mask_batch.to(device)
        mri_batch = mri_batch.to(device)
        B = mri_batch.shape[0]
        t = torch.randint(0, ddpm.timesteps, (B,), device=device, dtype=torch.long)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = ddpm.p_losses(mri_batch, mask_batch, t)
        val_loss += loss.item()
        n_batches += 1

    model.train()
    return val_loss / n_batches if n_batches > 0 else float("inf")


# ---------------------------------------------------------------------------
# Loss plot generation
# ---------------------------------------------------------------------------

def _save_loss_plot_ddpm(history: list[dict], save_path: str):
    """Generate and save a loss plot from current DDPM training history."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import matplotlib.pyplot as plt

    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, "b-", label="Train Loss", alpha=0.8)

    if "val_loss" in history[0]:
        val_epochs = [h["epoch"] for h in history if "val_loss" in h]
        val_loss = [h["val_loss"] for h in history if "val_loss" in h]
        ax.plot(val_epochs, val_loss, "r-", label="Val Loss", alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("DDPM Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved DDPM loss plot: {save_path}")


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
