"""
Training loop for med_ddpm_v2 — 2D Conditional DDPM.

Features:
  - EMA via copy.deepcopy (original Med-DDPM approach)
  - LR scheduler: Linear warmup → CosineAnnealingLR(eta_min=1e-5)
  - AMP GradScaler for mixed precision
  - Gradient clipping
  - Drive sync: checkpoint and sample grid copied to Google Drive after every save
  - Full resume: model, ema_model, optimizer, scheduler states
"""

import os
import json
import copy
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import ConditionalDDPM


# ---------------------------------------------------------------------------
# Drive sync helper
# ---------------------------------------------------------------------------

def _sync_to_drive(local_path: str, drive_base: str | None) -> None:
    """Copy a file from local outputs_v2 to Google Drive mirror."""
    if drive_base is None:
        return  # Not on Colab — skip silently
    try:
        outputs_base = "/content/Mask-to-MRI/outputs_v2"
        rel = Path(local_path).relative_to(outputs_base)
        drive_path = Path(drive_base) / rel
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, drive_path)
    except Exception as e:
        print(f"  Drive sync failed: {e}")


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def _save_checkpoint(
    diffusion_model: nn.Module,
    ema_model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    history: list[dict],
    checkpoint_dir: str,
    suffix: str = "v2",
):
    """Save all training states to checkpoint file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_{suffix}_epoch_{epoch}.pt")

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": diffusion_model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "history": history,
    }
    torch.save(checkpoint, path)
    print(f"  Saved checkpoint: {path}")
    return path


# ---------------------------------------------------------------------------
# Sample grid generation
# ---------------------------------------------------------------------------

def _save_sample_grid(
    model: ConditionalDDPM,
    ema_model: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    samples_dir: str,
    device: torch.device,
    n_samples: int = 4,
    suffix: str = "v2",
    ddim_steps: int = 250,
):
    """Generate a mask | fake (EMA) | real grid and save as PNG."""
    import numpy as np
    from PIL import Image

    os.makedirs(samples_dir, exist_ok=True)

    # Use EMA model for sampling after epoch 30; live model before
    use_ema = epoch >= 30
    sampling_model = ema_model if use_ema else model
    tag = "EMA" if use_ema else "live"
    print(f"  Using {tag} model for sampling (epoch {epoch})")

    samples = []
    fake_std_values = []

    with torch.no_grad():
        batches_collected = 0
        for mask, real in val_loader:
            if batches_collected >= n_samples:
                break
            mask = mask.to(device, non_blocking=True)
            real = real.to(device, non_blocking=True)

            # Generate synthetic MRI via DDIM
            fake = sampling_model.sample(mask, ddim_steps=ddim_steps)
            fake_std_values.append(fake.std().item())

            # Debug: FLAIR channel stats
            if batches_collected == 0:
                print(f"  [DEBUG] fake FLAIR mean: {fake[0,0].mean():.3f}, std: {fake[0,0].std():.3f}")
                print(f"  [DEBUG] real FLAIR mean: {real[0,0].mean():.3f}, std: {real[0,0].std():.3f}")

            # Denormalize: [-1,1] → [0,255] — single channel grayscale
            mask_np = ((mask[0, 0].cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            real_np = ((real[0, 0].cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            fake_np = ((fake[0, 0].cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            samples.append((mask_np, fake_np, real_np))
            batches_collected += 1

    # Sanity: detect if samples look like noise
    if fake_std_values:
        avg_fake_std = np.mean(fake_std_values)
        if avg_fake_std < 0.1:
            print(f"  WARNING: Samples look like noise (avg std={avg_fake_std:.3f}) — skipping grid")
            return None
        else:
            print(f"  Sample std: {avg_fake_std:.3f} — saving grid")

    # Build grid: 3 columns × n_samples rows (mask | fake | real)
    rows = []
    for mask_np, fake_np, real_np in samples:
        row = np.concatenate([mask_np, fake_np, real_np], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    img = Image.fromarray(grid, mode='L')
    path = os.path.join(samples_dir, f"{suffix}_samples_epoch_{epoch}.png")
    img.save(path)
    print(f"  Saved sample grid: {path}")
    return path


# ---------------------------------------------------------------------------
# Loss plot
# ---------------------------------------------------------------------------

def _save_loss_plot(history: list[dict], save_path: str):
    """Generate and save a loss plot from training history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, "b-", label="Train Loss", alpha=0.8)

    if any("val_loss" in h for h in history):
        val_epochs = [h["epoch"] for h in history if "val_loss" in h]
        val_loss = [h["val_loss"] for h in history if "val_loss" in h]
        ax.plot(val_epochs, val_loss, "r-", label="Val Loss", alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.set_title("DDPM Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved loss plot: {save_path}")


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_val_loss(
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    max_batches: int = 10,
) -> float:
    """Compute validation loss over a subset of val loader."""
    model.eval()
    val_loss = 0.0
    n_batches = 0

    for mask, mri in val_loader:
        if n_batches >= max_batches:
            break
        mask = mask.to(device)
        mri = mri.to(device)
        loss = model(mri, mask)
        val_loss += loss.item()
        n_batches += 1

    model.train()
    return val_loss / n_batches if n_batches > 0 else float("inf")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: ConditionalDDPM,
    config: dict,
    device: torch.device,
    checkpoint_dir: str | None = None,
    samples_dir: str | None = None,
    metrics_dir: str | None = None,
    resume_from: str | None = None,
):
    """
    Run the full DDPM training loop.

    Args:
        train_loader: Training DataLoader (yields mask, mri)
        val_loader: Validation DataLoader
        model: ConditionalDDPM instance
        config: Configuration dict (from config.py)
        device: torch.device
        checkpoint_dir: Override config path
        samples_dir: Override config path
        metrics_dir: Override config path
        resume_from: Path to checkpoint to resume from
    """
    # Resolve paths
    checkpoint_dir = checkpoint_dir or config["checkpoint_dir"]
    samples_dir = samples_dir or config["samples_dir"]
    metrics_dir = metrics_dir or config["metrics_dir"]
    drive_base = config.get("drive_base")

    epochs = config["epochs"]
    lr = config["lr"]
    warmup_epochs = config["warmup_epochs"]
    ema_decay = config["ema_decay"]
    update_ema_every = config.get("update_ema_every", 10)
    step_start_ema = config.get("step_start_ema", 2000)
    grad_clip = config.get("grad_clip", 1.0)
    save_every = config["save_every"]
    ddim_steps = config.get("ddim_steps", 250)

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # ── LR Scheduler: Linear warmup → CosineAnnealing ─────────────────
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
            ),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5,
            ),
        ],
        milestones=[warmup_epochs],
    )

    # ── EMA model (copy.deepcopy — original Med-DDPM approach) ────────
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    # Freeze EMA model parameters (they are only updated via EMA)
    for p in ema_model.parameters():
        p.requires_grad = False

    # ── AMP GradScaler ────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and config.get("amp", True) else None
    if scaler is not None:
        print(f"  AMP GradScaler enabled")

    # ── Resume from checkpoint ────────────────────────────────────────
    start_epoch = 0
    history = []
    global_step = 0

    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "ema_state_dict" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
        else:
            # Fall back to copying model weights if EMA not in checkpoint
            for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.copy_(live_p.data)
            print("  No EMA state in checkpoint — copied model weights to EMA")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"]
        global_step = checkpoint.get("global_step", start_epoch * len(train_loader))
        history = checkpoint.get("history", [])

        restored_lr = scheduler.get_last_lr()[0]
        print(f"  Resumed from checkpoint: {resume_from} (epoch {start_epoch})")
        print(f"  LR restored to: {restored_lr:.6f}")

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\nTraining DDPM v2: epoch {start_epoch + 1}–{epochs} of {epochs}")
    print(f"  LR={lr} (warmup {warmup_epochs} epochs → cosine decay, eta_min=1e-5)")
    print(f"  EMA decay={ema_decay}, update every {update_ema_every} steps")
    print(f"  DDIM steps={ddim_steps}, Grad clipping max_norm={grad_clip}")
    print(f"  Checkpoint every {save_every} epochs")
    print()

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        ema_model.eval()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for mask_batch, mri_batch in pbar:
            mask_batch = mask_batch.to(device, non_blocking=True)
            mri_batch = mri_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward + backward with AMP
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(mri_batch, mask_batch)
                    loss = loss.mean()  # DataParallel returns per-GPU losses
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(mri_batch, mask_batch)
                loss = loss.mean()  # DataParallel returns per-GPU losses
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # ── EMA update ────────────────────────────────────────────────
            if global_step >= step_start_ema and global_step % update_ema_every == 0:
                with torch.no_grad():
                    for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(ema_decay).add_(live_p.data, alpha=1.0 - ema_decay)

        avg_loss = epoch_loss / n_batches
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Validation + logging at checkpoint intervals ──────────────
        if epoch % save_every == 0 or epoch == epochs:
            val_loss = _evaluate_val_loss(val_loader, model, device)
            epoch_record = {
                "epoch": epoch,
                "loss": avg_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            }
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        else:
            epoch_record = {
                "epoch": epoch,
                "loss": avg_loss,
                "lr": current_lr,
            }
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        history.append(epoch_record)

        # ── Save checkpoint + samples ─────────────────────────────────
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = _save_checkpoint(
                model, ema_model, optimizer, scheduler, epoch, global_step, history,
                checkpoint_dir, suffix="v2",
            )
            _sync_to_drive(ckpt_path, drive_base)

            sample_path = _save_sample_grid(
                model, ema_model, val_loader, epoch, samples_dir, device,
                suffix="v2", ddim_steps=ddim_steps,
            )
            if sample_path:
                _sync_to_drive(sample_path, drive_base)

            _save_loss_plot(history, os.path.join(samples_dir, "v2_loss_curves.png"))
            _sync_to_drive(os.path.join(samples_dir, "v2_loss_curves.png"), drive_base)
            _save_metrics(history, os.path.join(metrics_dir, "v2_training_history.json"))
            _sync_to_drive(os.path.join(metrics_dir, "v2_training_history.json"), drive_base)

    print("\nDDPM v2 training complete.")
    return history


# ---------------------------------------------------------------------------
# Metrics saving
# ---------------------------------------------------------------------------

def _save_metrics(history: list[dict], save_path: str):
    """Save training history to JSON."""
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    serializable = [{k: float(v) for k, v in h.items()} for h in history]
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
