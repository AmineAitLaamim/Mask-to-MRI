"""
Training loop for med_ddpm_v3 — 2D Conditional DDPM.

Features:
  - EMA via state_dict copy (lower memory than deepcopy)
  - Min-SNR weighting (gamma=5) for 3.4x faster convergence
  - Fused AdamW optimizer for 20-30% faster steps
  - LR scheduler: Linear warmup → CosineAnnealingLR(eta_min=1e-5)
  - AMP GradScaler for mixed precision
  - Gradient clipping
  - Drive sync: checkpoint and sample grid copied to Google Drive after every save
  - Full resume: model, ema_model, optimizer, scheduler states
"""

import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import ConditionalDDPM
from .utils import _sync_to_drive


# ---------------------------------------------------------------------------
# Validation SSim
# ---------------------------------------------------------------------------

def _compute_val_ssim(model, val_loader, device, n_batches=4, ddim_steps=50):
    """Quick SSIM estimate on val set using fast 50-step DDIM."""
    from skimage.metrics import structural_similarity as ssim
    import numpy as np

    scores = []
    with torch.no_grad():
        for i, (mask, real) in enumerate(val_loader):
            if i >= n_batches:
                break
            mask = mask.to(device)
            fake = model.sample(mask, ddim_steps=ddim_steps)
            for b in range(fake.shape[0]):
                f = fake[b, 0].cpu().numpy()
                r = real[b, 0].numpy()
                scores.append(ssim(f, r, data_range=2.0))
    return float(np.mean(scores)) if scores else 0.0


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
    suffix: str = "v3",
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
    suffix: str = "v3",
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
    try:
        optimizer = optim.AdamW(model.parameters(), lr=lr, fused=config.get("fused_optimizer", False) and torch.cuda.is_available())
    except TypeError:
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

    # ── EMA model (state_dict copy — lower memory than deepcopy) ──────
    ema_model = ConditionalDDPM(config)
    ema_model.load_state_dict(model.state_dict())
    ema_model = ema_model.to(device)
    ema_model.eval()
    # Propagate CFG drop prob to EMA model (used at sampling time)
    ema_model.diffusion.cfg_drop_prob = model.diffusion.cfg_drop_prob
    # Freeze EMA model parameters (they are only updated via EMA)
    for p in ema_model.parameters():
        p.requires_grad = False

    # EMA decay schedule: ramp from ema_decay_start → ema_decay over ramp_epochs
    ema_decay_start = config.get("ema_decay_start", 0.9)
    ema_decay_ramp_epochs = config.get("ema_decay_ramp_epochs", 50)
    ema_decay = config["ema_decay"]

    def get_ema_decay(epoch: int) -> float:
        """Linear ramp of EMA decay: start low, increase to target."""
        if epoch <= ema_decay_ramp_epochs:
            return ema_decay_start + (ema_decay - ema_decay_start) * (epoch / ema_decay_ramp_epochs)
        return ema_decay

    # ── AMP GradScaler ────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and config.get("amp", True) else None
    if scaler is not None:
        print(f"  AMP GradScaler enabled")

    # ── Resume from checkpoint ────────────────────────────────────────
    start_epoch = 0
    history = []
    global_step = 0

    # Use function arg if provided, otherwise fall back to config
    if resume_from is None:
        resume_from = config.get("resume_from")

    if resume_from is not None and not os.path.exists(resume_from):
        print(f"  ⚠️  WARNING: resume_from file not found: {resume_from}")
        print(f"      Starting from scratch instead.")
        resume_from = None

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
    # When resuming, interpret "epochs" as "additional epochs to run"
    if resume_from and start_epoch > 0:
        total_epochs = start_epoch + epochs
        print(f"Resuming from epoch {start_epoch}, training {epochs} more → epoch {total_epochs}")
    else:
        total_epochs = epochs
        print(f"\nFine-tuning DDPM v3.1: epoch 1–{epochs} of {epochs}")

    print(f"  LR={lr} (warmup {warmup_epochs} epochs → cosine decay, eta_min=1e-5)")
    print(f"  EMA decay={ema_decay}, update every {update_ema_every} steps")
    print(f"  DDIM steps={ddim_steps}, Grad clipping max_norm={grad_clip}")
    print(f"  Checkpoint every {save_every} epochs")
    print()

    # Track loss history for spike detection
    _recent_losses = []

    for epoch in range(start_epoch + 1, total_epochs + 1):
        model.train()
        ema_model.eval()
        epoch_loss = 0.0
        n_batches = 0
        _batch_losses = []

        print(f"\n{'='*60}", flush=True)
        print(f"EPOCH {epoch}/{epochs}  |  LR: {scheduler.get_last_lr()[0]:.6f}  |  step: {global_step}", flush=True)
        print(f"{'='*60}", flush=True)

        # ── GPU memory at epoch start ──────────────────────────────────
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1e9
                reserv = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  [GPU {i}] {alloc:.1f}GB alloc | {reserv:.1f}GB reserved | {total:.1f}GB total", flush=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True, dynamic_ncols=True,
                    miniters=config.get("tqdm_miniters", 4), mininterval=config.get("tqdm_mininterval", 0.5))
        for mask_batch, mri_batch in pbar:
            mask_batch = mask_batch.to(device, non_blocking=True)
            mri_batch = mri_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward + backward with AMP
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(mri_batch, mask_batch)
                if loss.ndim > 0:
                    loss = loss.mean()
                # ── Check for NaN/Inf BEFORE backward to prevent weight corruption ──
                loss_val = loss.item()
                if loss_val != loss_val or loss_val == float("inf"):
                    print(f"\n  ⚠️  [step {global_step}] NaN/Inf loss={loss_val}, skipping batch", flush=True)
                    global_step += 1
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(mri_batch, mask_batch)
                if loss.ndim > 0:
                    loss = loss.mean()
                # ── Check for NaN/Inf BEFORE backward ──
                loss_val = loss.item()
                if loss_val != loss_val or loss_val == float("inf"):
                    print(f"\n  ⚠️  [step {global_step}] NaN/Inf loss={loss_val}, skipping batch", flush=True)
                    global_step += 1
                    continue
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            n_batches += 1
            global_step += 1
            _batch_losses.append(loss_val)

            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "grad": f"{grad_norm:.2f}",
                "step": global_step,
            })

            # ── EMA update with decay schedule ────────────────────────────
            current_ema_decay = get_ema_decay(epoch)
            if global_step >= step_start_ema and global_step % update_ema_every == 0:
                with torch.no_grad():
                    for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(current_ema_decay).add_(live_p.data, alpha=1.0 - current_ema_decay)

        avg_loss = epoch_loss / n_batches
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Epoch summary ─────────────────────────────────────────────
        loss_min = min(_batch_losses)
        loss_max = max(_batch_losses)
        _recent_losses.append(avg_loss)

        current_ema_decay = get_ema_decay(epoch)

        # Detect loss spike (current epoch >2x previous avg)
        spike_warning = ""
        if len(_recent_losses) >= 2 and avg_loss > 2.0 * _recent_losses[-2]:
            spike_warning = "  ⚠️  LOSS SPIKE"

        # Detect loss plateau (last 10 epochs within 0.5%)
        plateau_warning = ""
        if len(_recent_losses) >= 10:
            recent_10 = _recent_losses[-10:]
            if (max(recent_10) - min(recent_10)) / (min(recent_10) + 1e-8) < 0.005:
                plateau_warning = "  ⚠️  PLATEAU DETECTED"

        # AMP scaler scale factor (useful to detect underflow)
        scaler_scale = f"{scaler.get_scale():.0f}" if scaler is not None else "N/A"

        # ── Validation + logging at checkpoint intervals ──────────────
        if epoch % save_every == 0 or epoch == epochs:
            val_loss = _evaluate_val_loss(val_loader, model, device)
            # val_ssim = _compute_val_ssim(ema_model, val_loader, device)  # Disabled — too slow
            epoch_record = {
                "epoch": epoch,
                "loss": avg_loss,
                "val_loss": val_loss,
                # "val_ssim": val_ssim,
                "lr": current_lr,
            }
            overfit = ""
            if val_loss > avg_loss * 1.3:
                overfit = "  ⚠️  OVERFIT?"
            print(f"\n  Loss:     avg={avg_loss:.4f}  min={loss_min:.4f}  max={loss_max:.4f}{spike_warning}", flush=True)
            print(f"  Val Loss: {val_loss:.4f}{overfit}", flush=True)
            # print(f"  Val SSIM: {val_ssim:.4f}", flush=True)
            print(f"  LR:       {current_lr:.6f}", flush=True)
            print(f"  GradNorm: {grad_norm:.3f} (clip={grad_clip})", flush=True)
            print(f"  AMP scale:{scaler_scale}", flush=True)
            print(f"  EMA active: {global_step >= step_start_ema}{plateau_warning}", flush=True)
        else:
            epoch_record = {
                "epoch": epoch,
                "loss": avg_loss,
                "lr": current_lr,
            }
            print(f"\n  Loss:     avg={avg_loss:.4f}  min={loss_min:.4f}  max={loss_max:.4f}{spike_warning}", flush=True)
            print(f"  LR:       {current_lr:.6f}", flush=True)
            print(f"  GradNorm: {grad_norm:.3f} (clip={grad_clip})", flush=True)
            print(f"  AMP scale:{scaler_scale}{plateau_warning}", flush=True)

        history.append(epoch_record)

        # ── Save checkpoint + samples ─────────────────────────────────
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = _save_checkpoint(
                model, ema_model, optimizer, scheduler, epoch, global_step, history,
                checkpoint_dir, suffix="v3_1",
            )
            _sync_to_drive(ckpt_path, drive_base)

            sample_path = _save_sample_grid(
                model, ema_model, val_loader, epoch, samples_dir, device,
                suffix="v3_1", ddim_steps=ddim_steps,
            )
            if sample_path:
                _sync_to_drive(sample_path, drive_base)

            _save_loss_plot(history, os.path.join(samples_dir, "v3_1_loss_curves.png"))
            _sync_to_drive(os.path.join(samples_dir, "v3_1_loss_curves.png"), drive_base)
            _save_metrics(history, os.path.join(metrics_dir, "v3_1_training_history.json"))
            _sync_to_drive(os.path.join(metrics_dir, "v3_1_training_history.json"), drive_base)

    print("\nDDPM v3.1 fine-tuning complete.")
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