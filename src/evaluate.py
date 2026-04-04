"""
Evaluation metrics for Experiment A (GAN quality) and Experiment B (segmentation).

Experiment A:
  - FID (Fréchet Inception Distance)
  - SSIM (Structural Similarity Index)
  - PSNR (Peak Signal-to-Noise Ratio)

Experiment B:
  - Dice score comparison (real-only vs real+synthetic)
"""

import json
import os
from datetime import datetime

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def compute_ssim(fake: np.ndarray, real: np.ndarray) -> float:
    """
    Compute SSIM between two images.

    Args:
        fake: (H, W, 3) uint8
        real: (H, W, 3) uint8

    Returns:
        SSIM score (0–1, higher is better). Multi-channel average.
    """
    # Compute per-channel SSIM and average
    scores = []
    for c in range(fake.shape[-1]):
        s = ssim(real[:, :, c], fake[:, :, c], data_range=255)
        scores.append(s)
    return float(np.mean(scores))


def compute_ssim_batch(fake_batch: torch.Tensor, real_batch: torch.Tensor) -> float:
    """
    Compute mean SSIM over a batch of images.

    Args:
        fake_batch: (B, 3, H, W) normalized [-1, 1]
        real_batch: (B, 3, H, W) normalized [-1, 1]

    Returns:
        Mean SSIM score.
    """
    # Denormalize
    fake_np = ((fake_batch.cpu().permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    real_np = ((real_batch.cpu().permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    scores = []
    for i in range(fake_np.shape[0]):
        scores.append(compute_ssim(fake_np[i], real_np[i]))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(fake: np.ndarray, real: np.ndarray) -> float:
    """Compute PSNR between two images (in dB, higher is better)."""
    return float(psnr(real, fake, data_range=255))


def compute_psnr_batch(fake_batch: torch.Tensor, real_batch: torch.Tensor) -> float:
    """Compute mean PSNR over a batch."""
    fake_np = ((fake_batch.cpu().permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    real_np = ((real_batch.cpu().permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    scores = []
    for i in range(fake_np.shape[0]):
        scores.append(compute_psnr(fake_np[i], real_np[i]))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# FID (Fréchet Inception Distance)
# ---------------------------------------------------------------------------

def compute_fid_from_paths(real_dir: str, fake_dir: str, device: str = "cpu") -> float:
    """
    Compute FID between two directories of images.

    Uses pytorch-fid under the hood.

    Args:
        real_dir: Path to directory with real MRI images (.png)
        fake_dir: Path to directory with fake MRI images (.png)

    Returns:
        FID score (lower is better).
    """
    from pytorch_fid import fid_score

    fid = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=1,
        device=device,
        dims=2048,
        num_workers=0,
    )
    return float(fid)


# ---------------------------------------------------------------------------
# Dice score (Experiment B — segmentation)
# ---------------------------------------------------------------------------

def compute_dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """
    Compute Dice score between prediction and target binary masks.

    dice = (2 * |pred ∩ target|) / (|pred| + |target|)

    Args:
        pred: Binary mask (H, W) or (B, H, W) — values 0 or 1
        target: Binary mask same shape — values 0 or 1

    Returns:
        Dice score (0–1, 1 = perfect overlap).
    """
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return float(dice)


# ---------------------------------------------------------------------------
# Save evaluation results as timestamped JSON
# ---------------------------------------------------------------------------

def save_eval_results(
    metrics: dict,
    metrics_dir: str = "outputs/metrics",
    prefix: str = "eval",
) -> str:
    """
    Save evaluation metrics as a timestamped JSON file.

    Args:
        metrics: Dict of metric name → value
        metrics_dir: Output directory
        prefix: Filename prefix (default: "eval")

    Returns:
        Path to saved JSON file.
    """
    os.makedirs(metrics_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics["timestamp"] = datetime.now().isoformat()

    filename = f"{prefix}_{timestamp}.json"
    path = os.path.join(metrics_dir, filename)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  → Saved evaluation results: {path}")
    return path
