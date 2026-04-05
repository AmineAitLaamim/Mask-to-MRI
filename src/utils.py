"""
Utilities — seeding, config loading, logging, and visualization.
"""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Seed fixing for reproducibility
# ---------------------------------------------------------------------------

def fix_seed(seed: int = 42):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"  → Random seed fixed: {seed}")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(name: str = "mask-to-mri", log_file: str | None = None):
    """
    Set up a logger that prints to both console and optionally a file.

    Args:
        name: Logger name
        log_file: Optional path to log file

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Select the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"  → Device: {device}")
    return device


# ---------------------------------------------------------------------------
# Count model parameters
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(name: str, model: torch.nn.Module):
    """Print a one-line model summary."""
    n_params = count_parameters(model)
    print(f"  {name}: {n_params:,} parameters ({n_params/1e6:.2f}M)")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def make_sample_grid(
    masks: torch.Tensor,
    fake_images: torch.Tensor,
    real_images: torch.Tensor,
    n_cols: int = 3,
    n_rows: int = 4,
) -> np.ndarray:
    """
    Create a visual grid: mask | fake | real, side by side.

    Args:
        masks: (B, 1, H, W) normalized [-1, 1]
        fake_images: (B, 3, H, W) normalized [-1, 1]
        real_images: (B, 3, H, W) normalized [-1, 1]
        n_cols: Number of columns per sample (always 3: mask|fake|real)
        n_rows: Number of samples to show

    Returns:
        Grid as numpy array (uint8, RGB)
    """
    import numpy as np
    from PIL import Image

    n = min(n_rows, masks.shape[0])

    rows = []
    for i in range(n):
        # Denormalize
        mask_np = ((masks[i, 0].cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        fake_np = ((fake_images[i].cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        real_np = ((real_images[i].cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        # Convert mask to 3-channel
        mask_3ch = np.stack([mask_np] * 3, axis=-1)

        row = np.concatenate([mask_3ch, fake_np, real_np], axis=1)
        rows.append(row)

    return np.concatenate(rows, axis=0)


def plot_loss_curves(history: list[dict], save_path: str | None = None):
    """
    Plot training loss curves (D, G, G_adv, G_L1, Perceptual) over epochs.

    Args:
        history: List of per-epoch loss dicts from train()
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt

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

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved loss curves: {save_path}")

    plt.show()
