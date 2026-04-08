"""
DDPM Sampling/Inference — generate MRI images from segmentation masks.

Usage:
    Given a trained DDPM model and segmentation masks,
    generate synthetic MRI slices via the reverse diffusion process.
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .unet import ConditionalUNet
from .diffusion import DDPM
from .train import EMA, load_ddpm_checkpoint


@torch.no_grad()
def generate_from_masks(
    masks: torch.Tensor,
    model: ConditionalUNet,
    ddpm: DDPM,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate MRI images from segmentation masks using DDPM.

    Args:
        masks: (B, 1, H, W) — binary segmentation masks
        model: Trained conditional U-Net
        ddpm: DDPM wrapper with noise schedule
        device: torch device

    Returns:
        (B, 3, H, W) — generated MRI slices in [-1, 1]
    """
    model.eval()
    masks = masks.to(device)

    # Sample from DDPM
    generated = ddpm.sample(masks)
    return generated


def generate_and_save(
    mask_loader: DataLoader,
    model: ConditionalUNet,
    ddpm: DDPM,
    device: torch.device,
    output_dir: str = "data/synthetic",
    n_samples: int | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """
    Generate MRI images from a DataLoader of masks and save them.

    Args:
        mask_loader: DataLoader returning (mask, _) tuples
        model: Trained conditional U-Net
        ddpm: DDPM wrapper
        device: torch device
        output_dir: directory to save generated images
        n_samples: optional limit on number of samples

    Returns:
        List of (filename, tensor) tuples
    """
    import numpy as np
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    results = []
    count = 0

    pbar = tqdm(mask_loader, desc="Generating synthetic MRI")
    for mask_batch, _ in pbar:
        mask_batch = mask_batch.to(device)

        # Generate
        fake_batch = ddpm.sample(mask_batch)

        # Save each sample
        for i in range(fake_batch.shape[0]):
            if n_samples is not None and count >= n_samples:
                break

            # Denormalize and convert to uint8
            fake_np = ((fake_batch[i].cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            filename = f"synthetic_mri_{count:04d}.png"
            path = os.path.join(output_dir, filename)
            Image.fromarray(fake_np).save(path)
            results.append((path, fake_batch[i]))
            count += 1

        if n_samples is not None and count >= n_samples:
            break

    print(f"  → Generated {count} synthetic MRI images in {output_dir}")
    return results


def load_model_for_sampling(
    checkpoint_path: str,
    model: ConditionalUNet,
    device: torch.device = torch.device("cpu"),
    use_ema: bool = True,
) -> tuple[ConditionalUNet, EMA | None]:
    """
    Load a trained model for sampling.

    Args:
        checkpoint_path: path to .pt checkpoint
        model: U-Net model instance
        device: torch device
        use_ema: if True, apply EMA weights for better quality

    Returns:
        (model, ema) — model ready for sampling, optional EMA object
    """
    ema = EMA(model) if use_ema else None
    epoch, _ = load_ddpm_checkpoint(checkpoint_path, model, ema=ema, device=device)

    if use_ema and ema is not None:
        # Apply EMA shadow to model for sampling
        ema.apply_shadow()

    model.to(device)
    model.eval()

    print(f"  → Loaded DDPM checkpoint from epoch {epoch}")
    if use_ema:
        print("  → Using EMA weights for sampling")

    return model, ema
