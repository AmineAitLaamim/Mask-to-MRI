"""
DDPM Dataset — reuses LGG/BalancedLGG datasets directly for DDPM training.

No wrapping needed: both LGGDataset and BalancedLGGDataset already return
(mask, mri) tuples compatible with the DDPM training loop.
"""

import torch
from torch.utils.data import DataLoader


def build_ddpm_dataloaders(
    raw_dir: str,
    image_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 2,
    seed: int = 42,
    balanced: bool = True,
    tumor_ratio: float = 0.8,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for DDPM training.

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    import os

    # Clamp workers to system CPU count to avoid Colab warnings/freezes
    max_workers = os.cpu_count() or 2
    num_workers = min(num_workers, max_workers)

    from ..dataset import build_dataloaders

    return build_dataloaders(
        raw_dir=raw_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        balanced=balanced,
        tumor_ratio=tumor_ratio,
    )
