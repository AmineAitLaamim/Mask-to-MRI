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
    seed: int = 42,
    balanced: bool = True,
    tumor_ratio: float = 0.8,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for DDPM training.

    Reuses build_dataloaders from src/dataset.py directly — no wrapping needed
    since LGGDataset and BalancedLGGDataset already return (mask, mri) pairs.

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    from ..dataset import build_dataloaders

    loaders = build_dataloaders(
        raw_dir=raw_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=2,
        seed=seed,
        balanced=balanced,
        tumor_ratio=tumor_ratio,
    )

    # Optimize DataLoaders for speed: pin_memory + persistent workers + prefetch
    optimized = {}
    for split, loader in loaders.items():
        optimized[split] = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=(split == "train"),
        )

    return optimized
