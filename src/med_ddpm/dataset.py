"""
DDPM Dataset — wraps existing LGG dataset pairs for DDPM training.

Reuses the same patient-level splits and normalization from src/dataset.py.
Returns (mask, mri) pairs compatible with the DDPM training loop.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class DDPMConditionalDataset(Dataset):
    """
    Dataset for conditional DDPM: returns (mask, mri) pairs.

    This is a thin wrapper that's compatible with the existing
    LGGDataset output format — both return normalized [-1, 1] tensors.
    """

    def __init__(self, pairs: list[tuple[str, str]], image_size: int = 256,
                 augment: bool = False, seed: int = 42,
                 filter_empty_masks: bool = True, cache: bool = True):
        """
        Initialize by delegating to LGGDataset internally.

        Args:
            pairs: list of (img_path, mask_path) tuples
            image_size: spatial resolution
            augment: whether to apply augmentation
            seed: random seed
            filter_empty_masks: skip slices with no tumor
            cache: cache raw images in RAM
        """
        # Import here to avoid circular imports
        from ..dataset import LGGDataset

        self.lgg_dataset = LGGDataset(
            pairs=pairs,
            image_size=image_size,
            augment=augment,
            seed=seed,
            filter_empty_masks=filter_empty_masks,
            cache=cache,
        )

    def __len__(self) -> int:
        return len(self.lgg_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mask: (1, H, W) — condition
            mri:  (3, H, W) — target
        """
        mask, mri = self.lgg_dataset[idx]
        return mask, mri


def build_ddpm_dataloaders(
    raw_dir: str,
    image_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 42,
    balanced: bool = True,
    tumor_ratio: float = 0.8,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for DDPM training.

    Reuses build_dataloaders from src/dataset.py but wraps datasets
    in DDPMConditionalDataset.

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    from ..dataset import build_dataloaders

    loaders = build_dataloaders(
        raw_dir=raw_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        balanced=balanced,
        tumor_ratio=tumor_ratio,
    )

    # Wrap each loader's dataset in DDPMConditionalDataset
    ddpm_loaders = {}
    for split_name, loader in loaders.items():
        dataset = loader.dataset
        ddpm_loaders[split_name] = DataLoader(
            DDPMConditionalDataset(
                pairs=dataset.pairs if hasattr(dataset, 'pairs') else [],
                image_size=image_size,
                augment=(split_name == "train"),
                seed=seed,
                filter_empty_masks=(split_name == "train"),
                cache=True,
            ),
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return ddpm_loaders
