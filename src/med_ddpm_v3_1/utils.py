"""v3_1 utilities: merge all splits into one training dataset."""

import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from src.dataset import (
    get_patient_file_list,
    patient_level_split,
    BalancedLGGDataset,
    LGGDataset,
    FLAIRDataset,
)


def _sync_to_drive(local_path: str, drive_base: str | None) -> None:
    """Copy a file from local outputs_v3_1 to Google Drive mirror."""
    if drive_base is None:
        return
    try:
        outputs_base = "/content/Mask-to-MRI/outputs_v3_1"
        rel = Path(local_path).relative_to(outputs_base)
        drive_path = Path(drive_base) / rel
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, drive_path)
    except Exception as e:
        print(f"  Drive sync failed: {e}")


def build_all_data_flair_loader(
    raw_dir: str,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    tumor_ratio: float = 0.8,
    seed: int = 42,
) -> DataLoader:
    """
    Combine train + val + test into ONE balanced training loader.
    All data is merged, then balanced 80/20 sampling is applied.
    """
    import os

    patient_data = get_patient_file_list(raw_dir)
    splits = patient_level_split(patient_data, seed=seed)

    # Merge all splits
    all_pairs = []
    for pairs in splits.values():
        all_pairs.extend(pairs)

    max_workers = os.cpu_count() or 2
    num_workers = min(num_workers, max_workers)
    use_cuda = torch.cuda.is_available()
    pin_memory = use_cuda

    # Balanced dataset on ALL data
    base_dataset = BalancedLGGDataset(
        all_pairs,
        image_size=image_size,
        augment=True,
        tumor_ratio=tumor_ratio,
        seed=seed,
    )
    base_dataset.set_epoch(seed=seed)

    dataset = FLAIRDataset(base_dataset)
    total = len(base_dataset)
    tumor_count = len(base_dataset.tumor_pairs)
    bg_count = len(base_dataset.background_pairs)
    print(f"  All data merged: {len(all_pairs)} total pairs ({tumor_count} tumor, {bg_count} bg)")
    print(f"  Balanced epoch length: {total} samples ({tumor_ratio*100:.0f}% tumor)")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )
    return loader
