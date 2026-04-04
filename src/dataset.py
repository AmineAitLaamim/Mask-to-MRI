"""
LGG Dataset — loading, preprocessing, augmentation, and patient-level splitting.

Each .tif image has 3 channels representing different MRI sequences
(R, G/FLAIR, B). The mask is single-channel binary.
"""

import os
import random
from pathlib import Path

import albumentations as A
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _load_tif(path: str) -> np.ndarray:
    """Load a single .tif file."""
    return tifffile.imread(path)


def _has_tumor(mask: np.ndarray) -> bool:
    """Return True if the mask contains any tumor pixels."""
    return mask.max() > 0


def _normalize(image: np.ndarray) -> np.ndarray:
    """Normalize [0, 255] → [-1, 1]."""
    return (image.astype(np.float32) / 127.5) - 1.0


def _denormalize(image: np.ndarray) -> np.ndarray:
    """Convert [-1, 1] → [0, 255] for saving/visualization."""
    return ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Patient-level data extraction
# ---------------------------------------------------------------------------

def get_patient_file_list(raw_dir: str):
    """
    Scan raw_dir for patient folders and return a dict:
        {patient_id: [(img_path, mask_path), ...]}
    """
    raw_path = Path(raw_dir)
    patient_data = {}

    for patient_folder in sorted(raw_path.iterdir()):
        if not patient_folder.is_dir():
            continue
        patient_id = patient_folder.name
        pairs = []
        for mask_file in sorted(patient_folder.glob("*_mask.tif")):
            mask_name = mask_file.name  # e.g. TCGA_CS_4941_19960909_1_mask.tif
            base = mask_name.replace("_mask.tif", ".tif")
            img_path = patient_folder / base
            if img_path.exists():
                pairs.append((str(img_path), str(mask_file)))
        if pairs:
            patient_data[patient_id] = pairs

    return patient_data


def patient_level_split(
    patient_data: dict,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
):
    """
    Split patients into train/val/test at the patient level.
    Returns dict with 'train', 'val', 'test' → list of (img_path, mask_path).
    """
    patients = sorted(patient_data.keys())
    random.seed(seed)
    random.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_patients = patients[:n_train]
    val_patients = patients[n_train : n_train + n_val]
    test_patients = patients[n_train + n_val :]

    splits = {}
    for split_name, split_patients in [
        ("train", train_patients),
        ("val", val_patients),
        ("test", test_patients),
    ]:
        pairs = []
        for p in split_patients:
            pairs.extend(patient_data[p])
        splits[split_name] = pairs

    print(f"  Train: {len(train_patients)} patients, {len(splits['train'])} slices")
    print(f"  Val:   {len(val_patients)} patients, {len(splits['val'])} slices")
    print(f"  Test:  {len(test_patients)} patients, {len(splits['test'])} slices")

    return splits


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_augmentation(image_size: int = 256):
    """Albumentations augmentation for training data."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1.0),
                    A.CLAHE(p=1.0),
                ],
                p=0.3,
            ),
        ]
    )


def get_val_augmentation(image_size: int = 256):
    """Minimal augmentation for validation/test — only resize."""
    return A.Compose([A.Resize(image_size, image_size)])


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class LGGDataset(Dataset):
    """
    PyTorch Dataset for LGG mask→MRI pairs.

    Each item returns:
        mask_tensor:  (1, H, W)  normalized [-1, 1]  — condition input
        image_tensor: (3, H, W)  normalized [-1, 1]  — target MRI (R, G/FLAIR, B)
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        image_size: int = 256,
        augment: bool = False,
        seed: int = 42,
        filter_empty_masks: bool = True,
    ):
        # Filter out slices with completely black masks (no tumor)
        if filter_empty_masks:
            filtered = []
            for img_path, mask_path in pairs:
                mask = _load_tif(mask_path)
                if _has_tumor(mask):
                    filtered.append((img_path, mask_path))
            self.pairs = filtered
        else:
            self.pairs = list(pairs)

        self.image_size = image_size
        self.augment = augment
        self.transform = (
            get_train_augmentation(image_size)
            if augment
            else get_val_augmentation(image_size)
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        # Load raw data
        image = _load_tif(img_path)   # (H, W, 3) — different MRI sequences
        mask = _load_tif(mask_path)   # (H, W) or (H, W, 1)

        # Ensure mask is 2D and binary
        mask = np.squeeze(mask)
        mask = (mask > 0).astype(np.uint8) * 255

        # Apply augmentations
        transformed = self.transform(image=image, mask=mask)
        image_aug = transformed["image"]      # (H, W, 3)
        mask_aug = transformed["mask"]        # (H, W)

        # Normalize to [-1, 1]
        image_norm = _normalize(image_aug)    # (H, W, 3), float32
        mask_norm = _normalize(mask_aug.astype(np.float32))  # (H, W), float32

        # Convert to tensors: mask (1, H, W), image (3, H, W)
        mask_tensor = torch.from_numpy(mask_norm).unsqueeze(0)       # (1, H, W)
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1)  # (3, H, W)

        return mask_tensor, image_tensor


# ---------------------------------------------------------------------------
# Convenience: build DataLoaders from raw dir
# ---------------------------------------------------------------------------

def build_dataloaders(
    raw_dir: str,
    image_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    filter_empty_masks: bool = True,
):
    """
    Create train/val/test DataLoaders with patient-level splitting.

    Returns dict: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    patient_data = get_patient_file_list(raw_dir)
    splits = patient_level_split(patient_data, seed=seed)

    loaders = {}
    for split_name, pairs in splits.items():
        augment = split_name == "train"
        dataset = LGGDataset(
            pairs,
            image_size=image_size,
            augment=augment,
            seed=seed,
            filter_empty_masks=filter_empty_masks,
        )
        print(f"  {split_name}: {len(dataset)} samples after filtering")
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=augment,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )
        loaders[split_name] = loader

    return loaders
