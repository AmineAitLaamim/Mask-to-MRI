"""Datasets and dataloaders for Experiment B segmentation."""

from __future__ import annotations

from pathlib import Path
import random

import albumentations as A
import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.dataset import get_patient_file_list, patient_level_split, _load_tif


def _build_transform(image_size: int, augment: bool) -> A.Compose:
    ops: list[A.BasicTransform] = [A.Resize(image_size, image_size)]
    if augment:
        ops.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )
    return A.Compose(ops)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) / 127.5) - 1.0


def _binarize_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.float32)


def _filter_tumor_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    filtered = []
    for image_path, mask_path in pairs:
        mask = _load_tif(mask_path)
        if np.squeeze(mask).max() > 0:
            filtered.append((image_path, mask_path))
    return filtered


class RealFLAIRSegmentationDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], image_size: int = 256, augment: bool = False):
        self.pairs = list(pairs)
        self.transform = _build_transform(image_size=image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        image = tifffile.imread(image_path)
        mask = tifffile.imread(mask_path)

        flair = image[..., 1]
        mask = np.squeeze(mask)
        mask = (_binarize_mask(mask) * 255).astype(np.uint8)

        transformed = self.transform(image=flair, mask=mask)
        image = _normalize_image(transformed["image"])
        mask = _binarize_mask(transformed["mask"])

        image_t = torch.from_numpy(image).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return image_t, mask_t


class SyntheticPNGSegmentationDataset(Dataset):
    """Synthetic grayscale FLAIR PNGs paired with mask PNGs."""

    def __init__(self, synthetic_dir: str, image_size: int = 256, augment: bool = False):
        self.synthetic_dir = Path(synthetic_dir)
        self.transform = _build_transform(image_size=image_size, augment=augment)
        self.pairs = self._discover_pairs()
        if not self.pairs:
            raise ValueError(
                f"No synthetic image/mask pairs found in {self.synthetic_dir}. "
                "Expected names like synthetic_0001.png + synthetic_0001_mask.png."
            )

    def _discover_pairs(self) -> list[tuple[str, str]]:
        if not self.synthetic_dir.exists():
            return []

        pairs: list[tuple[str, str]] = []
        for image_path in sorted(self.synthetic_dir.glob("*.png")):
            name = image_path.name
            if name.endswith("_mask.png"):
                continue

            mask_candidate = None
            stem = image_path.stem
            candidates = [
                image_path.with_name(f"{stem}_mask.png"),
            ]
            if stem.endswith("_synthetic"):
                base = stem[: -len("_synthetic")]
                candidates.append(image_path.with_name(f"{base}_mask.png"))

            for candidate in candidates:
                if candidate.exists():
                    mask_candidate = candidate
                    break

            if mask_candidate is not None:
                pairs.append((str(image_path), str(mask_candidate)))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Failed to load synthetic pair: {image_path}, {mask_path}")

        transformed = self.transform(image=image, mask=mask)
        image = _normalize_image(transformed["image"])
        mask = _binarize_mask(transformed["mask"])

        image_t = torch.from_numpy(image).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return image_t, mask_t


def _create_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=drop_last,
    )


def build_experiment_b_dataloaders(config: dict, mode: str = "baseline") -> dict[str, DataLoader]:
    raw_dir = config["raw_dir"]
    image_size = config.get("image_size", 256)
    batch_size = config.get("batch_size", 4)
    num_workers = config.get("num_workers", 4)
    seed = config.get("seed", 42)

    patient_data = get_patient_file_list(raw_dir)
    splits = patient_level_split(patient_data, seed=seed)
    splits = {name: _filter_tumor_pairs(pairs) for name, pairs in splits.items()}

    train_dataset = RealFLAIRSegmentationDataset(
        splits["train"], image_size=image_size, augment=True
    )
    val_dataset = RealFLAIRSegmentationDataset(
        splits["val"], image_size=image_size, augment=False
    )
    test_dataset = RealFLAIRSegmentationDataset(
        splits["test"], image_size=image_size, augment=False
    )

    if mode == "augmented":
        synthetic_dataset = SyntheticPNGSegmentationDataset(
            config["synthetic_dir"], image_size=image_size, augment=True
        )
        target_count = len(train_dataset)
        if len(synthetic_dataset) < target_count:
            raise ValueError(
                f"Synthetic dataset too small for 1:1 ratio: found {len(synthetic_dataset)} pairs, "
                f"need at least {target_count} to match the real tumor-containing train set."
            )
        rng = random.Random(seed)
        chosen = rng.sample(range(len(synthetic_dataset)), k=target_count)
        synthetic_dataset = torch.utils.data.Subset(synthetic_dataset, chosen)
        train_dataset = ConcatDataset([train_dataset, synthetic_dataset])
    elif mode != "baseline":
        raise ValueError(f"Unsupported mode: {mode}")

    return {
        "train": _create_loader(
            train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True
        ),
        "val": _create_loader(
            val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
        ),
        "test": _create_loader(
            test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
        ),
    }
