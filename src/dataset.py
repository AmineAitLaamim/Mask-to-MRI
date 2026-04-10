"""
LGG Dataset — loading, preprocessing, augmentation, and patient-level splitting.

Each .tif image has 3 channels representing different MRI sequences
(R, G/FLAIR, B). The mask is single-channel binary.
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor
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
    """Albumentations augmentation for training data.

    Uses random jitter (resize + crop) as in the original pix2pix paper.
    Intensity augmentations (brightness/CLAHE) are reduced to avoid
    shifting normalized [-1, 1] channel means.
    """
    jitter_size = 286
    return A.Compose(
        [
            A.Resize(jitter_size, jitter_size),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                ],
                p=0.2,
            ),
            # Reduced brightness/CLAHE probability to avoid channel mean shift
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                    A.CLAHE(clip_limit=2.0, p=1.0),
                ],
                p=0.15,
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
    Optimized to cache RAW images in RAM for fast training.
    Augmentation is applied dynamically in __getitem__.

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
        cache: bool = True,
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

        # Cache RAW data in RAM if enabled
        self.cached_raw_data = []
        if cache:
            def _load_pair(p):
                return (_load_tif(p[0]), _load_tif(p[1]))

            print("    Caching raw images in RAM (Parallel Mode)...")
            with ThreadPoolExecutor(max_workers=16) as executor:
                self.cached_raw_data = list(executor.map(_load_pair, self.pairs))
            print(f"    Cached {len(self.cached_raw_data)} pairs.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        if self.cached_raw_data:
            image, mask = self.cached_raw_data[idx]
        else:
            img_path, mask_path = self.pairs[idx]
            image = _load_tif(img_path)
            mask = _load_tif(mask_path)

        # Ensure mask is 2D and binary
        mask = np.squeeze(mask)
        mask = (mask > 0).astype(np.uint8) * 255

        # Apply augmentations (always happens here, whether from cache or disk)
        transformed = self.transform(image=image, mask=mask)
        image_aug = transformed["image"]
        mask_aug = transformed["mask"]

        # Normalize to [-1, 1]
        image_norm = _normalize(image_aug)
        mask_norm = _normalize(mask_aug.astype(np.float32))

        # Convert to tensors
        mask_tensor = torch.from_numpy(mask_norm).unsqueeze(0)
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1)

        return mask_tensor, image_tensor


# ---------------------------------------------------------------------------
# Balanced Sampler — pre-builds epoch indices with exact tumor/background ratio
# ---------------------------------------------------------------------------

class BalancedSampler:
    """
    Pre-builds a shuffled list of indices for one epoch with an exact
    tumor/background ratio.  The dataset uses this list in __getitem__
    instead of calling random.choice() per-sample.

    Usage:
        indices = BalancedSampler.build_indices(n_tumor, n_bg, tumor_ratio, epoch_seed)
        dataset.set_epoch(indices)
    """

    @staticmethod
    def build_indices(n_tumor: int, n_bg: int, tumor_ratio: float, seed: int = 42) -> list[tuple[str, int]]:
        """
        Return a shuffled list of ('tumor'|'bg', index) tuples for one epoch.

        The epoch length is ceil(n_tumor / tumor_ratio), guaranteeing the
        exact tumor_ratio fraction of tumor samples.
        """
        import math

        epoch_len = math.ceil(n_tumor / tumor_ratio)
        n_tumor_epoch = math.ceil(epoch_len * tumor_ratio)
        n_bg_epoch = epoch_len - n_tumor_epoch

        # Clamp to available samples
        n_tumor_epoch = min(n_tumor_epoch, n_tumor)
        n_bg_epoch = min(n_bg_epoch, n_bg)

        # Sample with replacement if needed
        rng = random.Random(seed)
        tumor_indices = rng.choices(range(n_tumor), k=n_tumor_epoch)
        bg_indices = rng.choices(range(n_bg), k=n_bg_epoch)

        indices = [("tumor", i) for i in tumor_indices] + [("bg", i) for i in bg_indices]
        rng.shuffle(indices)
        return indices


# ---------------------------------------------------------------------------
# Balanced Dataset — samples tumor vs background at a fixed ratio
# ---------------------------------------------------------------------------

class BalancedLGGDataset(Dataset):
    """
    PyTorch Dataset that samples tumor and background slices at a fixed ratio.
    Uses BalancedSampler to pre-build epoch indices for deterministic,
    reproducible sampling without per-call random.choice().

    Each item returns:
        mask_tensor:  (1, H, W)  normalized [-1, 1]  — condition input
        image_tensor: (3, H, W)  normalized [-1, 1]  — target MRI (R, G/FLAIR, B)
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]] | None = None,
        tumor_pairs: list[tuple[str, str]] | None = None,
        background_pairs: list[tuple[str, str]] | None = None,
        image_size: int = 256,
        augment: bool = False,
        tumor_ratio: float = 0.8,
        cache: bool = True,
        seed: int = 42,
    ):
        self.tumor_ratio = tumor_ratio
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self.transform = (
            get_train_augmentation(image_size)
            if augment
            else get_val_augmentation(image_size)
        )
        self._epoch_indices: list[tuple[str, int]] | None = None
        self._epoch_seed = seed

        # If pre-separated lists are provided, use them (FAST)
        if tumor_pairs is not None and background_pairs is not None:
            self.tumor_pairs = tumor_pairs
            self.background_pairs = background_pairs
            # Always initialize cache lists (even if cache=False, for safe attribute access)
            self.cached_tumor_raw = []
            self.cached_bg_raw = []
        else:
            # Fallback: separate them from mixed list (SLOW - scans disk)
            if pairs is None:
                raise ValueError("Either 'pairs' or both 'tumor_pairs'/'background_pairs' must be provided")

            # Fallback: separate and cache in ONE pass
            self.tumor_pairs = []
            self.background_pairs = []
            # Always initialize cache lists for safe attribute access
            self.cached_tumor_raw = []
            self.cached_bg_raw = []

            print("    Scanning and caching in single pass...")
            for img_path, mask_path in pairs:
                image = _load_tif(img_path)
                mask = _load_tif(mask_path)
                if _has_tumor(mask):
                    self.tumor_pairs.append((img_path, mask_path))
                    if cache:
                        self.cached_tumor_raw.append((image, mask))
                else:
                    self.background_pairs.append((img_path, mask_path))
                    if cache:
                        self.cached_bg_raw.append((image, mask))

        print(f"    Tumor slices:      {len(self.tumor_pairs)}")
        print(f"    Background slices: {len(self.background_pairs)}")
        print(f"    Tumor ratio:       {tumor_ratio}")

        # Guards against empty lists
        if len(self.tumor_pairs) == 0:
            raise ValueError(
                "BalancedLGGDataset: tumor_pairs is empty. "
                "No tumor-containing slices found in the dataset."
            )
        if len(self.background_pairs) == 0:
            raise ValueError(
                "BalancedLGGDataset: background_pairs is empty. "
                "No background (tumor-free) slices found in the dataset."
            )

        # Cache RAW data in RAM if enabled (for pre-separated path only)
        if cache and tumor_pairs is not None and background_pairs is not None and not self.cached_tumor_raw:
            def _load_pair(p):
                return (_load_tif(p[0]), _load_tif(p[1]))

            print("    Caching raw images in RAM (Parallel Mode)...")
            with ThreadPoolExecutor(max_workers=16) as executor:
                self.cached_tumor_raw = list(executor.map(_load_pair, self.tumor_pairs))
                self.cached_bg_raw = list(executor.map(_load_pair, self.background_pairs))
            print(f"    Cached {len(self.cached_tumor_raw)} tumor + {len(self.cached_bg_raw)} bg pairs.")

    def set_epoch(self, seed: int | None = None):
        """Pre-build epoch indices using BalancedSampler for deterministic sampling."""
        if seed is None:
            self._epoch_seed += 1
            seed = self._epoch_seed
        else:
            self._epoch_seed = seed

        n_tumor = len(self.tumor_pairs)
        n_bg = len(self.background_pairs)
        self._epoch_indices = BalancedSampler.build_indices(
            n_tumor, n_bg, self.tumor_ratio, seed=seed
        )

    def __len__(self) -> int:
        if self._epoch_indices is not None:
            return len(self._epoch_indices)
        return int(len(self.tumor_pairs) / self.tumor_ratio)

    def _get_item_from_cache_or_disk(self, category: str, local_idx: int):
        """Helper to fetch (image, mask) raw arrays by category and local index."""
        if category == "tumor":
            cache = self.cached_tumor_raw
            pairs = self.tumor_pairs
        else:
            cache = self.cached_bg_raw
            pairs = self.background_pairs

        if cache:
            return cache[local_idx]

        img_path, mask_path = pairs[local_idx]
        return _load_tif(img_path), _load_tif(mask_path)

    def __getitem__(self, idx: int):
        # Use pre-built epoch indices if available
        if self._epoch_indices is not None:
            category, local_idx = self._epoch_indices[idx]
        else:
            # Fallback: random choice (legacy behavior)
            r = random.random()
            if r < self.tumor_ratio:
                category = "tumor"
                local_idx = random.randint(0, len(self.tumor_pairs) - 1)
            else:
                category = "bg"
                local_idx = random.randint(0, len(self.background_pairs) - 1)

        image, mask = self._get_item_from_cache_or_disk(category, local_idx)

        # Ensure mask is 2D and binary
        mask = np.squeeze(mask)
        mask = (mask > 0).astype(np.uint8) * 255

        # Apply augmentations
        transformed = self.transform(image=image, mask=mask)
        image_aug = transformed["image"]
        mask_aug = transformed["mask"]

        # Normalize
        image_norm = _normalize(image_aug)
        mask_norm = _normalize(mask_aug.astype(np.float32))

        # Convert to tensors
        mask_tensor = torch.from_numpy(mask_norm).unsqueeze(0)
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1)
        return mask_tensor, image_tensor


# ---------------------------------------------------------------------------
# Convenience: build DataLoaders from raw dir
# ---------------------------------------------------------------------------

def build_dataloaders(
    raw_dir: str,
    image_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 4,
    seed: int = 42,
    balanced: bool = True,
    tumor_ratio: float = 0.8,
):
    """
    Create train/val/test DataLoaders with patient-level splitting.

    Returns dict: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    import os

    patient_data = get_patient_file_list(raw_dir)
    splits = patient_level_split(patient_data, seed=seed)

    # Clamp workers to system CPU count to avoid warnings/freezes
    max_workers = os.cpu_count() or 2
    num_workers = min(num_workers, max_workers)

    use_cuda = torch.cuda.is_available()
    pin_memory = use_cuda

    loaders = {}
    for split_name, pairs in splits.items():
        augment = split_name == "train"
        if split_name == "train" and balanced:
            dataset = BalancedLGGDataset(
                pairs,
                image_size=image_size,
                augment=augment,
                tumor_ratio=tumor_ratio,
                seed=seed,
            )
            # Pre-build epoch indices for deterministic balanced sampling
            dataset.set_epoch(seed=seed)
        else:
            dataset = LGGDataset(
                pairs,
                image_size=image_size,
                augment=augment,
                seed=seed,
                filter_empty_masks=False,
            )
        print(f"  {split_name}: {len(dataset)} slices")

        is_train = split_name == "train"
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,  # Always shuffle for training (even with BalancedLGGDataset)
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None,
            drop_last=is_train,
        )
        loaders[split_name] = loader

    return loaders
