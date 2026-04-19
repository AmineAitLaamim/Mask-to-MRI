"""Configuration for Experiment B segmentation."""

import os


def _first_existing(*paths: str) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


_IS_COLAB = os.path.exists("/content")
_IS_KAGGLE = os.path.exists("/kaggle")

if _IS_COLAB:
    _RAW_DIR = "/content/drive/MyDrive/mask-to-mri/dataset/lgg-mri-segmentation"
    _SYNTHETIC_DIR = _first_existing(
        "/content/drive/MyDrive/mask-to-mri/dataset/synthetic_data",
        "/content/drive/MyDrive/mask-to-mri/outputs_v3/synthetic",
        "/content/drive/MyDrive/mask-to-mri/outputs_v3/synthetic",
    )
    _OUTPUTS_BASE = "/content/drive/MyDrive/mask-to-mri/experiment_B"
    _DRIVE_BASE = "/content/drive/MyDrive/mask-to-mri/experiment_B"
elif _IS_KAGGLE:
    _RAW_DIR = "/kaggle/input/lgg-mri-segmentation/lgg-mri-segmentation"
    _SYNTHETIC_DIR = "/kaggle/working/outputs_v3/synthetic"
    _OUTPUTS_BASE = "/kaggle/working/experiment_B"
    _DRIVE_BASE = None
else:
    _RAW_DIR = _first_existing(
        "data/raw/lgg-mri-segmentation",
        "dataset/lgg-mri-segmentation",
    )
    _SYNTHETIC_DIR = _first_existing(
        "outputs_v3/synthetic",
        "data/synthetic",
    )
    _OUTPUTS_BASE = "experiment_B"
    _DRIVE_BASE = None


# Toggle data augmentation for train transforms.
# Default True (experiments A & B). Set to False for experiments C & D.
USE_AUGMENTATION = True

# Fraction of real tumor-containing training slices to use.
# Default 1.0 — keeps all existing experiments intact.
# Set to 0.5 for experiments D & E (half real data).
REAL_DATA_FRACTION = 1.0

# Synthetic-to-real ratio for augmented mode.
# 1 = 1:1, 2 = 1:2 synthetic-to-real.
# Default 1 — preserves existing 1:1 behavior for experiments A, B, C.
SYNTHETIC_RATIO = 1

# Train on synthetic data only (experiment F).
# When True, REAL_DATA_FRACTION and SYNTHETIC_RATIO are ignored for training.
# Val and test always use real data regardless of this flag.
# Default False — keeps all existing experiments intact.
SYNTHETIC_ONLY = False


CONFIG = {
    "raw_dir": _RAW_DIR,
    "synthetic_dir": _SYNTHETIC_DIR,
    "outputs_base": _OUTPUTS_BASE,
    "drive_base": _DRIVE_BASE,
    "image_size": 256,
    "seed": 42,
    "batch_size": 8 if (_IS_COLAB or _IS_KAGGLE) else 4,
    "num_workers": 4,
    "epochs": 100,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "scheduler_t_max": 100,
    "loss_bce_weight": 0.5,
    "loss_dice_weight": 0.5,
    "threshold": 0.5,
    "save_every": 10,
    "amp": True,
    "feature_channels": [64, 128, 256, 512],
    "baseline_run_name": "baseline",
    "augmented_run_name": "augmented",
}
