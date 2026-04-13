"""
med_ddpm_v3_1 — Fine-tuning from epoch 90 checkpoint.

Lower LR (5e-5), all data, 30 epochs.
"""

from .config import CONFIG
from .model import ConditionalDDPM, GaussianDiffusion, UNetModel
from .train import train
from .sample import generate_synthetic

__all__ = [
    "CONFIG",
    "ConditionalDDPM",
    "GaussianDiffusion",
    "UNetModel",
    "train",
    "generate_synthetic",
]
