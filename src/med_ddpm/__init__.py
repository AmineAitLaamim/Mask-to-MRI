"""
med_ddpm — Medical Denoising Diffusion Probabilistic Model for MRI synthesis.

Conditional DDPM that generates brain MRI slices from tumor segmentation masks.
Alternative to pix2pix GAN with more stable training and better mode coverage.
"""

from .unet import ConditionalUNet, create_unet, SinusoidalTimeEmbedding
from .diffusion import DDPM, linear_beta_schedule, cosine_beta_schedule
from .dataset import DDPMConditionalDataset, build_ddpm_dataloaders
from .train import (
    train,
    EMA,
    find_latest_ddpm_checkpoint,
    load_ddpm_checkpoint,
    load_ddpm_metrics,
)
from .sample import generate_from_masks, generate_and_save, load_model_for_sampling

__all__ = [
    # U-Net
    "ConditionalUNet",
    "create_unet",
    "SinusoidalTimeEmbedding",
    # DDPM
    "DDPM",
    "linear_beta_schedule",
    "cosine_beta_schedule",
    # Dataset
    "DDPMConditionalDataset",
    "build_ddpm_dataloaders",
    # Training
    "train",
    "EMA",
    "find_latest_ddpm_checkpoint",
    "load_ddpm_checkpoint",
    "load_ddpm_metrics",
    # Sampling
    "generate_from_masks",
    "generate_and_save",
    "load_model_for_sampling",
]
