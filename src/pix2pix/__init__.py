"""
pix2pix — Conditional GAN for MRI synthesis from segmentation masks.

This package contains the generator, discriminator, loss functions, training loop,
evaluation metrics, and utilities for the pix2pix conditional GAN that synthesizes
brain MRI slices from tumor segmentation masks.
"""

from .generator import UNetGenerator, create_generator, weights_init_normal
from .discriminator import PatchGANDiscriminator, create_discriminator
from .losses import (
    GANLoss,
    PerceptualLoss,
    L1Loss,
    DiceLoss,
    DiceBCELoss,
    discriminator_loss_real,
    discriminator_loss_fake,
    generator_adversarial_loss,
)
from .train import train, load_checkpoint, find_latest_checkpoint, load_metrics
from .evaluate import (
    compute_ssim,
    compute_ssim_batch,
    compute_psnr,
    compute_psnr_batch,
    compute_fid_from_paths,
    compute_dice_score,
    save_eval_results,
)
from .utils import (
    load_config,
    fix_seed,
    setup_logger,
    get_device,
    count_parameters,
    print_model_summary,
    make_sample_grid,
    plot_loss_curves,
    plot_metrics_from_file,
)

__all__ = [
    # Generator
    "UNetGenerator",
    "create_generator",
    "weights_init_normal",
    # Discriminator
    "PatchGANDiscriminator",
    "create_discriminator",
    # Losses
    "GANLoss",
    "PerceptualLoss",
    "L1Loss",
    "DiceLoss",
    "DiceBCELoss",
    "discriminator_loss_real",
    "discriminator_loss_fake",
    "generator_adversarial_loss",
    # Training
    "train",
    "load_checkpoint",
    "find_latest_checkpoint",
    "load_metrics",
    # Evaluation
    "compute_ssim",
    "compute_ssim_batch",
    "compute_psnr",
    "compute_psnr_batch",
    "compute_fid_from_paths",
    "compute_dice_score",
    "save_eval_results",
    # Utilities
    "load_config",
    "fix_seed",
    "setup_logger",
    "get_device",
    "count_parameters",
    "print_model_summary",
    "make_sample_grid",
    "plot_loss_curves",
    "plot_metrics_from_file",
]
