"""
pix2pix — Conditional GAN for MRI synthesis from segmentation masks.

This package contains the generator, discriminator, loss functions, and training loop
for the pix2pix conditional GAN that synthesizes brain MRI slices from tumor
segmentation masks.
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
]
