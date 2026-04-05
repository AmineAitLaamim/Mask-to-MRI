"""
Loss functions for pix2pix GAN training and downstream segmentation evaluation.

1. LSGAN discriminator loss (MSE variant)
2. Generator adversarial + L1 pixel loss
3. Dice + BCE combined loss (for Experiment B segmentation U-Net)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ---------------------------------------------------------------------------
# LSGAN losses (MSE-based)
# ---------------------------------------------------------------------------

def discriminator_loss_real(d_pred_real: torch.Tensor) -> torch.Tensor:
    """Discriminator loss on real pairs: MSE(D(mask, real_MRI), 0.9)."""
    target = torch.ones_like(d_pred_real) * 0.9  # label smoothing
    return F.mse_loss(d_pred_real, target)


def discriminator_loss_fake(d_pred_fake: torch.Tensor) -> torch.Tensor:
    """Discriminator loss on fake pairs: MSE(D(mask, fake_MRI), 0.0)."""
    target = torch.zeros_like(d_pred_fake)
    return F.mse_loss(d_pred_fake, target)


def generator_adversarial_loss(d_pred_fake: torch.Tensor) -> torch.Tensor:
    """Generator adversarial loss: MSE(D(mask, fake_MRI), 1.0) — fool the discriminator."""
    target = torch.ones_like(d_pred_fake)
    return F.mse_loss(d_pred_fake, target)


# ---------------------------------------------------------------------------
# Perceptual loss (VGG19 features)
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG19 features.

    Compares L1 distance between VGG19 feature maps of fake and real images.
    Handles denormalization from [-1,1] → [0,1] → ImageNet-normalized.
    """

    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
        self.features = vgg.features[:18]  # up to relu3_2
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fake: (B, 3, H, W) in [-1, 1]
            real: (B, 3, H, W) in [-1, 1]

        Returns:
            L1 loss between VGG feature maps.
        """
        # VGG19 is moved to GPU once during GANLoss initialization in train.py

        # Denormalize: [-1, 1] → [0, 1]
        fake_denorm = (fake + 1.0) / 2.0
        real_denorm = (real + 1.0) / 2.0
        # ImageNet normalization (buffers are already on correct device)
        fake_norm = (fake_denorm - self.mean) / self.std
        real_norm = (real_denorm - self.mean) / self.std

        fake_feats = self.features(fake_norm)
        real_feats = self.features(real_norm)
        return F.l1_loss(fake_feats, real_feats)


# ---------------------------------------------------------------------------
# L1 pixel loss
# ---------------------------------------------------------------------------

class L1Loss(nn.Module):
    """L1 pixel-level loss between fake and real MRI."""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.loss(fake, real)


# ---------------------------------------------------------------------------
# Combined GAN loss (used inside training loop)
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """
    Combined generator loss: adversarial + lambda_l1 * L1 + lambda_perceptual * Perceptual.

    Usage inside training loop:
        loss_G, loss_G_adv, loss_G_L1, loss_perc = gan_loss(d_pred_fake, fake_MRI, real_MRI)
    """

    def __init__(self, lambda_l1: float = 10.0, lambda_perceptual: float = 10.0, use_perceptual: bool = True):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual = use_perceptual
        self.l1 = L1Loss()
        self.perceptual = PerceptualLoss() if use_perceptual else None

    def forward(self, d_pred_fake: torch.Tensor,
                fake: torch.Tensor, real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss_G: total generator loss
            loss_G_adv: adversarial component
            loss_G_L1: L1 pixel component
            loss_perceptual: perceptual (VGG feature) component
        """
        loss_G_adv = generator_adversarial_loss(d_pred_fake)
        loss_G_L1 = self.l1(fake, real)

        if self.use_perceptual and self.perceptual is not None:
            loss_perceptual = self.perceptual(fake, real)
        else:
            loss_perceptual = torch.tensor(0.0, device=fake.device)

        loss_G = loss_G_adv + self.lambda_l1 * loss_G_L1 + self.lambda_perceptual * loss_perceptual
        return loss_G, loss_G_adv, loss_G_L1, loss_perceptual


# ---------------------------------------------------------------------------
# Dice + BCE loss (Experiment B — segmentation U-Net)
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    dice = 1 - (2 * intersection + smooth) / (sum_pred + sum_target + smooth)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dimensions
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss for segmentation.

    loss = 0.5 * BCELoss(pred, target) + 0.5 * DiceLoss(pred, target)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.bce(pred, target) + 0.5 * self.dice(pred, target)
