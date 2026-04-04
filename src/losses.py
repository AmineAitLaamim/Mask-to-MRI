"""
Loss functions for pix2pix GAN training and downstream segmentation evaluation.

1. LSGAN discriminator loss (MSE variant)
2. Generator adversarial + L1 pixel loss
3. Dice + BCE combined loss (for Experiment B segmentation U-Net)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LSGAN losses (MSE-based)
# ---------------------------------------------------------------------------

def discriminator_loss_real(d_pred_real: torch.Tensor) -> torch.Tensor:
    """Discriminator loss on real pairs: MSE(D(mask, real_MRI), 1.0)."""
    target = torch.ones_like(d_pred_real)
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
    Combined generator loss: adversarial + lambda_l1 * L1.

    Usage inside training loop:
        loss_G, loss_G_adv, loss_G_L1 = gan_loss(d_pred_fake, fake_MRI, real_MRI)
    """

    def __init__(self, lambda_l1: float = 100.0):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.l1 = L1Loss()

    def forward(self, d_pred_fake: torch.Tensor,
                fake: torch.Tensor, real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss_G: total generator loss
            loss_G_adv: adversarial component
            loss_G_L1: L1 pixel component
        """
        loss_G_adv = generator_adversarial_loss(d_pred_fake)
        loss_G_L1 = self.l1(fake, real)
        loss_G = loss_G_adv + self.lambda_l1 * loss_G_L1
        return loss_G, loss_G_adv, loss_G_L1


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
