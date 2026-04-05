"""
Discriminator — PatchGAN (70×70 receptive field).

Always sees the pair (mask + MRI) concatenated together.
Input: (mask, MRI) → (1 + 3, 256, 256) = (4, 256, 256)
Output: (1, 30, 30) grid of real/fake scores per 70×70 patch.
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator.

    5 convolutional layers that progressively downsample to a 70×70
    effective receptive field per output neuron.

    Architecture (no norm on first layer, standard pix2pix):
        (4, 256, 256) → 64  → (64,  128, 128)
        (64, 128, 128) → 128 → (128, 64, 64)
        (128, 64, 64) → 256  → (256, 32, 32)
        (256, 32, 32) → 512  → (512, 31, 31)   ← stride=1
        (512, 31, 31) → 1    → (1,  30, 30)    ← patch-level scores
    """

    def __init__(
        self,
        in_channels: int = 4,   # 1 (mask) + 3 (MRI) = 4
        num_filters: int = 64,
    ):
        super().__init__()

        # Layer 1 — no activation after spectral norm, LeakyReLU separate
        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 2 — spectral norm replaces InstanceNorm2d
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 3 — spectral norm replaces InstanceNorm2d
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 4 — spectral norm replaces InstanceNorm2d
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Layer 5 — spectral norm, output patch map
        self.layer5 = nn.utils.spectral_norm(nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask:  (B, 1, H, W)  binary mask
            image: (B, 3, H, W)  MRI image (real or fake)

        Returns:
            Patch scores: (B, 1, 30, 30)
        """
        x = torch.cat([mask, image], dim=1)  # (B, 4, H, W)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


def create_discriminator(in_channels: int = 4, num_filters: int = 64) -> PatchGANDiscriminator:
    """Create and initialize the PatchGAN discriminator."""
    model = PatchGANDiscriminator(in_channels=in_channels, num_filters=num_filters)
    # Use same initialization as generator
    from .generator import weights_init_normal
    model.apply(weights_init_normal)
    return model
