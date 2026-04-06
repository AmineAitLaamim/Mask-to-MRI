"""
Generator — U-Net encoder-decoder with skip connections and instance normalization.

Takes a 1-channel binary mask → produces a 3-channel MRI slice.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Convolution → Normalization → LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, norm_layer: nn.Module, use_relu: bool = True, stride: int = 2, padding: int = 1):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=padding, bias=False)]
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if use_relu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransposeConvBlock(nn.Module):
    """Transposed convolution → Normalization → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, norm_layer: nn.Module, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# U-Net Generator
# ---------------------------------------------------------------------------

class UNetGenerator(nn.Module):
    """
    U-Net generator for pix2pix.

    Encoder: 4 downsampling blocks
    Bottleneck: 1 block
    Decoder: 4 upsampling blocks with skip connections
    Output: ConvTranspose2d → Tanh

    Args:
        in_channels:  Number of input channels (mask = 1)
        out_channels: Number of output channels (MRI = 3)
        num_filters:  Base number of filters (default 64)
        norm:         'instance' or 'batch' normalization
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        num_filters: int = 64,
        norm: str = "instance",
    ):
        super().__init__()

        NormLayer = self._get_norm_layer(norm)

        # ----- Encoder (downsampling path) -----
        # No norm on the first layer (standard pix2pix)
        self.enc1 = ConvBlock(in_channels, num_filters, norm_layer=None, use_relu=True)        # 64
        self.enc2 = ConvBlock(num_filters, num_filters * 2, norm_layer=NormLayer)               # 128
        self.enc3 = ConvBlock(num_filters * 2, num_filters * 4, norm_layer=NormLayer)           # 256
        self.enc4 = ConvBlock(num_filters * 4, num_filters * 8, norm_layer=NormLayer)           # 512

        # ----- Bottleneck -----
        # Use kernel=3, stride=1, padding=1 to preserve spatial dimensions
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=3, stride=1, padding=1, bias=False),
            NormLayer(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ----- Decoder (upsampling path) with skip connections -----
        # dec4 input = bottleneck(512) + enc4(512) = 1024  ← key fix
        self.dec4 = TransposeConvBlock(num_filters * 8 * 2, num_filters * 4, norm_layer=NormLayer, use_dropout=True)   # 1024 → 256, 16→32
        self.dec3 = TransposeConvBlock(num_filters * 4 * 2, num_filters * 2, norm_layer=NormLayer, use_dropout=True)   # 512  → 128, 32→64
        self.dec2 = TransposeConvBlock(num_filters * 2 * 2, num_filters,     norm_layer=NormLayer, use_dropout=False)  # 256  → 64,  64→128
        self.dec1 = TransposeConvBlock(num_filters * 2,     num_filters,     norm_layer=NormLayer, use_dropout=False)  # 128  → 64,  128→256

        # ----- Final output layer (no upsampling — already at 256×256) -----
        self.final = nn.Sequential(
            nn.Conv2d(num_filters, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def _get_norm_layer(norm: str) -> type[nn.Module]:
        if norm == "instance":
            return nn.InstanceNorm2d
        elif norm == "batch":
            return nn.BatchNorm2d
        else:
            raise ValueError(f"Unknown norm: {norm}. Use 'instance' or 'batch'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connections.

        Args:
            x: Input mask tensor of shape (B, 1, 256, 256)

        Returns:
            Output MRI tensor of shape (B, 3, 256, 256)
        """
        # Encoder
        e1 = self.enc1(x)   # (B, 64, 128, 128)
        e2 = self.enc2(e1)  # (B, 128, 64, 64)
        e3 = self.enc3(e2)  # (B, 256, 32, 32)
        e4 = self.enc4(e3)  # (B, 512, 16, 16)

        # Bottleneck (stride=1, no downsampling)
        b = self.bottleneck(e4)  # (B, 512, 16, 16)

        # Decoder — concatenate before each block (correct order)
        d4 = self.dec4(torch.cat([b,  e4], dim=1))  # 512+512 → 256, 32×32
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # 256+256 → 128, 64×64
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # 128+128 → 64,  128×128
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # 64+64   → 64,  256×256

        return self.final(d1)  # (B, out_ch, 256, 256)


# ---------------------------------------------------------------------------
# Weight initialization (from original pix2pix paper)
# ---------------------------------------------------------------------------

def weights_init_normal(m: nn.Module):
    """
    Initialize convolutional weights from N(0, 0.02).
    InstanceNorm gamma → N(1, 0.02), beta → 0.
    """
    classname = m.__class__.__name__
    if classname.startswith("Conv"):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    elif "Norm" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_generator(in_channels: int = 1, out_channels: int = 3,
                     num_filters: int = 64, norm: str = "instance") -> UNetGenerator:
    """Create and initialize the U-Net generator."""
    model = UNetGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        num_filters=num_filters,
        norm=norm,
    )
    model.apply(weights_init_normal)
    return model
