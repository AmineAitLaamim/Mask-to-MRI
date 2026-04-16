"""Standard 2D U-Net for Experiment B tumor segmentation."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegmentationUNet(nn.Module):
    """Single-channel FLAIR to single-channel mask segmentation U-Net."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | tuple[int, ...] = (64, 128, 256, 512),
    ):
        super().__init__()
        features = list(features)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        current_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(current_in, feature))
            current_in = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        current_in = features[-1] * 2
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(current_in, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            current_in = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]

            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


def create_unet(config: dict | None = None) -> SegmentationUNet:
    config = config or {}
    return SegmentationUNet(
        in_channels=1,
        out_channels=1,
        features=config.get("feature_channels", [64, 128, 256, 512]),
    )
