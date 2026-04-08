"""
Noise-predicting U-Net for DDPM.

Key differences from pix2pix generator:
  - Sinusoidal timestep embeddings injected into each block
  - GroupNorm (not InstanceNorm) — DDPM standard
  - Optional self-attention at specified resolutions
  - Conditioning via mask concatenation at input
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for diffusion timesteps.
    Produces a (dim,) vector for each timestep, similar to transformer PE.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) tensor of timesteps (integers 0..T-1)
        Returns:
            (B, dim) timestep embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        # Log-spaced frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]  # (B, half_dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Conv → GroupNorm → SiLU → Conv → GroupNorm → SiLU + residual connection."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, norm_groups: int = 8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(norm_groups, out_ch),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(norm_groups, out_ch),
            nn.SiLU(),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        # Add timestep info: broadcast (B, C) → (B, C, 1, 1)
        t_emb = self.time_mlp(t)[:, :, None, None]
        h = h + t_emb
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention at a specified spatial resolution."""

    def __init__(self, dim: int, norm_groups: int = 8, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.norm = nn.GroupNorm(norm_groups, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.out = nn.Conv2d(dim, dim, 1)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)  # (B, C, H, W) each
        # Reshape for multi-head: (B, num_heads, head_dim, HW)
        q = q.reshape(B, self.num_heads, self.head_dim, -1)
        k = k.reshape(B, self.num_heads, self.head_dim, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, -1)
        # Attention: (B, heads, HW, HW)
        attn = (q.transpose(2, 3) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v.transpose(2, 3)).transpose(2, 3)  # (B, heads, head_dim, HW)
        out = out.reshape(B, C, H, W)
        return self.out(out) + x


class Downsample(nn.Module):
    """2×2 average pool + conv."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Upsample(nn.Module):
    """Nearest-neighbor upsample + conv."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Conditional U-Net for DDPM
# ---------------------------------------------------------------------------

class ConditionalUNet(nn.Module):
    """
    Noise-predicting U-Net for conditional DDPM.

    Input: (noisy_image, mask) concatenated → (out_ch + cond_ch, H, W)
    Output: predicted noise ε of shape (out_ch, H, W)

    Architecture:
      - 4-level U-Net with residual blocks + timestep embeddings
      - Optional attention at specified resolutions
      - GroupNorm throughout
    """

    def __init__(
        self,
        in_channels: int = 3,        # MRI channels
        cond_channels: int = 1,      # mask channels
        num_filters: int = 64,
        time_emb_dim: int = 256,
        norm_groups: int = 8,
        attention_resolutions: list[int] | None = None,
        num_heads: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.attention_resolutions = set(attention_resolutions or [])
        self.num_heads = num_heads

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(num_filters),
            nn.Linear(num_filters, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input: concatenate noisy image + condition mask
        total_in = in_channels + cond_channels

        # Initial conv
        self.init_conv = nn.Conv2d(total_in, num_filters, 3, padding=1)

        # Downsampling path
        self.down1 = Block(num_filters, num_filters, time_emb_dim, norm_groups)
        self.down2 = Block(num_filters, num_filters * 2, time_emb_dim, norm_groups)
        self.down3 = Block(num_filters * 2, num_filters * 4, time_emb_dim, norm_groups)
        self.down4 = Block(num_filters * 4, num_filters * 8, time_emb_dim, norm_groups)

        # Attention at bottleneck
        self.attn = AttentionBlock(num_filters * 8, norm_groups, num_heads) if 16 in self.attention_resolutions else nn.Identity()

        # Middle
        self.mid = Block(num_filters * 8, num_filters * 8, time_emb_dim, norm_groups)

        # Upsampling path
        self.up1 = Block(num_filters * 8 * 2, num_filters * 4, time_emb_dim, norm_groups)
        self.up2 = Block(num_filters * 4 * 2, num_filters * 2, time_emb_dim, norm_groups)
        self.up3 = Block(num_filters * 2 * 2, num_filters, time_emb_dim, norm_groups)
        self.up4 = Block(num_filters * 2, num_filters, time_emb_dim, norm_groups)

        # Downsampling/Upsampling operators
        self.ds1 = Downsample(num_filters)
        self.ds2 = Downsample(num_filters * 2)
        self.ds3 = Downsample(num_filters * 4)
        # Upsamplers must match the OUTPUT channels of their corresponding blocks
        self.us1 = Upsample(num_filters * 4)   # 256 ← output of up1
        self.us2 = Upsample(num_filters * 2)   # 128 ← output of up2
        self.us3 = Upsample(num_filters)        # 64  ← output of up3

        # Final conv
        self.final = nn.Sequential(
            nn.GroupNorm(norm_groups, num_filters),
            nn.SiLU(),
            nn.Conv2d(num_filters, in_channels, 1),  # Predict noise for in_channels
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy image (B, in_ch, H, W)
            t: Timestep (B,) — integer timesteps 0..T-1
            mask: Conditioning mask (B, cond_ch, H, W)

        Returns:
            Predicted noise (B, in_ch, H, W)
        """
        # Concatenate condition
        h = torch.cat([x, mask], dim=1)
        h = self.init_conv(h)

        # Timestep embedding
        t_emb = self.time_embed(t)

        # Downsampling
        d1 = self.down1(h, t_emb)       # 64,  256²
        d2 = self.down2(self.ds1(d1), t_emb)  # 128, 128²
        d3 = self.down3(self.ds2(d2), t_emb)  # 256, 64²
        d4 = self.down4(self.ds3(d3), t_emb)  # 512, 32²

        # Bottleneck
        b = self.attn(d4)
        b = self.mid(b, t_emb)

        # Upsampling with skip connections — concat at same resolution, then upsample
        u1 = self.us1(self.up1(torch.cat([b, d4], dim=1), t_emb))  # 256, 64²
        u2 = self.us2(self.up2(torch.cat([u1, d3], dim=1), t_emb))  # 128, 128²
        u3 = self.us3(self.up3(torch.cat([u2, d2], dim=1), t_emb))  # 64,  256²
        u4 = self.up4(torch.cat([u3, d1], dim=1), t_emb)             # 64,  256²

        return self.final(u4)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_unet(
    in_channels: int = 3,
    cond_channels: int = 1,
    num_filters: int = 64,
    time_emb_dim: int = 256,
    norm: str = "group",
    attention_resolutions: list[int] | None = None,
    num_heads: int = 1,
) -> ConditionalUNet:
    """Create conditional U-Net for DDPM."""
    norm_groups = 8 if norm == "group" else 1
    return ConditionalUNet(
        in_channels=in_channels,
        cond_channels=cond_channels,
        num_filters=num_filters,
        time_emb_dim=time_emb_dim,
        norm_groups=norm_groups,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
    )
