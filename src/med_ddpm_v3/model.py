"""
2D Conditional DDPM — adapted from original Med-DDPM (Dorjsembe et al. 2024).

Architecture:
  - U-Net noise predictor with ResBlocks, attention, timestep embeddings
  - Gaussian diffusion with cosine noise schedule
  - Mask conditioning via channel-wise concatenation: x_noisy(1) + mask(1) → 2 channels
  - Single-channel FLAIR output

Original 3D→2D changes:
  - Conv3d → Conv2d, AvgPool3d → AvgPool2d, interpolate 3D → 2D
  - Remove depth_size entirely
  - Tensor shapes: (B,C,D,H,W) → (B,C,H,W)

Diffusion math (p_losses, q_sample, p_sample, ddim_sample) is copied EXACTLY from original.
"""

import math
import copy
from abc import abstractmethod
from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Helpers (from original guided-diffusion)
# ---------------------------------------------------------------------------

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    """GroupNorm with 32 groups (original Med-DDPM standard)."""
    return GroupNorm32(32, channels)


def extract(a, t, x_shape):
    """Gather values from 1-D tensor a at indices t, reshape for broadcasting."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def timestep_embedding(timesteps, dim, max_period=1000):
    """Create sinusoidal timestep embeddings (from original)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ---------------------------------------------------------------------------
# GroupNorm32 (runs norm in float32 for stability)
# ---------------------------------------------------------------------------

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


# ---------------------------------------------------------------------------
# Checkpointing (gradient checkpointing for memory efficiency)
# ---------------------------------------------------------------------------

def checkpoint(func, inputs, params, flag):
    """Evaluate a function without caching intermediate activations."""
    if flag:
        return CheckpointFunction.apply(func, len(inputs), *inputs, *params)
    else:
        return func(*inputs)


class CheckpointFunction(Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors, ctx.input_params, output_tensors
        return (None, None) + input_grads


# ---------------------------------------------------------------------------
# Timestep blocks
# ---------------------------------------------------------------------------

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """A sequential module that passes timestep embeddings to children that support it."""
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Upsample / Downsample (2D — adapted from original 3D)
# ---------------------------------------------------------------------------

class Upsample(nn.Module):
    """Nearest-neighbor 2× upsampling + optional convolution."""
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Stride-2 convolution for downsampling."""
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# ---------------------------------------------------------------------------
# ResBlock (2D — adapted from original 3D)
# ---------------------------------------------------------------------------

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    Adapted from original Med-DDPM: Conv3d → Conv2d, removed 3D-specific code.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


# ---------------------------------------------------------------------------
# AttentionBlock (2D — adapted from original)
# ---------------------------------------------------------------------------

class QKVAttentionLegacy(nn.Module):
    """QKV attention with split-before-softmax (stable for f16)."""
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    Spatial self-attention allowing positions to attend to each other.
    Adapted from original Med-DDPM (guided-diffusion base).
    """
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


# ---------------------------------------------------------------------------
# UNetModel (2D — adapted from original Med-DDPM 3D U-Net)
# ---------------------------------------------------------------------------

class UNetModel(nn.Module):
    """
    The full 2D U-Net model with attention and timestep embedding.

    Adapted from original Med-DDPM:
      - dims=3 → dims=2 (all Conv3d → Conv2d, Pool3d → Pool2d)
      - Removed depth_size, removed 3D interpolation
      - in_channels = 2 (1 noisy FLAIR + 1 mask concatenated)
      - out_channels = 1 (predicted noise ε)

    Input: (B, 2, H, W) — noisy_flair + mask concatenated
    Output: (B, 1, H, W) — predicted noise
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # Input block — receives (noisy_mri + mask) concatenated = 4 channels
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1  # current downsampling factor

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=int(mult * model_channels),
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads,
                                       num_head_channels=num_head_channels,
                                       use_checkpoint=use_checkpoint)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch,
                                 use_scale_shift_norm=use_scale_shift_norm,
                                 use_checkpoint=use_checkpoint, down=True)
                        if resblock_updown
                        else Downsample(ch, conv_resample, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout,
                     use_scale_shift_norm=use_scale_shift_norm,
                     use_checkpoint=use_checkpoint),
            AttentionBlock(ch, num_heads=num_heads,
                           num_head_channels=num_head_channels,
                           use_checkpoint=use_checkpoint),
            ResBlock(ch, time_embed_dim, dropout,
                     use_scale_shift_norm=use_scale_shift_norm,
                     use_checkpoint=use_checkpoint),
        )
        self._feature_size += ch

        # Output blocks (upsampling path)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich, time_embed_dim, dropout,
                        out_channels=int(model_channels * mult),
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads,
                                       num_head_channels=num_head_channels,
                                       use_checkpoint=use_checkpoint)
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch,
                                 use_scale_shift_norm=use_scale_shift_norm,
                                 use_checkpoint=use_checkpoint, up=True)
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # Output head — zero-init for stable training start
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        Args:
            x: (B, 2, H, W) — noisy_flair(1) + mask(1) concatenated
            timesteps: (B,) — integer timesteps
            y: optional class labels (not used for our conditional generation)
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y is not None
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)


# ---------------------------------------------------------------------------
# GaussianDiffusion (2D — adapted from original Med-DDPM trainer.py)
#
# ALL diffusion math is copied EXACTLY from the original paper's code.
# Only changes: removed depth_size, fixed tensor shapes from 5D→4D.
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion for 2D images.

    Copied EXACTLY from original Med-DDPM (diffusion_model/trainer.py).
    Changes from original 3D:
      - Removed depth_size parameter
      - Removed x_hat terms (always 0, for future class-conditional support)
      - Sample shape: (B, C, H, W) instead of (B, C, D, H, W)
      - p_sample_loop iterates with condition_tensors injected at every step
    """

    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels,
        timesteps=1000,
        loss_type="l1",
        betas=None,
        min_snr_gamma=5,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.min_snr_gamma = min_snr_gamma

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        timesteps_val, = betas.shape
        self.num_timesteps = int(timesteps_val)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # Forward process coefficients
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        # Posterior coefficients q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer("posterior_log_variance_clipped",
                             to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1",
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2",
                             to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

    # ── Forward process (q) ──────────────────────────────────────────

    def q_sample(self, x_start, t, noise=None):
        """Add noise to x_start at timestep t: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon"""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        """Recover x_0 estimate from noisy x_t and predicted noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute posterior mean q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ── Reverse process (p) ──────────────────────────────────────────

    def p_mean_variance(self, x, t, condition_tensors=None, clip_denoised=True):
        """Compute posterior mean and variance for p(x_{t-1} | x_t)."""
        # Condition: concatenate mask with noisy image
        model_input = torch.cat([x, condition_tensors], dim=1) if condition_tensors is not None else x
        noise_pred = self.denoise_fn(model_input, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, condition_tensors=None, clip_denoised=True, repeat_noise=False):
        """Single reverse diffusion step: sample x_{t-1} from x_t."""
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, condition_tensors=condition_tensors, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors=None):
        """Full reverse diffusion loop: from noise to image."""
        device = self.betas.device
        b = shape[0]
        # Correlated noise: same spatial pattern across all channels
        # Prevents each channel from diverging independently
        noise_1ch = torch.randn(b, 1, shape[2], shape[3], device=device)
        img = noise_1ch.expand(b, shape[1], shape[2], shape[3]).clone()

        for i in tqdm_wrapper(reversed(range(0, self.num_timesteps)),
                               desc="sampling loop time step", total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, condition_tensors=condition_tensors)

        return img

    @torch.no_grad()
    def sample(self, batch_size=2, condition_tensors=None):
        """Generate images conditioned on masks."""
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size),
            condition_tensors=condition_tensors,
        )

    # ── DDIM Sampling ─────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(self, shape, condition_tensors=None, ddim_steps=50, ddim_eta=0.0):
        """
        DDIM fast sampling (Song et al. 2021).

        Adapted from original Med-DDPM's fast_sampling/guided_diffusion/gaussian_diffusion.py.
        Mask is injected at every denoising step via model_input = torch.cat([x, mask], 1).
        """
        batch, device = shape[0], self.betas.device
        total_timesteps, sampling_timesteps = self.num_timesteps, ddim_steps
        eta = ddim_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), ..., (0, -1)]

        # Correlated noise: same spatial pattern across all channels
        noise_1ch = torch.randn(shape[0], 1, shape[2], shape[3], device=device)
        img = noise_1ch.expand(shape[0], shape[1], shape[2], shape[3]).clone()

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            # Inject mask at every step
            model_input = torch.cat([img, condition_tensors], dim=1) if condition_tensors is not None else img
            noise_pred = self.denoise_fn(model_input, time_cond)

            x_start = self.predict_start_from_noise(img, t=time_cond, noise=noise_pred)
            x_start = x_start.clamp(-1.0, 1.0)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * noise_pred + sigma * noise

        return img

    # ── Training loss ─────────────────────────────────────────────────

    def p_losses(self, x_start, t, condition_tensors=None, noise=None):
        """
        Compute training loss: predict noise ε from noisy x_t.

        Min-SNR weighting (gamma=5):
          - Standard DDPM weights all timesteps equally
          - Early timesteps (high noise) dominate loss
          - Min-SNR reweights to focus on critical structure-forming timesteps
          - Reported 3.4× faster convergence (Hang et al. 2023)
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Condition: concatenate mask with noisy image before U-Net
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy_cond = torch.cat([x_noisy, condition_tensors], dim=1) if condition_tensors is not None else x_noisy
        noise_pred = self.denoise_fn(x_noisy_cond, t)

        # Per-sample L1 loss
        loss = (noise - noise_pred).abs()  # (B, C, H, W)

        # Min-SNR weighting
        gamma = getattr(self, 'min_snr_gamma', 5)
        alpha_bar_t = extract(self.alphas_cumprod, t, loss.shape)
        snr = alpha_bar_t / (1 - alpha_bar_t)  # SNR(t) = alpha_bar / (1 - alpha_bar)
        weight = torch.stack([snr, gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr
        # weight shape: (B,) → broadcast to (B, C, H, W)
        weight = weight.view(loss.shape[0], *([1] * (loss.dim() - 1)))
        loss = (loss * weight).mean()

        return loss

    def forward(self, x_start, condition_tensors=None):
        """
        Forward pass for training.

        Args:
            x_start: (B, C, H, W) — clean MRI
            condition_tensors: (B, 1, H, W) — conditioning mask

        Returns:
            loss: scalar training loss
        """
        b, c, h, w = x_start.shape
        assert h == self.image_size and w == self.image_size, \
            f"Expected {self.image_size}x{self.image_size}, got {h}x{w}"

        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        return self.p_losses(x_start, t, condition_tensors=condition_tensors)


# ---------------------------------------------------------------------------
# tqdm helper (works in both notebook and script contexts)
# ---------------------------------------------------------------------------

def tqdm_wrapper(iterable, desc="", total=None):
    """Use tqdm if available, else return iterable as-is."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total)
    except ImportError:
        return iterable


# ---------------------------------------------------------------------------
# ConditionalDDPM wrapper — high-level API
# ---------------------------------------------------------------------------

class ConditionalDDPM(nn.Module):
    """
    High-level wrapper that creates the U-Net + GaussianDiffusion.

    Usage:
        model = ConditionalDDPM(config)
        loss = model(mri, mask)       # training forward
        fake = model.sample(mask)      # generate synthetic MRI
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Build 2D U-Net — adapted from original Med-DDPM architecture
        attention_resolutions = []
        for res in config.get("attention_resolutions", "16").split(","):
            attention_resolutions.append(config["image_size"] // int(res))

        self.denoise_fn = UNetModel(
            image_size=config["image_size"],
            in_channels=config["in_channels"],   # 2 = 1 noisy_flair + 1 mask
            model_channels=config["num_channels"],
            out_channels=config["out_channels"],  # 1 = predicted noise ε
            num_res_blocks=config["num_res_blocks"],
            attention_resolutions=tuple(attention_resolutions),
            dropout=config.get("dropout", 0.0),
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            num_classes=None,
            use_checkpoint=config.get("use_checkpoint", False),
            num_heads=config.get("num_heads", 4),
            num_head_channels=-1,
            use_scale_shift_norm=config.get("use_scale_shift_norm", False),
            resblock_updown=config.get("resblock_updown", False),
        )

        # Build GaussianDiffusion — math copied exactly from original
        self.diffusion = GaussianDiffusion(
            denoise_fn=self.denoise_fn,
            image_size=config["image_size"],
            channels=config["out_channels"],  # 1
            timesteps=config["timesteps"],
            loss_type=config.get("loss_type", "l1"),
            min_snr_gamma=config.get("min_snr_gamma", 5),
        )

    def forward(self, mri: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Training forward: compute diffusion loss.

        Args:
            mri: (B, 3, H, W) — clean MRI in [-1, 1]
            mask: (B, 1, H, W) — binary segmentation mask

        Returns:
            loss: scalar L1 noise prediction loss
        """
        return self.diffusion(mri, condition_tensors=mask)

    @torch.no_grad()
    def sample(
        self,
        mask: torch.Tensor,
        batch_size: int | None = None,
        ddim_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Generate synthetic MRI conditioned on mask.

        Args:
            mask: (B, 1, H, W) — conditioning segmentation mask
            batch_size: override batch size (default: mask.shape[0])
            ddim_steps: number of DDIM steps (default: config value)

        Returns:
            (B, 3, H, W) — generated MRI in [-1, 1]
        """
        b = batch_size or mask.shape[0]
        steps = ddim_steps if ddim_steps is not None else self.config.get("ddim_steps", 250)

        shape = (b, self.config["out_channels"], self.config["image_size"], self.config["image_size"])

        # Use DDIM for speed (250 steps vs 250 full = same for config default)
        # If ddim_steps == num_timesteps, use full p_sample_loop (identical results)
        if steps >= self.diffusion.num_timesteps:
            return self.diffusion.p_sample_loop(shape, condition_tensors=mask)
        else:
            return self.diffusion.ddim_sample(
                shape, condition_tensors=mask,
                ddim_steps=steps, ddim_eta=0.0,
            )
