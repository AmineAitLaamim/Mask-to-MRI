"""
DDPM — Forward/reverse diffusion process and noise schedule.

Implements the denoising diffusion probabilistic model:
  Forward (q):  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
  Reverse (p):  x_{t-1} = 1/sqrt(alpha_t) * (x_t - sigma_t * epsilon_theta) + beta_t * z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule utilities
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear noise schedule: beta_t goes from beta_start to beta_end."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule (proposed in Nichol & Dhariwal 2021).
    Produces better sample quality than linear schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alpha_bars = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]  # Normalize
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# ---------------------------------------------------------------------------
# DDPM — Forward process + sampling
# ---------------------------------------------------------------------------

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.

    Wraps a noise-predicting U-Net and provides:
      - p_losses(): training loss (predict noise)
      - sample(): generate new images from noise conditioned on masks
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",  # linear | cosine
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Compute beta schedule
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)

        # Precompute alpha values
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Register as buffers (not trainable, moved to device automatically)
        self.register_buffer("betas", betas.to(device))
        self.register_buffer("alphas", alphas.to(device))
        self.register_buffer("alpha_bars", alpha_bars.to(device))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars).to(device))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1 - alpha_bars).to(device))

        # For sampling: posterior variance
        self.register_buffer("posterior_mean_coef1", self._compute_posterior_coef1().to(device))
        self.register_buffer("posterior_mean_coef2", self._compute_posterior_coef2().to(device))
        self.register_buffer("posterior_variance", self._compute_posterior_variance().to(device))

    def _compute_posterior_coef1(self) -> torch.Tensor:
        """sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)"""
        alpha_bar_prev = torch.cat([torch.ones(1, device=self.betas.device), self.alpha_bars[:-1]])
        return self.sqrt_alpha_bars * (1 - alpha_bar_prev) / (1 - self.alpha_bars)

    def _compute_posterior_coef2(self) -> torch.Tensor:
        """sqrt(alpha_bar_{t-1}) * (1 - alpha_t) / (1 - alpha_bar_t)"""
        alpha_bar_prev = torch.cat([torch.ones(1, device=self.betas.device), self.alpha_bars[:-1]])
        return torch.sqrt(alpha_bar_prev) * self.betas / (1 - self.alpha_bars)

    def _compute_posterior_variance(self) -> torch.Tensor:
        """Variance of q(x_{t-1} | x_t, x_0)"""
        alpha_bar_prev = torch.cat([torch.ones(1, device=self.betas.device), self.alpha_bars[:-1]])
        return self.betas * (1 - alpha_bar_prev) / (1 - self.alpha_bars)

    # ------------------------------------------------------------------
    # Forward process (q) — add noise to image
    # ------------------------------------------------------------------

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from q(x_t | x_0): add noise to x_start at timestep t.

        Args:
            x_start: (B, C, H, W) — clean image
            t: (B,) — timesteps
            noise: (B, C, H, W) — optional noise (default: random)

        Returns:
            x_noisy: (B, C, H, W) — noisy image at timestep t
            noise: the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Gather coefficients for each sample's timestep
        sqrt_ab = self._gather(self.sqrt_alpha_bars, t)
        sqrt_1mab = self._gather(self.sqrt_one_minus_alpha_bars, t)

        x_noisy = sqrt_ab * x_start + sqrt_1mab * noise
        return x_noisy, noise

    def _gather(self, arr: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Gather values from arr at indices t, reshape for broadcasting."""
        return arr[t].reshape(-1, 1, 1, 1)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        x_start: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute DDPM training loss: MSE between predicted and actual noise.

        Args:
            x_start: (B, C, H, W) — clean MRI
            mask: (B, 1, H, W) — conditioning segmentation mask
            t: (B,) — random timesteps
            noise: optional fixed noise

        Returns:
            loss: scalar MSE loss
        """
        x_noisy, noise = self.q_sample(x_start, t, noise=noise)

        # Model predicts the noise that was added
        noise_pred = self.model(x_noisy, t, mask)

        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    # Reverse process (p) — denoise step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Single reverse diffusion step: sample x_{t-1} from x_t.

        Uses the parameterized reverse:
          x_{t-1} = coef1 * x_0_pred + coef2 * x_t + sigma * z
        """
        # Predict noise
        noise_pred = self.model(x_t, t, mask)

        # Compute x_0 estimate from noise prediction
        # x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
        sqrt_ab = self._gather(self.sqrt_alpha_bars, t)
        sqrt_1mab = self._gather(self.sqrt_one_minus_alpha_bars, t)
        x_0_pred = (x_t - sqrt_1mab * noise_pred) / sqrt_ab
        x_0_pred = torch.clip(x_0_pred, -1.0, 1.0)  # Clip to valid range

        # Compute mean of posterior
        coef1 = self._gather(self.posterior_mean_coef1, t)
        coef2 = self._gather(self.posterior_mean_coef2, t)
        mean = coef1 * x_0_pred + coef2 * x_t

        # Add noise (except at t=0)
        var = self._gather(self.posterior_variance, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, 1, 1, 1)
        return mean + nonzero_mask * torch.sqrt(var) * noise

    # ------------------------------------------------------------------
    # Full sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, mask: torch.Tensor, shape: tuple[int, int, int, int] | None = None) -> torch.Tensor:
        """
        Generate images from noise conditioned on masks.

        Args:
            mask: (B, 1, H, W) — conditioning masks
            shape: optional override for output shape (B, C, H, W)

        Returns:
            (B, C, H, W) — generated MRI slices
        """
        B = mask.shape[0]
        C = self.model.in_channels if hasattr(self.model, "in_channels") else 3
        H, W = mask.shape[2], mask.shape[3]
        if shape is not None:
            B, C, H, W = shape

        # Start from pure noise
        x = torch.randn(B, C, H, W, device=mask.device)

        # Reverse diffusion: T → 0
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((B,), t, device=mask.device, dtype=torch.long)
            x = self.p_sample(x, t_batch, mask)

        return x
