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

        # Posterior coefficients (following original Med-DDPM: mobaidoctor/med-ddpm)
        # coef1 (for x_0): beta_t * sqrt(alpha_bar_{t-1}) / (1 - alpha_bar_t)
        # coef2 (for x_t): (1 - alpha_bar_{t-1}) * sqrt(alpha_t) / (1 - alpha_bar_t)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bars[:-1]]).to(device)
        self.register_buffer("posterior_mean_coef1", (betas.to(device) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bars.to(device))))
        self.register_buffer("posterior_mean_coef2", ((1 - alpha_bar_prev) * torch.sqrt(alphas.to(device)) / (1 - alpha_bars.to(device))))
        self.register_buffer("posterior_variance", self._compute_posterior_variance().to(device))

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

        # L1 loss produces sharper samples (matches original Med-DDPM)
        return F.l1_loss(noise_pred, noise)

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

    # ------------------------------------------------------------------
    # Debug: verify noise schedule
    # ------------------------------------------------------------------

    def verify_noise_schedule(self, x_start: torch.Tensor) -> dict:
        """
        Verify that the noise schedule is correct.
        Returns dict with std values at t=0, t=500, t=999.
        Expected: ~0.3, ~0.7, ~1.0 (for normalized data)
        """
        with torch.no_grad():
            noise = torch.randn_like(x_start)
            B = x_start.shape[0]

            x_t0, _ = self.q_sample(x_start, torch.zeros(B, dtype=torch.long, device=x_start.device), noise=noise)
            x_t500, _ = self.q_sample(x_start, torch.full((B,), 500, dtype=torch.long, device=x_start.device), noise=noise)
            x_t999, _ = self.q_sample(x_start, torch.full((B,), 999, dtype=torch.long, device=x_start.device), noise=noise)

        result = {
            "t=0 std": x_t0.std().item(),
            "t=500 std": x_t500.std().item(),
            "t=999 std": x_t999.std().item(),
        }
        print("  → Noise schedule verification:")
        for k, v in result.items():
            print(f"     {k}: {v:.4f}")
        return result


# ---------------------------------------------------------------------------
# DDIM Sampling — fast generation in 50-100 steps instead of 1000
# ---------------------------------------------------------------------------

class DDIMSampler:
    """
    Denoising Diffusion Implicit Models (DDIM) sampler.

    DDIM produces deterministic samples (no random noise added during
    reverse diffusion) and can use far fewer steps than the training
    timesteps (e.g. 50-100 instead of 1000).

    Based on: Song et al. "Denoising Diffusion Implicit Models" (2021)
    """

    def __init__(self, ddpm: DDPM, ddim_steps: int = 50, eta: float = 0.0):
        """
        Args:
            ddpm: Trained DDPM model (provides the noise schedule)
            ddim_steps: Number of denoising steps (default 50)
            eta: Stochasticity parameter (0 = fully deterministic DDIM,
                 1 = equivalent to DDPM). Default 0 for speed.
        """
        self.ddpm = ddpm
        self.ddim_steps = ddim_steps
        self.eta = eta

        # Subsample timesteps: uniform spacing over [0, T-1]
        T = ddpm.timesteps
        c = T // ddim_steps
        self.ddim_timesteps = torch.arange(0, T, c, device=ddpm.betas.device)

        # Precompute DDIM coefficients
        # alpha_bar for selected timesteps
        self.alpha_bars = ddpm.alpha_bars[self.ddim_timesteps]

        # For DDIM, at step i we go from t_i to t_{i-1} (previous DDIM timestep)
        # alpha_bar_prev[i] = alpha_bar at the PREVIOUS DDIM timestep
        # For i=0 (final step, t→0): we use alpha_bar[-1] = 1.0 as a placeholder
        #   (but direction coefficient will be zero anyway)
        alpha_bars_shifted = torch.cat([self.alpha_bars[1:], torch.ones(1, device=ddpm.betas.device)])
        # alpha_bars_shifted[i] = alpha_bars[i+1] for i < N-1, 1.0 for i = N-1
        # We need the OPPOSITE direction: alpha_bar_prev[i] = alpha_bar at timestep BEFORE current
        # i.e. alpha_bar_prev[i] = alpha_bars[i-1] for i>0, and 1.0 for i=0
        alpha_bar_prev_arr = torch.cat([torch.ones(1, device=ddpm.betas.device), self.alpha_bars[:-1]])
        self.alpha_bars_prev = alpha_bar_prev_arr

        # sigma (variance) — when eta=0, sigma=0 (deterministic)
        alpha_t = ddpm.alphas[self.ddim_timesteps]
        variance = (1 - self.alpha_bars_prev) / (1 - self.alpha_bars) * (1 - alpha_t)
        self.sigma = eta * torch.sqrt(torch.clamp(variance, min=1e-20))

    @torch.no_grad()
    def sample(self, mask: torch.Tensor, shape: tuple[int, int, int, int] | None = None) -> torch.Tensor:
        """
        Generate images using DDIM fast sampling.

        Args:
            mask: (B, 1, H, W) — conditioning masks
            shape: optional override for output shape (B, C, H, W)

        Returns:
            (B, C, H, W) — generated MRI slices
        """
        B = mask.shape[0]
        C = self.ddpm.model.in_channels if hasattr(self.ddpm.model, "in_channels") else 3
        H, W = mask.shape[2], mask.shape[3]
        if shape is not None:
            B, C, H, W = shape

        # Start from pure noise
        x = torch.randn(B, C, H, W, device=mask.device)

        # Reverse diffusion: step through the subsampled timesteps
        for i in reversed(range(self.ddim_steps)):
            t = self.ddim_timesteps[i]
            t_batch = torch.full((B,), t.item(), device=mask.device, dtype=torch.long)

            # Predict noise
            noise_pred = self.ddpm.model(x, t_batch, mask)

            # Get current and previous alpha_bar
            alpha_bar_t = self.alpha_bars[i]
            alpha_bar_prev = self.alpha_bars_prev[i]
            sigma = self.sigma[i]

            # Compute x_0 estimate from noise prediction
            sqrt_ab = torch.sqrt(alpha_bar_t)
            sqrt_1mab = torch.sqrt(1 - alpha_bar_t)
            x_0_pred = (x - sqrt_1mab * noise_pred) / sqrt_ab
            x_0_pred = torch.clip(x_0_pred, -1.0, 1.0)

            # Compute direction pointing to x_t
            direction = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * noise_pred

            # Add stochastic noise (only if eta > 0)
            if self.eta > 0 and i > 0:
                noise = torch.randn_like(x)
                stochastic = sigma * noise
            else:
                stochastic = 0

            # Compute x_{t-1}
            x = torch.sqrt(alpha_bar_prev) * x_0_pred + direction + stochastic

        return x
