"""
med_ddpm_v3 — 2D Conditional DDPM for Brain MRI Synthesis (Optimized).

Based on: Dorjsembe et al. 2024 — Conditional Diffusion Models for Semantic 3D Brain MRI Synthesis
Adapted from 3D → 2D, with optimizations:
  - Min-SNR weighting (gamma=5) for 3.4x faster convergence
  - Fused AdamW optimizer for 20-30% faster steps
  - Gradient checkpointing for 40-50% less VRAM
  - Optimized tqdm for less overhead
"""

from .config import CONFIG
from .model import ConditionalDDPM, GaussianDiffusion, UNetModel
from .train import train
from .sample import generate_synthetic

__all__ = [
    "CONFIG",
    "ConditionalDDPM",
    "GaussianDiffusion",
    "UNetModel",
    "train",
    "generate_synthetic",
]
