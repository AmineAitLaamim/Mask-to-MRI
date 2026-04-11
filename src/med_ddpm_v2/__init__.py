"""
med_ddpm_v2 — 2D Conditional DDPM for Brain MRI Synthesis.

Based on: Dorjsembe et al. 2024 — Conditional Diffusion Models for Semantic 3D Brain MRI Synthesis
Adapted from 3D → 2D, using original paper's architecture.
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
