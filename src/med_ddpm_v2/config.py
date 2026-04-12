"""
Configuration for med_ddpm_v2 — 2D Conditional DDPM for Brain MRI Synthesis.

Based on: Dorjsembe et al. 2024 — Conditional Diffusion Models for Semantic 3D Brain MRI Synthesis
Adapted from 3D → 2D, using original paper's architecture.
"""

import os

# Detect Colab vs local
_IS_COLAB = os.path.exists("/content")

if _IS_COLAB:
    _OUTPUTS_BASE = "/content/Mask-to-MRI/outputs_v2"
    _DRIVE_BASE = "/content/drive/MyDrive/mask-to-mri/outputs_v2"
    _RAW_DIR = "dataset/lgg-mri-segmentation"
else:
    _OUTPUTS_BASE = "outputs_v2"
    _DRIVE_BASE = None  # Drive sync disabled on non-Colab
    _RAW_DIR = "dataset/lgg-mri-segmentation"

CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────
    "raw_dir": _RAW_DIR,
    "image_size": 256,
    "tumor_ratio": 0.8,
    "seed": 42,

    # ── Model (matches original Med-DDPM architecture, adapted 3D→2D) ─
    "num_channels": 64,          # base model channels
    "num_res_blocks": 1,         # 1 residual block per level (original)
    "in_channels": 2,            # noisy_flair(1) + mask(1)
    "out_channels": 1,           # predicted noise for FLAIR only
    "attention_resolutions": "16",  # attention at 256//16 = 16×16 spatial
    "num_heads": 4,              # attention heads (original)
    "use_scale_shift_norm": False,
    "resblock_updown": False,
    "dropout": 0.0,

    # ── Diffusion ─────────────────────────────────────────────────────
    "timesteps": 1000,           # full noise schedule (original paper)
    "beta_schedule": "cosine",   # cosine schedule (original)
    "loss_type": "l1",           # L1 loss (original — sharper samples)
    "ddim_steps": 250,           # DDIM fast sampling (250 << 1000)

    # ── Training ──────────────────────────────────────────────────────
    "epochs": 200,
    "batch_size": 4,
    "lr": 1e-4,
    "warmup_epochs": 5,
    "ema_decay": 0.995,
    "update_ema_every": 1,         # update EMA every batch (standard for diffusion)
    "step_start_ema": 2000,      # start EMA after this many steps
    "grad_clip": 1.0,
    "save_every": 10,
    "amp": True,

    # ── Paths — local outputs ─────────────────────────────────────────
    "checkpoint_dir": os.path.join(_OUTPUTS_BASE, "checkpoints"),
    "samples_dir":    os.path.join(_OUTPUTS_BASE, "samples"),
    "metrics_dir":    os.path.join(_OUTPUTS_BASE, "metrics"),
    "synthetic_dir":  os.path.join(_OUTPUTS_BASE, "synthetic"),

    # ── Paths — Google Drive mirror ───────────────────────────────────
    "drive_base": _DRIVE_BASE,
}
