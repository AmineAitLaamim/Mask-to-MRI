"""
Configuration for med_ddpm_v3 — 2D Conditional DDPM for Brain MRI Synthesis.

Optimized from med_ddpm_v2 with:
  - Min-SNR weighting (gamma=5) for 3.4× faster convergence
  - Fused AdamW optimizer for 20-30% faster steps
  - Gradient checkpointing for 40-50% less VRAM
  - Optimized tqdm for less overhead
"""

import os

# Detect environment
_IS_COLAB  = os.path.exists("/content")
_IS_KAGGLE = os.path.exists("/kaggle")

if _IS_COLAB:
    _OUTPUTS_BASE = "/content/Mask-to-MRI/outputs_v3"
    _DRIVE_BASE   = "/content/drive/MyDrive/mask-to-mri/outputs_v3"
    _RAW_DIR      = "dataset/lgg-mri-segmentation"
    _BATCH_SIZE   = 8   # safe for Colab T4 (16 GB) with checkpointing
elif _IS_KAGGLE:
    _OUTPUTS_BASE = "/kaggle/working/outputs_v3"
    _DRIVE_BASE   = None
    _RAW_DIR      = "/kaggle/input/lgg-mri-segmentation/lgg-mri-segmentation"
    _BATCH_SIZE   = 8
else:
    _OUTPUTS_BASE = "outputs_v3"
    _DRIVE_BASE   = None
    _RAW_DIR      = "dataset/lgg-mri-segmentation"
    _BATCH_SIZE   = 4

CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────
    "raw_dir": _RAW_DIR,
    "image_size": 256,
    "tumor_ratio": 0.8,
    "seed": 42,

    # ── Model (U-Net noise predictor, 3D→2D adapted) ─────────────────
    "num_channels": 64,
    "num_res_blocks": 1,
    "in_channels": 2,            # noisy_flair(1) + mask(1)
    "out_channels": 1,           # predicted noise for FLAIR
    "attention_resolutions": "16",
    "num_heads": 4,
    "use_scale_shift_norm": False,
    "resblock_updown": False,
    "dropout": 0.1,             # NEW: prevents overfitting on small dataset
    "use_checkpoint": False,       # Disabled — batch=8 fits in T4 VRAM without it

    # ── Diffusion ─────────────────────────────────────────────────────
    "timesteps": 1000,
    "beta_schedule": "cosine",
    "loss_type": "l1",
    "ddim_steps": 250,

    # ── NEW: Min-SNR Weighting ────────────────────────────────────────
    "min_snr_gamma": 5,           # gamma=5 → 3.4× faster convergence

    # ── Training ──────────────────────────────────────────────────────
    "epochs": 200,
    "batch_size": _BATCH_SIZE,
    "lr": 1e-4,
    "warmup_epochs": 5,
    "ema_decay": 0.995,
    "ema_decay_start": 0.9,       # NEW: ramp EMA decay from 0.9 → 0.995
    "ema_decay_ramp_epochs": 50,  # NEW: epochs to ramp EMA decay
    "update_ema_every": 1,
    "step_start_ema": 2000,
    "grad_clip": 1.0,
    "cfg_drop_prob": 0.1,         # NEW: classifier-free guidance mask dropout
    "save_every": 10,
    "amp": True,
    "fused_optimizer": True,      # NEW: fused AdamW (20-30% faster)

    # ── tqdm optimization ─────────────────────────────────────────────
    "tqdm_miniters": 4,           # NEW: update every 4 batches
    "tqdm_mininterval": 0.5,      # NEW: minimum 0.5s between updates

    # ── Paths ─────────────────────────────────────────────────────────
    "checkpoint_dir": os.path.join(_OUTPUTS_BASE, "checkpoints"),
    "samples_dir":    os.path.join(_OUTPUTS_BASE, "samples"),
    "metrics_dir":    os.path.join(_OUTPUTS_BASE, "metrics"),
    "synthetic_dir":  os.path.join(_OUTPUTS_BASE, "synthetic"),
    "drive_base":     _DRIVE_BASE,
}
