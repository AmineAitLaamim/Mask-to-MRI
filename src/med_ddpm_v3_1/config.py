"""
Configuration for med_ddpm_v3_1 — Fine-tuning from epoch 90 checkpoint.

Changes from v3:
  - Lower LR (5e-5) for fine-tuning
  - No warmup (already at good weights)
  - All data used (train + val + test)
  - Fewer epochs (fine-tuning, not from scratch)
"""

import os

_IS_COLAB  = os.path.exists("/content")
_IS_KAGGLE = os.path.exists("/kaggle")

if _IS_COLAB:
    _OUTPUTS_BASE = "/content/Mask-to-MRI/outputs_v3_1"
    _DRIVE_BASE   = "/content/drive/MyDrive/mask-to-mri/outputs_v3_1"
    _RAW_DIR      = "dataset/lgg-mri-segmentation"
    _BATCH_SIZE   = 8
    _V3_CHECKPOINT = "/content/drive/MyDrive/mask-to-mri/outputs_v3/checkpoints/checkpoint_v3_epoch_90.pt"
elif _IS_KAGGLE:
    _OUTPUTS_BASE = "/kaggle/working/outputs_v3_1"
    _DRIVE_BASE   = None
    _RAW_DIR      = "/kaggle/input/lgg-mri-segmentation/lgg-mri-segmentation"
    _BATCH_SIZE   = 8
    _V3_CHECKPOINT = None
else:
    _OUTPUTS_BASE = "outputs_v3_1"
    _DRIVE_BASE   = None
    _RAW_DIR      = "dataset/lgg-mri-segmentation"
    _BATCH_SIZE   = 4
    _V3_CHECKPOINT = None

CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────
    "raw_dir": _RAW_DIR,
    "image_size": 256,
    "tumor_ratio": 0.8,            # Keep 80/20 balanced
    "seed": 42,

    # ── Model (same as v3) ───────────────────────────────────────────
    "num_channels": 64,
    "num_res_blocks": 1,
    "in_channels": 2,
    "out_channels": 1,
    "attention_resolutions": "16,32",
    "num_heads": 4,
    "use_scale_shift_norm": False,
    "resblock_updown": False,
    "dropout": 0.15,             # Slightly higher for fine-tuning
    "use_checkpoint": False,

    # ── Diffusion ─────────────────────────────────────────────────────
    "timesteps": 1000,
    "beta_schedule": "cosine",
    "loss_type": "l1",
    "ddim_steps": 250,
    "min_snr_gamma": 5,

    # ── Fine-tuning Training ──────────────────────────────────────────
    "epochs": 30,                # Fine-tune for 30 more epochs
    "batch_size": _BATCH_SIZE,
    "lr": 5e-5,                  # Half the original LR
    "warmup_epochs": 0,          # No warmup when fine-tuning
    "ema_decay": 0.999,          # Higher EMA for smoother tracking
    "ema_decay_start": 0.99,
    "ema_decay_ramp_epochs": 10,
    "update_ema_every": 1,
    "step_start_ema": 50,        # Start EMA quickly
    "grad_clip": 1.0,
    "cfg_drop_prob": 0.1,
    "save_every": 5,
    "amp": True,
    "fused_optimizer": True,

    # ── tqdm ──────────────────────────────────────────────────────────
    "tqdm_miniters": 4,
    "tqdm_mininterval": 0.5,

    # ── Paths ─────────────────────────────────────────────────────────
    "checkpoint_dir": os.path.join(_OUTPUTS_BASE, "checkpoints"),
    "samples_dir":    os.path.join(_OUTPUTS_BASE, "samples"),
    "metrics_dir":    os.path.join(_OUTPUTS_BASE, "metrics"),
    "synthetic_dir":  os.path.join(_OUTPUTS_BASE, "synthetic"),
    "drive_base":     _DRIVE_BASE,

    # ── Resume checkpoint (epoch 90 from v3) ──────────────────────────
    "resume_from": _V3_CHECKPOINT,
}
