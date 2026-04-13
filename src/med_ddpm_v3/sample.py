"""
Generate synthetic MRIs from training masks using trained DDPM v3 model.

Usage:
    generate_synthetic(
        checkpoint_path="outputs_v3/checkpoints/checkpoint_v3_epoch_200.pt",
        output_dir="outputs_v3/synthetic",
        raw_dir="dataset/lgg-mri-segmentation",
        config=CONFIG,
    )

Generates one synthetic MRI per tumor-containing training mask.
"""

import os
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

from .model import ConditionalDDPM
from .utils import _sync_to_drive


# ---------------------------------------------------------------------------
# Training masks dataset (only tumor-containing slices)
# ---------------------------------------------------------------------------

class TrainingMasksDataset(Dataset):
    """
    Load ONLY training split masks that contain tumors.
    Reuses the same patient-level split logic as src.dataset.
    """

    def __init__(self, raw_dir: str, image_size: int = 256, seed: int = 42):
        from src.dataset import get_patient_file_list, patient_level_split, LGGDataset

        patient_data = get_patient_file_list(raw_dir)
        splits = patient_level_split(patient_data, seed=seed)
        train_pairs = splits["train"]  # list of (img_path, mask_path) tuples

        # Create a temporary dataset to load masks
        self.dataset = LGGDataset(train_pairs, image_size=image_size, augment=False)
        self.pairs = []  # store (mask_tensor, stem)
        self.stems = []

        # Collect only tumor-containing masks
        for idx in range(len(self.dataset)):
            mask, mri = self.dataset[idx]
            if (mask > 0).any():  # Has tumor
                self.pairs.append(mask)
                _, mask_path = train_pairs[idx]
                stem = Path(mask_path).stem.replace("_mask", "")
                self.stems.append(stem)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx], self.stems[idx]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: str,
    config: dict,
    device: torch.device,
) -> tuple[ConditionalDDPM, dict]:
    """
    Load a ConditionalDDPM model from a checkpoint.

    Uses EMA model weights if available (better quality).

    Returns:
        (model, checkpoint_data)
    """
    model = ConditionalDDPM(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Prefer EMA weights if available (better sample quality)
    if "ema_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_state_dict"])
        print(f"  Loaded EMA weights from checkpoint (epoch {checkpoint['epoch']})")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded model weights from checkpoint (epoch {checkpoint['epoch']})")

    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_synthetic(
    checkpoint_path: str,
    output_dir: str,
    config: dict,
    raw_dir: str | None = None,
    ddim_steps: int | None = None,
):
    """
    Generate synthetic MRIs from all tumor-containing training masks.

    Args:
        checkpoint_path: Path to trained checkpoint
        output_dir: Directory to save synthetic images
        config: Configuration dict
        raw_dir: Raw dataset directory (for loading training masks)
        ddim_steps: Number of DDIM sampling steps (default: from config)
    """
    raw_dir = raw_dir or config["raw_dir"]
    ddim_steps = ddim_steps or config.get("ddim_steps", 250)
    drive_base = config.get("drive_base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, ckpt = load_model_from_checkpoint(checkpoint_path, config, device)

    # Load training masks
    print("Loading training masks...")
    masks_dataset = TrainingMasksDataset(raw_dir, image_size=config["image_size"], seed=config["seed"])
    print(f"  Found {len(masks_dataset)} tumor-containing training masks")

    # Generate synthetic MRIs
    print(f"Generating synthetic MRIs (DDIM {ddim_steps} steps)...")
    count = 0

    for idx in tqdm(range(len(masks_dataset)), desc="Generating"):
        mask, stem = masks_dataset[idx]
        mask = mask.unsqueeze(0).to(device)  # (1, 1, H, W)

        with torch.no_grad():
            fake = model.sample(mask, ddim_steps=ddim_steps)

        # Denormalize: [-1,1] → [0,255] — single channel grayscale
        fake_np = ((fake[0, 0].cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        mask_np = ((mask[0, 0].cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        # Save synthetic FLAIR as grayscale (single channel)
        fake_gray_path = os.path.join(output_dir, f"{stem}_synthetic_flair.png")
        Image.fromarray(fake_np, mode='L').save(fake_gray_path)

        # Save as 3-channel RGB (FLAIR copied to all channels) for experiment B compatibility
        fake_rgb = np.stack([fake_np, fake_np, fake_np], axis=-1)
        fake_rgb_path = os.path.join(output_dir, f"{stem}_synthetic.png")
        Image.fromarray(fake_rgb).save(fake_rgb_path)

        # Save mask
        mask_path = os.path.join(output_dir, f"{stem}_mask.png")
        Image.fromarray(mask_np.astype(np.uint8)).save(mask_path)

        count += 1

    # Batch sync entire directory to Drive (much faster than per-file)
    if drive_base is not None:
        print("Syncing synthetic images to Drive (one-time batch copy)...")
        import shutil
        drive_synthetic = Path(drive_base) / "synthetic"
        drive_synthetic.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(output_dir, drive_synthetic, dirs_exist_ok=True)
            print(f"  Synced {count} image pairs to Drive")
        except Exception as e:
            print(f"  Drive sync failed: {e}")

    print(f"\nGenerated {count} synthetic MRI pairs to {output_dir}")
    print(f"  → {count} synthetic images")
    print(f"  → {count} mask images")
