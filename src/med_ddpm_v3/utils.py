"""Shared utilities for med_ddpm_v3."""

import shutil
from pathlib import Path


def _sync_to_drive(local_path: str, drive_base: str | None) -> None:
    """Copy a file from local outputs_v3 to Google Drive mirror."""
    if drive_base is None:
        return  # Not on Colab — skip silently
    try:
        outputs_base = "/content/Mask-to-MRI/outputs_v3"
        rel = Path(local_path).relative_to(outputs_base)
        drive_path = Path(drive_base) / rel
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, drive_path)
    except Exception as e:
        print(f"  Drive sync failed: {e}")
