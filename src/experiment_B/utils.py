"""Utility helpers for Experiment B."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def sync_to_drive(local_path: str, outputs_base: str, drive_base: str | None) -> None:
    """Mirror a file or directory from local Colab outputs to Drive."""
    if drive_base is None:
        return

    try:
        local = Path(local_path)
        rel = local.relative_to(outputs_base)
        drive_path = Path(drive_base) / rel
        if local.resolve() == drive_path.resolve():
            return
        if local.is_dir():
            shutil.copytree(local, drive_path, dirs_exist_ok=True)
        else:
            drive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, drive_path)
    except Exception as exc:
        print(f"  Drive sync failed: {exc}")
