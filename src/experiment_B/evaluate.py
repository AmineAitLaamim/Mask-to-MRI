"""Evaluation helpers for Experiment B."""

from __future__ import annotations

import json
import os

import torch

from .dataset import build_experiment_b_dataloaders
from .losses import DiceBCELoss
from .metrics import compute_batch_confusion, dice_from_totals, iou_from_totals
from .model import create_unet
from .utils import sync_to_drive


@torch.no_grad()
def evaluate_experiment_b(
    config: dict,
    checkpoint_path: str,
    split: str = "test",
) -> dict:
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mode = checkpoint.get("mode", "baseline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_unet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loaders = build_experiment_b_dataloaders(config, mode=mode)
    loader = loaders[split]
    criterion = DiceBCELoss(
        bce_weight=config.get("loss_bce_weight", 0.5),
        dice_weight=config.get("loss_dice_weight", 0.5),
    )
    threshold = config.get("threshold", 0.5)

    total_loss = 0.0
    total_intersection = 0.0
    total_pred_sum = 0.0
    total_target_sum = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, masks).item())
        intersection, pred_sum, target_sum = compute_batch_confusion(
            logits, masks, threshold=threshold
        )
        total_intersection += intersection
        total_pred_sum += pred_sum
        total_target_sum += target_sum
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError(f"No batches available for split {split}")

    results = {
        "mode": mode,
        "split": split,
        "checkpoint_path": checkpoint_path,
        "loss": total_loss / n_batches,
        "dice": dice_from_totals(total_intersection, total_pred_sum, total_target_sum),
        "iou": iou_from_totals(total_intersection, total_pred_sum, total_target_sum),
    }

    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    save_path = os.path.join(metrics_dir, f"{split}_metrics.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    sync_to_drive(save_path, config["outputs_base"], config.get("drive_base"))

    results["metrics_path"] = save_path
    return results
