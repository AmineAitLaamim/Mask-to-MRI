"""Training loop for Experiment B segmentation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm

from .dataset import build_experiment_b_dataloaders
from .losses import DiceBCELoss
from .metrics import compute_batch_confusion, dice_from_totals, iou_from_totals
from .model import create_unet
from .utils import sync_to_drive


def _prepare_run_dirs(base_dir: str, run_name: str) -> dict[str, str]:
    run_dir = Path(base_dir) / run_name
    dirs = {
        "run_dir": str(run_dir),
        "checkpoints": str(run_dir / "checkpoints"),
        "metrics": str(run_dir / "metrics"),
        "plots": str(run_dir / "plots"),
        "samples": str(run_dir / "samples"),
    }
    for path in dirs.values():
        if path.endswith("_dir"):
            continue
    for key in ["checkpoints", "metrics", "plots", "samples"]:
        os.makedirs(dirs[key], exist_ok=True)
    return dirs


@torch.no_grad()
def _evaluate_loader(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_intersection = 0.0
    total_pred_sum = 0.0
    total_target_sum = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += float(loss.item())
        intersection, pred_sum, target_sum = compute_batch_confusion(
            logits, masks, threshold=threshold
        )
        total_intersection += intersection
        total_pred_sum += pred_sum
        total_target_sum += target_sum
        n_batches += 1

    if n_batches == 0:
        return {"loss": float("inf"), "dice": 0.0, "iou": 0.0}

    return {
        "loss": total_loss / n_batches,
        "dice": dice_from_totals(total_intersection, total_pred_sum, total_target_sum),
        "iou": iou_from_totals(total_intersection, total_pred_sum, total_target_sum),
    }


def _save_history(history: list[dict], metrics_path: str) -> None:
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _save_plots(history: list[dict], plots_dir: str) -> None:
    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [row["train_loss"] for row in history], label="Train Loss")
    ax.plot(epochs, [row["val_loss"] for row in history], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss_curves.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [row["val_dice"] for row in history], label="Val Dice")
    ax.plot(epochs, [row["val_iou"] for row in history], label="Val IoU")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "val_metrics.png"), dpi=120)
    plt.close(fig)


def train_experiment_b(config: dict, mode: str = "baseline") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = config["baseline_run_name"] if mode == "baseline" else config["augmented_run_name"]
    run_dirs = _prepare_run_dirs(config["outputs_base"], run_name)
    drive_base = config.get("drive_base")

    loaders = build_experiment_b_dataloaders(config, mode=mode)
    model = create_unet(config).to(device)
    criterion = DiceBCELoss(
        bce_weight=config.get("loss_bce_weight", 0.5),
        dice_weight=config.get("loss_dice_weight", 0.5),
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get("scheduler_t_max", config["epochs"]),
    )
    threshold = config.get("threshold", 0.5)
    use_amp = device.type == "cuda" and config.get("amp", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history: list[dict] = []
    best_val_dice = -1.0
    best_checkpoint_path = os.path.join(run_dirs["checkpoints"], "best.pt")
    start_epoch = 0

    resume_from = config.get("resume_from")
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint.get("epoch", 0))
        if history:
            best_val_dice = max(row.get("val_dice", -1.0) for row in history)
        print(f"Resumed {mode} training from epoch {start_epoch}: {resume_from}")

    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        pbar = tqdm(loaders["train"], desc=f"{mode} epoch {epoch}/{config['epochs']}")

        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

            loss_val = float(loss.item())
            train_loss += loss_val
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        scheduler.step()
        train_loss = train_loss / max(n_batches, 1)
        val_metrics = _evaluate_loader(
            model, loaders["val"], criterion, device, threshold=threshold
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_record)

        checkpoint = {
            "epoch": epoch,
            "mode": mode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "config": config,
        }
        if epoch % config.get("save_every", 5) == 0 or epoch == config["epochs"]:
            epoch_ckpt_path = os.path.join(run_dirs["checkpoints"], f"epoch_{epoch:03d}.pt")
            torch.save(checkpoint, epoch_ckpt_path)
            sync_to_drive(epoch_ckpt_path, config["outputs_base"], drive_base)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(checkpoint, best_checkpoint_path)
            sync_to_drive(best_checkpoint_path, config["outputs_base"], drive_base)

        history_path = os.path.join(run_dirs["metrics"], "training_history.json")
        _save_history(history, history_path)
        _save_plots(history, run_dirs["plots"])
        sync_to_drive(history_path, config["outputs_base"], drive_base)
        sync_to_drive(run_dirs["plots"], config["outputs_base"], drive_base)

    return {
        "mode": mode,
        "run_dir": run_dirs["run_dir"],
        "best_checkpoint": best_checkpoint_path,
        "history_path": os.path.join(run_dirs["metrics"], "training_history.json"),
        "history": history,
    }
