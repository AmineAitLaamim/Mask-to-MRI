"""Segmentation metrics for Experiment B."""

import torch


def _prepare_predictions(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()
    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    return preds, targets


def compute_batch_dice(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0
) -> float:
    preds, targets = _prepare_predictions(logits, targets, threshold)
    intersection = (preds * targets).sum(dim=1)
    score = (2.0 * intersection + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
    return float(score.mean().item())


def compute_batch_iou(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0
) -> float:
    preds, targets = _prepare_predictions(logits, targets, threshold)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    score = (intersection + smooth) / (union + smooth)
    return float(score.mean().item())


def compute_batch_confusion(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> tuple[float, float, float]:
    """Return global intersection, prediction sum, and target sum for a batch."""
    preds, targets = _prepare_predictions(logits, targets, threshold)
    intersection = float((preds * targets).sum().item())
    pred_sum = float(preds.sum().item())
    target_sum = float(targets.sum().item())
    return intersection, pred_sum, target_sum


def dice_from_totals(
    intersection: float, pred_sum: float, target_sum: float, smooth: float = 1.0
) -> float:
    return float((2.0 * intersection + smooth) / (pred_sum + target_sum + smooth))


def iou_from_totals(
    intersection: float, pred_sum: float, target_sum: float, smooth: float = 1.0
) -> float:
    union = pred_sum + target_sum - intersection
    return float((intersection + smooth) / (union + smooth))
