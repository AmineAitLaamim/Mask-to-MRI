"""High-level orchestration for Experiment B."""

from __future__ import annotations

import json
import os

from .evaluate import evaluate_experiment_b
from .train import train_experiment_b
from .utils import sync_to_drive


def run_full_experiment_b(config: dict) -> dict:
    baseline_train = train_experiment_b(config, mode="baseline")
    baseline_test = evaluate_experiment_b(
        config, baseline_train["best_checkpoint"], split="test"
    )

    augmented_train = train_experiment_b(config, mode="augmented")
    augmented_test = evaluate_experiment_b(
        config, augmented_train["best_checkpoint"], split="test"
    )

    comparison = {
        "baseline": baseline_test,
        "augmented": augmented_test,
        "delta_dice": augmented_test["dice"] - baseline_test["dice"],
        "delta_iou": augmented_test["iou"] - baseline_test["iou"],
    }

    comparison_path = os.path.join(config["outputs_base"], "comparison_summary.json")
    os.makedirs(config["outputs_base"], exist_ok=True)
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    sync_to_drive(comparison_path, config["outputs_base"], config.get("drive_base"))

    comparison["comparison_path"] = comparison_path
    return comparison
