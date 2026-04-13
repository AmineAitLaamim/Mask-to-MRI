# Med-DDPM v3.1 — Fine-Tuning Report

**Project:** Mask-to-MRI — Conditional Diffusion for Brain MRI Synthesis  
**Version:** v3.1 (Fine-tuning from v3 epoch 90)  
**Date:** April 13, 2026  
**Target Platform:** Google Colab (T4 GPU, 16 GB VRAM)

---

## 1. Project Overview

**Goal:** Fine-tune the best v3 checkpoint (epoch 90) with more data and a lower learning rate to prevent overfitting and improve generalization.

**Motivation:** v3 training peaked at epoch 90, then degraded. v3.1 addresses this by:
- Starting from the best checkpoint (epoch 90)
- Using a lower learning rate (5e-5 instead of 1e-4)
- Merging all data (train+val+test) for 3× more samples
- Fine-tuning for only 30 more epochs

---

## 2. Differences from v3

| Parameter | v3 | v3.1 | Reason |
|-----------|-----|------|--------|
| **Starting point** | From scratch | Epoch 90 checkpoint | Best model already trained |
| **Learning rate** | 1e-4 | **5e-5** | Half — gentle fine-tuning |
| **Warmup epochs** | 5 | **0** | Already at good weights |
| **Data used** | Train only (~1,331) | **All splits merged (~3,929)** | 3× more samples |
| **Balanced sampling** | 80/20 | **80/20** | Same tumor/healthy ratio |
| **Dropout** | 0.1 | **0.15** | More regularization |
| **EMA decay** | 0.995 | **0.999** | Smoother averaging |
| **Total epochs** | 200 | **30 additional** | Fine-tuning, not from scratch |
| **Save frequency** | Every 10 | **Every 5** | Catch best model precisely |
| **Output dir** | outputs_v3 | **outputs_v3_1** | Separate from v3 |
| **Checkpoint suffix** | v3 | **v3_1** | No collision with v3 |

---

## 3. Architecture

### Data Pipeline

```
All patient splits (train + val + test)
    │
    ├─ Merge all pairs → ~3,929 total
    │
    ├─ Tumor filter
    │   ├─ Tumor:    ~1,065 slices
    │   └─ Healthy:  ~2,864 slices
    │
    ├─ Balanced sampling (80% tumor, 20% healthy)
    │   ├─ ~3,143 tumor samples/epoch (with replacement)
    │   ├─ ~786 healthy samples/epoch (with replacement)
    │   └─ Total: ~3,929 samples/epoch (~491 batches × 8)
    │
    └─ FLAIR extraction (channel 1 only)
```

### Training Loop

1. Load v3 epoch 90 checkpoint
2. Override `resume_from` to use that checkpoint
3. Training loop runs from epoch 91 to 120 (`start_epoch + epochs`)
4. NaN/Inf batches are skipped (not corrupted)
5. EMA decay ramps from 0.99 → 0.999 over 10 epochs
6. Checkpoints saved every 5 epochs to Drive

---

## 4. Key Optimizations

### 4.1 Lower Learning Rate (5e-5)

Fine-tuning a well-trained model requires smaller steps to avoid destroying learned weights. Half the original LR is a standard fine-tuning heuristic.

### 4.2 All Data Merged

v3 used only the train split (~1,331 tumor slices). v3.1 merges all three splits (~3,929 total), giving the model 3× more data to learn from during fine-tuning.

### 4.3 Higher Dropout (0.15)

Increased from 0.1 to 0.15 to provide stronger regularization against the overfitting that occurred after epoch 90 in v3.

### 4.4 Smoother EMA (0.999)

Higher EMA decay means the EMA model tracks the live model more smoothly, producing higher-quality samples at checkpoint time.

### 4.5 NaN/Inf Protection

The NaN check runs **before** `loss.backward()` and `optimizer.step()`. If detected, the batch is skipped entirely — no weight corruption.

---

## 5. File Structure

```
mask-to-mri/
├── src/
│   └── med_ddpm_v3_1/
│       ├── __init__.py          # Package init, exports
│       ├── config.py            # Fine-tuning hyperparameters
│       ├── model.py             # Same as v3 (identical copy)
│       ├── train.py             # Training loop (NaN-safe, resume-aware)
│       ├── sample.py            # Generation utility (cfg_scale support)
│       └── utils.py             # Drive sync + all-data dataloader
├── notebooks/
│   └── med_ddpm_v3_1_train_colab.ipynb  # Colab fine-tuning notebook
└── docs/
    └── med_ddpm_v3_1_report.md  # This file
```

---

## 6. Configuration Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr` | 5e-5 | Half of v3 |
| `epochs` | 30 | Additional epochs (total: 90 + 30 = 120) |
| `warmup_epochs` | 0 | No warmup when fine-tuning |
| `dropout` | 0.15 | Higher regularization |
| `ema_decay` | 0.999 | Smoother EMA |
| `ema_decay_start` | 0.99 | Start near final value |
| `ema_decay_ramp_epochs` | 10 | Quick ramp |
| `batch_size` | 8 | Same as v3 (T4 fits) |
| `tumor_ratio` | 0.8 | 80/20 balanced |
| `save_every` | 5 | Catch best model precisely |
| `resume_from` | v3 epoch 90 checkpoint | Auto-detected on Colab |

---

## 7. How to Use

### On Google Colab

1. Open `notebooks/med_ddpm_v3_1_train_colab.ipynb`
2. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**
3. Run cells 1-5 (setup + dataset upload)
4. Run cells 6-11 (import + model creation + sanity check)
5. Run cell 12 (restore checkpoint from Drive if needed)
6. Run cell 13 (verify v3 epoch 90 checkpoint found)
7. Run cell 14 (start fine-tuning) — ~80 minutes for 30 epochs
8. Checkpoints saved every 5 epochs to `outputs_v3_1/checkpoints/`

### After Fine-Tuning

- **Cell 15:** Plot fine-tuning loss curves
- **Cell 16:** Generate synthetic FLAIR images from best checkpoint

### Expected Drive Usage

```
MyDrive/mask-to-mri/outputs_v3_1/
├── checkpoints/
│   ├── checkpoint_v3_1_epoch_95.pt
│   ├── checkpoint_v3_1_epoch_100.pt
│   ├── ...
│   └── checkpoint_v3_1_epoch_120.pt
├── samples/
│   ├── v3_1_samples_epoch_95.png
│   ├── ...
│   └── v3_1_loss_curves.png
├── metrics/
│   └── v3_1_training_history.json
└── synthetic/
    └── (generated images)
```

---

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: dataset/lgg-mri-segmentation` | Run Cell 5 first to extract dataset from Drive |
| `No checkpoint found — starting from scratch` | Upload v3 epoch 90 checkpoint to Drive at `outputs_v3/checkpoints/` |
| Training does zero iterations | Check that `resume_from` path points to an existing file |
| NaN/Inf warnings | Normal — batches are skipped automatically |

---

## 9. Commit History

| Commit | Hash | Description |
|--------|------|-------------|
| v3.1 creation | `0f5937b` | Create med_ddpm_v3_1 for fine-tuning from epoch 90 |
| Bug fixes | `046c91d` | Fix _sync_to_drive, NaN check ordering, resume warning |
| Epoch fix | `a9dceea` | CRITICAL: interpret epochs as "additional" when resuming |
| Notebook fix | `2a6a0f0` | Add dataset extraction cell and raw_dir override |
| Notebook rewrite | `a1eb69f` | Full rewrite matching v3 cell structure |

---

**Author:** Mask-to-MRI Team  
**Repository:** https://github.com/AmineAitLaamim/Mask-to-MRI
