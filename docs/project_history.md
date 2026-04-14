# Mask-to-MRI — Complete Project History

**Project:** Conditional Diffusion for Brain MRI Synthesis from Tumor Segmentation Masks  
**Team:** Mask-to-MRI  
**Dataset:** LGG Segmentation (TCGA low-grade glioma) — 110 patients, ~3,900 slices  
**Target Platform:** Google Colab (T4 GPU, 16 GB VRAM)

---

## Table of Contents

1. [Phase 1: pix2pix — Baseline GAN](#phase-1-pix2pix--baseline-gan)
2. [Phase 2: med_ddpm (v1) — First Diffusion Model](#phase-2-med_ddpm-v1--first-diffusion-model)
3. [Phase 3: med_ddpm_v2 — Single-Channel FLAIR](#phase-3-med_ddpm_v2--single-channel-flair)
4. [Phase 4: med_ddpm_v3 — Optimized Training](#phase-4-med_ddpm_v3--optimized-training)
5. [Phase 5: med_ddpm_v3.1 — Fine-Tuning](#phase-5-med_ddpm_v31--fine-tuning)
6. [Phase 6: Synthetic Data Generation](#phase-6-synthetic-data-generation)
7. [Lessons Learned](#lessons-learned)
8. [Final Architecture](#final-architecture)
9. [File Structure](#file-structure)
10. [How to Reproduce](#how-to-reproduce)

---

## Phase 1: pix2pix — Baseline GAN

### Goal
Train a pix2pix conditional GAN to generate 3-channel MRI (R/T1, G/FLAIR, B/T2) from binary tumor masks.

### Architecture
- **Generator:** U-Net with 4 encoder blocks, 4 decoder blocks, skip connections
- **Discriminator:** PatchGAN (70×70 receptive field)
- **Loss:** LSGAN adversarial + L1 pixel loss (λ=100)

### Problems Encountered
1. **3-channel output was too hard to learn** — MRI channels have different physical meanings
2. **GAN training was unstable** — mode collapse, vanishing gradients
3. **Small dataset (~1,373 tumor slices)** caused severe overfitting

### Outcome
Decided to switch to diffusion models — more stable training, better sample quality on small datasets.

---

## Phase 2: med_ddpm (v1) — First Diffusion Model

### Goal
Adapt the Med-DDPM paper (Dorjsembe et al. 2024) from 3D → 2D for LGG MRI slices.

### Key Changes
- **3D → 2D:** All Conv3d → Conv2d, AvgPool3d → AvgPool2d
- **Removed depth_size** parameter entirely
- **Tensor shapes:** (B,C,D,H,W) → (B,C,H,W)

### Architecture
- **U-Net noise predictor:** 64 base channels, 1 ResBlock per level, attention at 16×16
- **Diffusion:** Cosine noise schedule, 1000 timesteps, L1 loss
- **Sampling:** DDIM 250 steps, EMA weights

### Problems Encountered
1. **3-channel MRI still too complex** — each channel has different contrast
2. **Training took ~10 hours** for 200 epochs on Colab T4
3. **Overfitting started at epoch ~80** — val loss started climbing

### Outcome
Good proof of concept, but needed simplification. Decided to focus on single-channel FLAIR.

---

## Phase 3: med_ddpm_v2 — Single-Channel FLAIR

### Goal
Simplify to single-channel FLAIR — the most informative MRI sequence for tumor segmentation.

### Key Changes
- **Input:** 2 channels (noisy FLAIR + mask) instead of 4 (noisy 3-channel MRI + mask)
- **Output:** 1 channel (predicted noise for FLAIR) instead of 3
- **Dataset:** Extract only channel 1 (G/FLAIR) from RGB .tif files

### Results
- **Faster training** — fewer parameters, simpler data
- **Better sample quality** — model focuses on one modality
- **Still overfitting** — peaked at epoch ~90, then degraded

### Bugs Fixed During v2
1. **EMA OOM** — `copy.deepcopy` crashed on Colab. Fixed with `state_dict` copy.
2. **DataParallel crash** — `loss.mean()` on non-scalar loss. Fixed with conditional `.mean()`.
3. **Wrong docstrings** — claimed 4 channels, actually 2. Fixed.
4. **v2→v3 suffix confusion** — checkpoint naming conflicts. Fixed with unique suffixes.

### Outcome
FLAIR-only works well. Epoch 90 is the best checkpoint before overfitting. Ready for optimization.

---

## Phase 4: med_ddpm_v3 — Optimized Training

### Goal
Take all lessons learned from v2 and build an optimized training pipeline.

### Research-Backed Optimizations Added

| Optimization | Source | Impact |
|---|---|---|
| **Min-SNR Weighting (gamma=5)** | Hang et al. 2023 | 3.4× faster convergence |
| **Fused AdamW** | PyTorch 2.0+ | 20-30% faster optimizer step |
| **U-Net Dropout (0.1)** | Standard regularization | 5-10% less overfitting |
| **Classifier-Free Guidance (CFG)** | Ho & Salimans 2022 | Sharper tumor boundaries |
| **EMA Decay Schedule** | Common practice | Better early samples (0.9→0.995 ramp) |
| **Optimized tqdm** | Empirical | 5-10% less epoch overhead |
| **TF32 Auto-Enable** | NVIDIA Ampere+ | 2-3× faster matmul |
| **32×32 Multi-Scale Attention** | Architecture improvement | Better global structure |
| **NaN/Inf Protection** | Safety feature | Skips bad batches instead of corrupting |

### Critical Bugs Found During v3 Development
1. **Training loop ran zero iterations** when resuming — `range(91, 31)` = empty. Fixed by interpreting `epochs` as "additional" when resuming.
2. **NaN check after optimizer.step()** — weights already corrupted. Moved check BEFORE backward pass.
3. **Missing `_sync_to_drive` in utils.py** — ImportError crash. Added function.
4. **Checkpoint suffix collision** — v3 saved as `checkpoint_v3_epoch_*.pt` when it should be `v3_1`. Fixed.
5. **Resume from wrong path** — config `resume_from` overridden function argument. Fixed with fallback logic.

### Training Results
- **Peak epoch:** 90 (lowest val_loss = 0.0232, SSIM = 0.203)
- **Training time:** ~3 hours for 90 epochs (vs ~10 hours in v2)
- **Overfitting point:** epoch 90 → val_loss starts climbing after

### Outcome
v3 epoch 90 is the best checkpoint. Ready for fine-tuning and generation.

---

## Phase 5: med_ddpm_v3.1 — Fine-Tuning

### Goal
Fine-tune the best v3 checkpoint (epoch 90) with more data and lower learning rate to push past the overfitting wall.

### Key Changes from v3
| Parameter | v3 | v3.1 | Reason |
|---|---|---|---|
| Starting point | From scratch | Epoch 90 checkpoint | Best model already |
| Learning rate | 1e-4 | **5e-5** | Gentle fine-tuning |
| Warmup epochs | 5 | **0** | Already at good weights |
| Data | Train only (~1,331) | **All splits merged (~3,929)** | 3× more data |
| Dropout | 0.1 | **0.15** | More regularization |
| EMA decay | 0.995 | **0.999** | Smoother averaging |
| Total epochs | 200 | **30 additional** | Fine-tuning only |

### Data Pipeline
```
All patient splits (train + val + test) → ~3,929 pairs
    ↓
Balanced sampling (80% tumor, 20% healthy)
    ↓
FLAIR extraction (channel 1 only)
    ↓
Single DataLoader (~491 batches × 8 per epoch)
```

### Outcome
Fine-tuning extends training from epoch 90 to 120, potentially finding a better model that generalizes further.

---

## Phase 6: Synthetic Data Generation

### Generation Pipeline
```
Load checkpoint (EMA weights)
    ↓
Load training masks
    ↓
For each mask:
    1. Tumor size filter (skip >8% or <50px tumors)
    2. Generate with DDIM 250 steps
    3. Quality filter (skip if std<0.15 or mean>-0.3)
    4. Save as synthetic_XXXX.png + mask
    ↓
Save to local + Google Drive
```

### Key Design Decisions
1. **EMA weights only** — raw weights produce blurry samples
2. **Sequential naming** — `synthetic_0001.png`, `synthetic_0002.png`, ...
3. **Dual filters** — tumor size + quality — to eliminate noise/bad samples
4. **Save to Drive** — persists across Colab session restarts
5. **Preview cell** — side-by-side mask vs generated FLAIR

### Quality Comparison
| Method | Speed | Quality | When to use |
|---|---|---|---|
| DDIM 250 + EMA | 16s/image | Good | Default |
| Full DDPM 1000 | 60s/image | Sharper | Final outputs |
| DDIM + CFG 2.0 | 32s/image | Sharpest edges | Experimentation |

---

## Lessons Learned

### What Worked
1. **Single-channel FLAIR** — simpler, faster, better quality than 3-channel MRI
2. **EMA weights** — consistently sharper than raw model weights
3. **Min-SNR weighting** — genuinely faster convergence, no downsides
4. **Patient-level splitting** — critical for valid evaluation (don't split by slice)
5. **Balanced sampling** — 80/20 tumor/healthy ratio is the sweet spot

### What Didn't Work
1. **pix2pix GAN** — unstable on small dataset, poor quality
2. **3-channel MRI output** — too complex for ~1,300 samples
3. **Training past epoch 90** — model overfits, quality degrades
4. **Vertical flip + RandomRotate90** — anatomically unrealistic for brain MRI
5. **DataParallel on Colab** — scalar loss crash, not worth the complexity

### What We'd Do Differently
1. **Start with single-channel** — skip 3-channel entirely
2. **Use gradient checkpointing** — would allow 2× batch size
3. **Train with perceptual loss** — VGG feature loss might improve sharpness
4. **Use early stopping** — automatically stop when val_loss starts climbing
5. **More aggressive augmentations** — small rotation ±15°, elastic distortion

---

## Final Architecture

### Model
```
U-Net Noise Predictor
├── in_channels: 2 (noisy FLAIR + mask)
├── out_channels: 1 (predicted noise ε)
├── num_channels: 64
├── num_res_blocks: 1
├── attention: 16×16, 32×32 spatial
├── dropout: 0.1 (v3) / 0.15 (v3.1)
└── ~39.7M parameters

Gaussian Diffusion
├── timesteps: 1000
├── noise schedule: cosine
├── loss: L1 + Min-SNR (gamma=5)
└── sampling: DDIM 250 steps / full DDPM 1000 steps
```

### Training Pipeline
```
Patient-level split (88/11/11)
    ↓
BalancedLGGDataset (80% tumor, 20% healthy)
    ↓
FLAIR extraction (channel 1)
    ↓
Augmentations: resize 286→256, random crop, horizontal flip, small rotation
    ↓
U-Net + Diffusion training (Min-SNR, fused AdamW, AMP, EMA)
    ↓
Checkpoint every 5-10 epochs → Drive
```

### Generation Pipeline
```
Load EMA checkpoint → Filter masks → DDIM sampling → Quality filter → Save
```

---

## File Structure

```
mask-to-mri/
├── src/
│   ├── dataset.py                    # LGGDataset, BalancedLGGDataset, FLAIRDataset
│   ├── pix2pix/                      # Original GAN (deprecated)
│   ├── med_ddpm/                     # v1 — 3-channel diffusion (deprecated)
│   ├── med_ddpm_v2/                  # v2 — single-channel FLAIR (working)
│   ├── med_ddpm_v3/                  # v3 — optimized (best: epoch 90)
│   │   ├── __init__.py
│   │   ├── config.py                 # Training hyperparameters
│   │   ├── model.py                  # U-Net + GaussianDiffusion + Min-SNR
│   │   ├── train.py                  # Training loop with all optimizations
│   │   ├── sample.py                 # Generation utility
│   │   └── utils.py                  # Drive sync helper
│   └── med_ddpm_v3_1/                # v3.1 — fine-tuning from epoch 90
│       ├── __init__.py
│       ├── config.py                 # Fine-tuning hyperparameters
│       ├── model.py                  # Same as v3
│       ├── train.py                  # Resume-aware training loop
│       ├── sample.py                 # Generation with cfg_scale
│       └── utils.py                  # All-data dataloader + Drive sync
├── notebooks/
│   ├── med_ddpm_v3_train_colab.ipynb      # v3 training notebook
│   ├── med_ddpm_v3_1_train_colab.ipynb    # v3.1 fine-tuning notebook
│   └── generate_synthetic_v3_90.ipynb     # Standalone generation notebook
├── docs/
│   ├── med_ddpm_v3_report.md              # v3 technical report
│   ├── med_ddpm_v3_1_report.md            # v3.1 fine-tuning report
│   └── project_history.md                 # This file
└── config.yaml                       # Original pix2pix config (deprecated)
```

---

## How to Reproduce

### 1. Train v3 from Scratch
```
Open: notebooks/med_ddpm_v3_train_colab.ipynb
Runtime: GPU T4
Time: ~3 hours for 90 epochs (best checkpoint)
```

### 2. Fine-Tune from Epoch 90
```
Open: notebooks/med_ddpm_v3_1_train_colab.ipynb
Runtime: GPU T4
Requires: v3 epoch 90 checkpoint on Drive
Time: ~80 minutes for 30 more epochs
```

### 3. Generate Synthetic Images
```
Open: notebooks/generate_synthetic_v3_90.ipynb
Or: Run the single-cell generation script
Runtime: GPU T4
Output: synthetic_0001.png, synthetic_0002.png, ... (to Drive)
```

### Dataset Setup
1. Download LGG Segmentation dataset from Kaggle
2. Upload as `lgg-mri-segmentation.zip` to Drive at `MyDrive/mask-to-mri/dataset/`
3. Notebook auto-extracts on first run

---

**Author:** Amine  
**Repository:** https://github.com/AmineAitLaamim/Mask-to-MRI  
**Last Updated:** April 2026
