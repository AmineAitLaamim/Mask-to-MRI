# Mask-to-MRI

**Conditional Diffusion for Brain MRI Synthesis from Tumor Segmentation Masks**

Train a conditional DDPM to generate synthetic brain MRI FLAIR images from binary tumor segmentation masks, then use the synthetic data to augment downstream tumor segmentation.

## The Problem

Medical imaging datasets are often severely limited due to privacy concerns, high acquisition costs, and the need for expert annotation. For brain tumor segmentation, having only ~1,000 real training slices is often too few to train a highly robust model without extreme overfitting.

**The Solution:** This project uses a Denoising Diffusion Probabilistic Model (DDPM) to artificially multiply the available dataset. By generating realistic synthetic MRI slices from randomly generated or existing tumor masks, we can safely augment the training data and improve the segmentation model's performance without requiring additional real patient data.

## Overview

This project implements the solution by:

1. **Training a conditional diffusion model** (Med-DDPM v3) to synthesize realistic FLAIR MRI slices from tumor masks
2. **Generating a synthetic dataset** (~1,065 image/mask pairs) with quality filtering
3. **Evaluating downstream impact** through a systematic series of segmentation experiments (A–F)

### Pipeline

```
Binary Tumor Mask → Med-DDPM v3 (DDIM 250 steps) → Synthetic FLAIR MRI
                                                          ↓
                                        Augment real dataset for segmentation
                                                          ↓
                                        U-Net segmentation → Dice / IoU
```

## Dataset

**LGG Segmentation** (TCGA low-grade glioma) from Kaggle:

- 110 patients, ~3,900 total slices
- 256×256 single-channel FLAIR images with binary tumor masks
- Patient-level split: 88 train / 11 val / 11 test

After filtering to tumor-containing slices only:

| Split | Slices |
|-------|--------|
| Train | ~1,065 |
| Val   | 151    |
| Test  | 157    |

## Model Architecture

### Diffusion Model (Med-DDPM v3)

Adapted from [Dorjsembe et al. 2024](https://arxiv.org/abs/2305.18453) (3D → 2D), conditioned on the segmentation mask via channel concatenation.

| Component | Details |
|-----------|---------|
| Architecture | U-Net noise predictor |
| Input | 2 channels (noisy FLAIR + mask) |
| Output | 1 channel (predicted noise) |
| Base channels | 64, multipliers (1, 2, 4, 8) |
| Attention | 16×16 and 32×32 spatial |
| Parameters | ~39.7M |
| Diffusion | 1000 timesteps, cosine schedule |
| Sampling | DDIM 250 steps (~16s/image on T4) |

**Key optimizations in v3:**

- Min-SNR weighting (γ=5) — 3.4× faster convergence
- Fused AdamW — 20-30% faster optimizer step
- Classifier-free guidance (CFG) — sharper tumor boundaries
- EMA decay schedule (0.9 → 0.995 ramp)
- Multi-scale attention at 16×16 and 32×32

Best checkpoint: **epoch 90** (val_loss = 0.0232)

### Segmentation Model (Experiment B)

Standard U-Net for single-channel FLAIR tumor segmentation.

| Component | Details |
|-----------|---------|
| Input | 1-channel FLAIR |
| Output | 1-channel binary mask |
| Encoder | [64, 128, 256, 512] |
| Bottleneck | 1024 channels |
| Decoder | ConvTranspose2d + skip connections |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Optimizer | Adam, lr=1e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 100 |

## Experiment Results

All experiments evaluated on the same **157 real test slices** using the best checkpoint (selected by highest validation Dice).

| Exp | Description | Real Train | Synthetic | Total Train | Aug | Test Dice | Test IoU | Test Loss |
|-----|-------------|------------|-----------|-------------|-----|-----------|----------|-----------|
| A | Baseline | 100% (~1065) | 0% (0) | ~1065 | ✅ | **0.8992** | **0.8169** | 0.0988 |
| B | Augmented | 50% (~1065) | 50% (~1065) | ~2130 | ✅ | 0.8971 | 0.8134 | 0.0939 |
| C | Baseline no-aug | 100% (~1065) | 0% (0) | ~1065 | ❌ | 0.8931 | 0.8069 | 0.1080 |
| C+ | Augmented no-aug | 50% (~1065) | 50% (~1065) | ~2130 | ❌ | 0.8981 | 0.8150 | 0.1007 |
| D | Half real | 100% (~532) | 0% (0) | ~532 | ❌ | 0.8894 | 0.8009 | 0.1241 |
| E1 | Half real + syn 1:1 | 50% (~532) | 50% (~532) | ~1064 | ❌ | 0.8963 | 0.8121 | 0.0994 |
| E2 | Half real + syn 1:2 | 33% (~532) | 67% (~1064) | ~1596 | ❌ | 0.8978 | 0.8145 | 0.0994 |
| F | Synthetic only | 0% (0) | 100% (~1065) | ~1065 | ❌ | 0.6967 | 0.5346 | 0.2756 |

### Key Findings

1. **Synthetic data compensates for missing real data**: D → E1 → E2 shows that adding synthetic pairs recovers most of the performance gap. E2 with only ~33% real data in the training mix nearly matches full-data baseline A.

2. **Augmentation and synthetic data have comparable effects**: A vs C (+0.6% from augmentation) and C vs C+ (+0.5% from synthetic data) show similar improvements.

3. **Synthetic-only training has a large gap** (F: 0.6967 vs A: 0.8992). The DDPM-generated images alone are insufficient for competitive segmentation — real data remains essential.

4. **All real-data experiments cluster within ~1% Dice** (0.889–0.899), suggesting the model saturates on this dataset size.

## Project Structure

```
Mask-to-MRI/
├── src/
│   ├── dataset.py                         # LGG dataset, patient-level splits
│   ├── med_ddpm_v3/                       # Diffusion model (best: epoch 90)
│   │   ├── config.py                      # Training hyperparameters
│   │   ├── model.py                       # U-Net + GaussianDiffusion + Min-SNR + CFG
│   │   ├── train.py                       # Training loop
│   │   ├── sample.py                      # Generation utility
│   │   └── utils.py                       # Drive sync helper
│   └── experiment_B/                      # Segmentation experiments
│       ├── config.py                      # Central config (flags for all experiments)
│       ├── model.py                       # U-Net segmentation model
│       ├── dataset.py                     # Real + synthetic dataloaders
│       ├── losses.py                      # Dice + BCE loss
│       ├── metrics.py                     # Dice, IoU computation
│       ├── train.py                       # Training loop with checkpointing
│       ├── evaluate.py                    # Test evaluation
│       └── utils.py                       # Drive sync
├── notebooks/
│   ├── med_ddpm_v3_train_colab.ipynb      # Diffusion model training
│   ├── generate_synthetic_v3_90.ipynb     # Synthetic data generation
│   ├── generate_synthetic_improved.ipynb  # Improved generation with quality filtering
│   ├── experiment_B_train_colab.ipynb     # Experiments A & B
│   ├── experiment_CD_train_colab.ipynb    # Experiments C & C+
│   ├── experiment_DE_train_colab.ipynb    # Experiments D, E1, E2
│   └── experiment_F_train_colab.ipynb     # Experiment F (synthetic only)
├── docs/
│   ├── project_explanation.md             # Project overview and design decisions
│   ├── project_history.md                 # Full development history (pix2pix → v3)
│   ├── med_ddpm_v3_report.md              # v3 technical report
│   └── experiment_B_implementation.md     # Segmentation experiment implementation notes
└── requirements.txt
```

## Quick Start

### Prerequisites

- Google Colab with T4 GPU (16 GB VRAM)
- Google Drive for persistent storage
- LGG Segmentation dataset from [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

### 1. Dataset Setup

Upload the dataset to Google Drive:

```
MyDrive/mask-to-mri/dataset/lgg-mri-segmentation.zip
```

### 2. Train the Diffusion Model

Open `notebooks/med_ddpm_v3_train_colab.ipynb` on Colab and run all cells.

- Training time: ~3 hours for 90 epochs
- Checkpoints saved to Drive every 10 epochs
- Best checkpoint: epoch 90

### 3. Generate Synthetic Data

Open `notebooks/generate_synthetic_improved.ipynb` on Colab.

- Generates ~1,065 synthetic FLAIR/mask pairs
- Two-stage quality filtering (mask size + output quality)
- Saved as `synthetic_data.zip` on Drive

### 4. Run Segmentation Experiments

Each experiment has its own notebook:

| Notebook | Experiments | Config |
|----------|------------|--------|
| `experiment_B_train_colab.ipynb` | A (baseline), B (augmented) | `EXPERIMENT_MODE = "baseline"` or `"augmented"` |
| `experiment_CD_train_colab.ipynb` | C (no-aug baseline), C+ (no-aug augmented) | Same modes, `USE_AUGMENTATION = False` |
| `experiment_DE_train_colab.ipynb` | D, E1, E2 (half real ± synthetic) | `REAL_DATA_FRACTION = 0.5`, vary `SYNTHETIC_RATIO` |
| `experiment_F_train_colab.ipynb` | F (synthetic only) | `SYNTHETIC_ONLY = True` |

Each experiment runs for 100 epochs with automatic checkpointing and resume support.

## Google Drive Layout

```
MyDrive/mask-to-mri/
├── dataset/
│   ├── lgg-mri-segmentation.zip
│   └── synthetic_data.zip
└── experiment_B/
    ├── baseline/               ← A
    ├── augmented/              ← B
    ├── baseline_noaug/         ← C
    ├── augmented_noaug/        ← C+
    ├── half_real/              ← D
    ├── half_real_syn1to1/      ← E1
    ├── half_real_syn1to2/      ← E2
    └── synthetic_only/         ← F
```

Each experiment folder contains: `checkpoints/`, `metrics/`, `plots/`, `samples/`.

## Development History

| Phase | Model | Outcome |
|-------|-------|---------|
| 1 | pix2pix GAN | Unstable on small dataset, poor quality |
| 2 | Med-DDPM v1 (3-channel) | Too complex for ~1,300 samples |
| 3 | Med-DDPM v2 (FLAIR only) | Good quality, simpler training |
| 4 | **Med-DDPM v3 (optimized)** | **Best model — 3× faster, epoch 90** |
| 5 | Synthetic generation | ~1,065 quality-filtered pairs |
| 6 | Experiment B (A–F) | Full ablation study completed |

## References

1. Dorjsembe et al. 2024 — *Conditional Diffusion Models for Semantic 3D Brain MRI Synthesis* (base architecture)
2. Hang et al. 2023 — *Efficient Diffusion Training via Min-SNR Weighting Strategy* (loss weighting)
3. Ho & Salimans 2022 — *Classifier-Free Diffusion Guidance* (CFG)
4. Ho et al. 2020 — *Denoising Diffusion Probabilistic Models* (DDPM)
5. Song et al. 2021 — *Denoising Diffusion Implicit Models* (DDIM fast sampling)

## Author

**Amine Ait Laamim**

Repository: [github.com/AmineAitLaamim/Mask-to-MRI](https://github.com/AmineAitLaamim/Mask-to-MRI)
