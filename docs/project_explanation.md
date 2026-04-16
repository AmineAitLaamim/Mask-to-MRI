# Mask-to-MRI: Project Explanation

## 1. Goal

Train a conditional diffusion model to generate synthetic brain MRI FLAIR images from binary tumor segmentation masks. The synthetic images are then used to augment the real dataset and improve downstream tumor segmentation Dice scores in Experiment B.

## 2. Dataset

Dataset: LGG Segmentation (TCGA low-grade glioma)

- 110 patients
- Approximately 3,900 total slices
- 256x256 single-channel FLAIR images paired with binary tumor masks

Patient-level split is used to avoid leakage:

- 88 patients train
- 11 patients validation
- 11 patients test

After filtering to tumor-containing slices only:

- Approximately 1,065 train
- Approximately 151 validation
- Approximately 157 test

Normalization:

- Images normalized to `[-1, 1]`
- Masks normalized to `[-1, 1]`

## 3. Model Architecture: Med-DDPM v3

Base model: adapted from Dorjsembe et al. 2024 from 3D to 2D, conditioned on the segmentation mask via channel concatenation.

How conditioning works:

- Input: noisy FLAIR image `(1 channel)` plus mask `(1 channel)` = `2 channels`
- Output: predicted noise `(1 channel)`
- At sampling time, the model starts from pure Gaussian noise and iteratively denoises conditioned on the mask

### U-Net Noise Predictor

- `num_channels = 64`
- `num_res_blocks = 1`
- Channel multipliers: `(1, 2, 4, 8)`
- Attention at `16x16` and `32x32`
- Dropout `p = 0.1`
- Approximately `39.7M` parameters

### Diffusion Process

- `1000` timesteps
- Cosine noise schedule
- DDIM `250`-step fast sampling
- Approximate speed: `~16s` per image on Colab T4

## 4. Training

Platform: Google Colab T4 GPU (`16 GB VRAM`)

Key settings:

- Batch size: `8`
- Learning rate: `1e-4`
- Scheduler: `5` warmup epochs, then cosine decay to `1e-5`
- AMP mixed precision enabled
- Gradient clipping with `max_norm = 1.0`
- Checkpoints every `10` epochs with auto-resume
- EMA model used for sampling at checkpoint time

### Training Progression

| Epoch | Val Loss | Notes |
|---|---:|---|
| 10 | 0.0338 | Live model, too bright |
| 30 | 0.0291 | EMA active, improving |
| 70 | 0.0232 | Best val loss so far |
| 90 | 0.0232 | Best checkpoint, correct intensity |
| 80 | 0.0295 | Temporary collapse, recovered |
| 110 | 0.0258 | v3.1 fine-tuning, worse |

Best checkpoint: epoch `90`

Why epoch 90 is preferred:

- Lowest validation loss (`0.0232`)
- Correct intensity range
- Best visual quality across the sample grids
- `3/4` sample grids showing clean brain anatomy with correct tumor regions

## 5. Synthetic Data Generation

Checkpoint used: `v3` epoch `90` with EMA weights

### Stage 1: Pre-generation Mask Filter

```python
tumor_ratio = tumor_pixels / (256 * 256)
if tumor_ratio > 0.08 or tumor_pixels < 50:
    skip  # large tumors produce noise, tiny tumors are uninformative
```

Valid masks: `1,022` out of `1,065`

### Stage 2: Post-generation Quality Filter

```python
if fake_mean > -0.2 or fake_std > 0.45:
    skip  # wrong intensity range or pure noise
```

This rejects outputs with:

- Incorrect intensity range, especially overly bright outputs
- Pure noise or unstable generations

### Generation Settings

- `250` DDIM steps per image
- `2` synthetics per valid mask
- Diffusion is stochastic, so repeated runs with the same mask can produce different outputs

Target before quality filtering:

- Approximately `2,044` synthetic images

Final usable dataset:

- Survivors of the post-generation filter
- Sampled down to approximately `1,065` synthetic images
- Final real-to-synthetic ratio targeted at `1:1`

### Output Format

- Grayscale PNG for synthetic FLAIR
- Matching mask saved alongside each synthetic image

## 6. Experiment B Plan

Goal: show that synthetic data improves tumor segmentation Dice scores.

Architecture: standard U-Net

- Input: `1`-channel FLAIR
- Output: `1`-channel binary mask
- Encoder: `4` levels with features `[64, 128, 256, 512]`
- Bottleneck: `1024` channels
- Decoder: `ConvTranspose2d` upsampling with skip connections

Loss:

- `0.5 * BCEWithLogitsLoss`
- `0.5 * DiceLoss`

Two models are trained with identical settings except for the training data.

| Setting | Model A | Model B |
|---|---|---|
| Training data | Real only (`~1,065` slices) | Real + synthetic (`~2,130` slices) |
| Architecture | U-Net | U-Net |
| Loss | BCE + Dice | BCE + Dice |
| Optimizer | Adam, `lr=1e-4` | Adam, `lr=1e-4` |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Epochs | 100 | 100 |
| Augmentation | Flip + RandomRotate90 | Flip + RandomRotate90 |
| Evaluation | Test set (`157` slices) | Test set (`157` slices) |

Metrics:

- Dice coefficient
- IoU

Evaluation uses the held-out test set only:

- `157` slices
- Never seen by the DDPM
- Never seen by the segmentation model during training

### Expected Result

- Model A Dice: approximately `0.62` to `0.65`
- Model B Dice: approximately `0.63` to `0.67`
- Expected improvement: approximately `+0.01` to `+0.05` Dice

This follows Dorjsembe et al. 2024:

- Real only: Dice `0.6531`
- Real + synthetic: Dice `0.6675`
- Improvement: `+0.0144`

## 7. Key Design Decisions

### Why patient-level split

Slice-level split would leak information because slices from the same patient are strongly correlated. Patient-level splitting ensures the test set contains completely unseen anatomy.

### Why a 1:1 synthetic-to-real ratio

This is a common setting in medical augmentation literature. Larger synthetic ratios increase the risk that the segmentation model learns synthetic artifacts instead of real anatomy.

### Why epoch 90

Epoch `90` combines:

- Lowest validation loss
- Correct intensity range
- Strongest visual quality in generated sample grids

### Why filter large tumors above 8%

Masks covering more than `8%` of the image tended to produce noisy generations. The model did not see enough large-tumor examples during training to generalize well in that regime.

### Why use BCE + Dice loss

- BCE alone does not handle class imbalance well because tumor pixels are sparse
- Dice alone is unstable early in training
- The combination gives more stable optimization and better tumor-boundary behavior
