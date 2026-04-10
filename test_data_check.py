import tifffile
import numpy as np
from pathlib import Path

raw_dir = Path('data/raw/lgg-mri-segmentation')
print(f'raw_dir exists: {raw_dir.exists()}')

# Find first patient folder
for patient in sorted(raw_dir.iterdir())[:1]:
    if patient.is_dir():
        print(f'Patient: {patient.name}')
        files = list(patient.glob('*.tif'))[:3]
        for f in files:
            print(f'  File: {f.name}')
            img = tifffile.imread(str(f))
            print(f'    Shape: {img.shape}, dtype: {img.dtype}')
            if '_mask' not in f.name:
                print(f'    Ch0 mean: {img[:,:,0].mean():.2f}, Ch1 mean: {img[:,:,1].mean():.2f}, Ch2 mean: {img[:,:,2].mean():.2f}')
        break

# Now test the dataset pipeline
print('\n--- Testing dataset pipeline ---')
from src.dataset import build_dataloaders

loaders = build_dataloaders(
    raw_dir=str(raw_dir),
    image_size=256,
    batch_size=2,
    num_workers=0,
    seed=42,
    balanced=True,
)

train_loader = loaders['train']
print(f'\nTrain loader: {len(train_loader)} batches')

# Get one batch
for mask_batch, mri_batch in train_loader:
    print(f'mask_batch shape: {mask_batch.shape}, range: [{mask_batch.min():.3f}, {mask_batch.max():.3f}], mean: {mask_batch.mean():.3f}')
    print(f'mri_batch shape: {mri_batch.shape}, range: [{mri_batch.min():.3f}, {mri_batch.max():.3f}]')
    print(f'mri_batch Ch0 (R/T1) mean: {mri_batch[:,0].mean():.3f}')
    print(f'mri_batch Ch1 (G/FLAIR) mean: {mri_batch[:,1].mean():.3f}')
    print(f'mri_batch Ch2 (B/T2) mean: {mri_batch[:,2].mean():.3f}')
    break
