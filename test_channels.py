"""Verify that raw MRI data has 3 genuinely different channels."""
import tifffile
import numpy as np
from pathlib import Path
import sys

raw_dir = Path('data/raw/lgg-mri-segmentation')
patients = sorted(raw_dir.iterdir())

found = False
for patient in patients:
    if not patient.is_dir():
        continue
    imgs = sorted(patient.glob('*.tif'))
    for img_path in imgs:
        if '_mask' not in img_path.name:
            img = tifffile.imread(str(img_path))
            print(f'File: {img_path.name}')
            print(f'  Shape: {img.shape}, dtype: {img.dtype}')
            print(f'  Ch0 (R/T1)   mean={img[:,:,0].mean():.2f}, std={img[:,:,0].std():.2f}')
            print(f'  Ch1 (G/FLAIR) mean={img[:,:,1].mean():.2f}, std={img[:,:,1].std():.2f}')
            print(f'  Ch2 (B/T2)   mean={img[:,:,2].mean():.2f}, std={img[:,:,2].std():.2f}')
            
            # After normalization
            norm = (img.astype(np.float32) / 127.5) - 1.0
            print(f'  After norm Ch0 mean={norm[:,:,0].mean():.3f}')
            print(f'  After norm Ch1 mean={norm[:,:,1].mean():.3f}')
            print(f'  After norm Ch2 mean={norm[:,:,2].mean():.3f}')
            found = True
            break
    if found:
        break

if not found:
    print('ERROR: No MRI files found!')
    sys.exit(1)

# Now test the DataLoader
print('\n--- DataLoader test ---')
from src.dataset import build_dataloaders

loaders = build_dataloaders(
    raw_dir=str(raw_dir),
    image_size=256,
    batch_size=1,
    num_workers=0,
    seed=42,
    balanced=False,  # Use plain LGGDataset for simplicity
)

val_loader = loaders['val']
for mask_batch, mri_batch in val_loader:
    print(f'mri_batch shape: {mri_batch.shape}')
    print(f'mri_batch Ch0 mean={mri_batch[:,0].mean():.3f}, std={mri_batch[:,0].std():.3f}')
    print(f'mri_batch Ch1 mean={mri_batch[:,1].mean():.3f}, std={mri_batch[:,1].std():.3f}')
    print(f'mri_batch Ch2 mean={mri_batch[:,2].mean():.3f}, std={mri_batch[:,2].std():.3f}')
    break
