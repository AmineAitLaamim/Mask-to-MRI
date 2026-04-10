"""Test if albumentations is corrupting channel differences."""
import numpy as np
import tifffile
from pathlib import Path
import albumentations as A

raw_dir = Path('data/raw/lgg-mri-segmentation')

# Load a sample image
for patient in sorted(raw_dir.iterdir())[:1]:
    if patient.is_dir():
        for img_path in sorted(patient.glob('*.tif')):
            if '_mask' not in img_path.name:
                img = tifffile.imread(str(img_path))
                print(f'Raw image shape: {img.shape}, dtype: {img.dtype}')
                print(f'Raw Ch0 mean={img[:,:,0].mean():.2f}, Ch1 mean={img[:,:,1].mean():.2f}, Ch2 mean={img[:,:,2].mean():.2f}')
                break
        break

# Test val augmentation (just resize)
val_aug = A.Compose([A.Resize(256, 256)])
val_result = val_aug(image=img)
val_img = val_result['image']
print(f'\nAfter val_aug (Resize only):')
print(f'  Ch0 mean={val_img[:,:,0].mean():.2f}, Ch1 mean={val_img[:,:,1].mean():.2f}, Ch2 mean={val_img[:,:,2].mean():.2f}')

# Test train augmentation
train_aug = A.Compose([
    A.Resize(286, 286),
    A.RandomCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.Blur(blur_limit=3, p=1.0),
    ], p=0.2),
    A.OneOf([
        A.RandomBrightnessContrast(p=1.0),
        A.CLAHE(p=1.0),
    ], p=0.3),
])

# Run multiple times to see if channels get mixed
print(f'\nAfter train_aug (5 runs):')
for i in range(5):
    result = train_aug(image=img)
    aug_img = result['image']
    print(f'  Run {i}: Ch0 mean={aug_img[:,:,0].mean():.2f}, Ch1 mean={aug_img[:,:,1].mean():.2f}, Ch2 mean={aug_img[:,:,2].mean():.2f}')

# Test normalization
normalized = (val_img.astype(np.float32) / 127.5) - 1.0
print(f'\nAfter val_aug + normalize:')
print(f'  Ch0 mean={normalized[:,:,0].mean():.3f}, Ch1 mean={normalized[:,:,1].mean():.3f}, Ch2 mean={normalized[:,:,2].mean():.3f}')
