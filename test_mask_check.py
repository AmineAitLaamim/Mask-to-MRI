import tifffile
import numpy as np
from pathlib import Path

raw_dir = Path('data/raw/lgg-mri-segmentation')
for patient in sorted(raw_dir.iterdir())[:1]:
    if patient.is_dir():
        masks = list(patient.glob('*_mask.tif'))
        if masks:
            m = tifffile.imread(str(masks[0]))
            print(f'Mask shape: {m.shape}, dtype: {m.dtype}')
            print(f'Mask unique values: {np.unique(m)}')
            print(f'Mask min: {m.min()}, max: {m.max()}, mean: {m.mean():.2f}')
            
            # What happens after _normalize?
            normalized = (m.astype(np.float32) / 127.5) - 1.0
            print(f'After _normalize: unique values ~ {np.unique(normalized)[:5]}...')
            print(f'After _normalize: min={normalized.min():.3f}, max={normalized.max():.3f}, mean={normalized.mean():.3f}')
        break

# Now test with the actual dataset __getitem__
print('\n--- Testing LGGDataset __getitem__ ---')
from src.dataset import LGGDataset

pairs = [(str(f), str(f).replace('.tif', '_mask.tif')) 
         for f in sorted(raw_dir.glob('*/*.tif')) 
         if '_mask' not in str(f) and (str(f).replace('.tif', '_mask.tif'))]

# Use a real mask
for patient in sorted(raw_dir.iterdir())[:1]:
    if patient.is_dir():
        imgs = sorted(patient.glob('*.tif'))
        real_pairs = []
        for img_path in imgs[:2]:
            if '_mask' not in img_path.name:
                mask_path = str(img_path).replace('.tif', '_mask.tif')
                if Path(mask_path).exists():
                    real_pairs.append((str(img_path), mask_path))
        if real_pairs:
            ds = LGGDataset(real_pairs, image_size=256, augment=False, cache=True, filter_empty_masks=False)
            mask_tensor, image_tensor = ds[0]
            print(f'mask_tensor shape: {mask_tensor.shape}')
            print(f'mask_tensor unique: {mask_tensor.unique()}')
            print(f'mask_tensor min: {mask_tensor.min():.3f}, max: {mask_tensor.max():.3f}, mean: {mask_tensor.mean():.3f}')
            print(f'image_tensor shape: {image_tensor.shape}')
            print(f'image_tensor Ch means: {image_tensor.mean(dim=(1,2))}')
        break
