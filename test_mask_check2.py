import tifffile
import numpy as np
from pathlib import Path

raw_dir = Path('data/raw/lgg-mri-segmentation')
print(f'raw_dir exists: {raw_dir.exists()}')
patients = sorted(raw_dir.iterdir())
print(f'Found {len(patients)} patient dirs')

for patient in patients[:2]:
    if patient.is_dir():
        print(f'\nPatient: {patient.name}')
        masks = list(patient.glob('*_mask.tif'))
        print(f'  Found {len(masks)} masks')
        if masks:
            m = tifffile.imread(str(masks[0]))
            print(f'  Mask shape: {m.shape}, dtype: {m.dtype}')
            print(f'  Mask unique values (up to 10): {np.unique(m)[:10]}')
            print(f'  Mask min: {m.min()}, max: {m.max()}, mean: {m.mean():.2f}')
            
            # What happens after _normalize?
            normalized = (m.astype(np.float32) / 127.5) - 1.0
            uniq_norm = np.unique(normalized)
            print(f'  After _normalize unique (up to 10): {uniq_norm[:10]}')
            print(f'  After _normalize: min={normalized.min():.3f}, max={normalized.max():.3f}, mean={normalized.mean():.3f}')
        break
