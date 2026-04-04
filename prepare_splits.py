"""
Pre-process and split the dataset locally.
Generates JSON files in data/splits/ that can be uploaded to Colab to skip slow scanning.

Usage:
    uv run python prepare_splits.py
"""

import os
import json
import random
from pathlib import Path
import tifffile

DATASET_DIR = Path('data/raw/lgg-mri-segmentation')
OUTPUT_DIR = Path('data/splits')
SEED = 42

def prepare():
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning dataset at {DATASET_DIR}...")
    
    # 1. Collect all valid (tumor) pairs
    all_pairs = []
    patients = []
    
    for patient_folder in sorted(DATASET_DIR.iterdir()):
        if not patient_folder.is_dir():
            continue
            
        patients.append(patient_folder.name)
        valid_count = 0
        
        for mask_file in sorted(patient_folder.glob("*_mask.tif")):
            # Fast check: does this mask have tumor?
            mask = tifffile.imread(str(mask_file))
            if mask.max() > 0:
                # Store relative path to ensure it works on Colab
                # e.g., "TCGA_CS_4941_19960909/mask.tif"
                rel_mask = mask_file.relative_to(DATASET_DIR).as_posix()
                rel_img = str(rel_mask).replace("_mask.tif", ".tif")
                all_pairs.append((rel_img, rel_mask))
                valid_count += 1
        
        print(f"  {patient_folder.name}: {valid_count} tumor slices")

    print(f"\nTotal tumor slices found: {len(all_pairs)}")
    print(f"Total patients: {len(patients)}")

    # 2. Patient-level split
    random.seed(SEED)
    random.shuffle(patients)

    n_train = int(len(patients) * 0.8)
    n_val = int(len(patients) * 0.1)
    # Rest goes to test

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train : n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    print(f"Split: {len(train_patients)} Train, {len(val_patients)} Val, {len(test_patients)} Test")

    # 3. Distribute pairs into splits
    splits = {'train': [], 'val': [], 'test': []}
    
    for img, mask in all_pairs:
        patient_name = Path(img).parts[0] # e.g., "TCGA_CS_4941_19960909"
        if patient_name in train_patients:
            splits['train'].append((img, mask))
        elif patient_name in val_patients:
            splits['val'].append((img, mask))
        else:
            splits['test'].append((img, mask))

    # 4. Save JSON files
    for name, data in splits.items():
        out_path = OUTPUT_DIR / f"{name}_split.json"
        with open(out_path, 'w') as f:
            json.dump(data, f)
        print(f"Saved {out_path} with {len(data)} pairs.")
        
    print("\nDone! Upload the JSON files in data/splits/ to your Google Drive.")

if __name__ == "__main__":
    prepare()
