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

    # 1. Collect ALL pairs (tumor + background) and tumor-only pairs
    all_pairs = []         # All slices (tumor + background)
    tumor_pairs = []       # Tumor slices only
    patients = []

    for patient_folder in sorted(DATASET_DIR.iterdir()):
        if not patient_folder.is_dir():
            continue

        patients.append(patient_folder.name)
        tumor_count = 0
        total_count = 0

        for mask_file in sorted(patient_folder.glob("*_mask.tif")):
            total_count += 1
            mask = tifffile.imread(str(mask_file))
            rel_mask = mask_file.relative_to(DATASET_DIR).as_posix()
            rel_img = str(rel_mask).replace("_mask.tif", ".tif")
            all_pairs.append((rel_img, rel_mask))
            
            if mask.max() > 0:
                tumor_pairs.append((rel_img, rel_mask))
                tumor_count += 1

        print(f"  {patient_folder.name}: {tumor_count}/{total_count} tumor slices")

    print(f"\nTotal slices: {len(all_pairs)}")
    print(f"Tumor slices: {len(tumor_pairs)}")
    print(f"Background slices: {len(all_pairs) - len(tumor_pairs)}")
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
    # Train gets ALL slices (tumor + background) so GAN learns full brain anatomy
    # Val/Test get only tumor slices for consistent evaluation
    splits = {'train': [], 'val': [], 'test': []}

    # Train: all slices from train patients
    for img, mask in all_pairs:
        patient_name = Path(img).parts[0]
        if patient_name in train_patients:
            splits['train'].append((img, mask))

    # Val/Test: tumor slices only
    for img, mask in tumor_pairs:
        patient_name = Path(img).parts[0]
        if patient_name in val_patients:
            splits['val'].append((img, mask))
        elif patient_name in test_patients:
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
