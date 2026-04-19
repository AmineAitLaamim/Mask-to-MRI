import re

path = r"C:\Users\pc\Desktop\cours\S6\Vision par Ordinateur\Mask-to-MRI\notebooks\experiment_B_train_colab.ipynb"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace synthetic data extraction and path
content = content.replace(
    'SYNTHETIC_DIR = f"{DRIVE_DIR}/dataset/synthetic_data"',
    'SYNTHETIC_DIR = f"{COLAB_WORKING}/Mask-to-MRI/dataset/synthetic_data"',
)
content = content.replace(
    'shutil.unpack_archive(synth_zip, f"{DRIVE_DIR}/dataset")',
    'shutil.unpack_archive(synth_zip, f"{COLAB_WORKING}/Mask-to-MRI/dataset")',
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Done - Changed to local Colab storage")
