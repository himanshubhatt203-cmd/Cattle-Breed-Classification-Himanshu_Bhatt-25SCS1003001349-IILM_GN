""" Splitting using Stratified Splitting; ensures each split has same class ratio"""

import os
import shutil
import numpy as np
from collections import defaultdict
from torchvision import datasets

# ==== CHANGE THESE PATHS TO MATCH YOUR COMPUTER ====
DATA_DIR = r"C:\Users\himan\Downloads\AI Model\Processed Data"      # where your images are now
OUTPUT_DIR = r"C:\Users\himan\Downloads\AI Model\Splitted_Data"     # where you want train/val/test folders created
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== LOAD THE DATASET ====
base_dataset = datasets.ImageFolder(DATA_DIR)
class_names = base_dataset.classes   # list of breed/class names

# ==== GROUP INDICES BY CLASS ====
idxs_by_class = defaultdict(list)
for idx, (_, label) in enumerate(base_dataset.samples):
    idxs_by_class[label].append(idx)

# ==== SPLIT RATIOS (change if needed) ====
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# ==== SPLIT EACH CLASS ====
train_idx, val_idx, test_idx = [], [], []
rng = np.random.default_rng(seed=42)  # seed for reproducibility

for lbl, idxs in idxs_by_class.items():
    idxs = np.array(idxs)
    rng.shuffle(idxs)
    n = len(idxs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Make sure all images are used
    n_test = n - n_train - n_val
    train_idx.extend(idxs[:n_train].tolist())
    val_idx.extend(idxs[n_train:n_train + n_val].tolist())
    test_idx.extend(idxs[n_train + n_val:].tolist())

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# Helper to get (file_path, class_idx) for each index
all_samples = base_dataset.samples

def copy_files(indices, split_name):
    for idx in indices:
        img_path, class_idx = all_samples[idx]
        class_name = class_names[class_idx]
        # Make new folder for this class in this split, if needed
        dest_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        # Destination path
        dest_path = os.path.join(dest_dir, os.path.basename(img_path))
        shutil.copy2(img_path, dest_path)  # copy2 preserves metadata

print("Copying train files...")
copy_files(train_idx, "train")
print("Copying validation files...")
copy_files(val_idx, "val")
print("Copying test files...")
copy_files(test_idx, "test")
print("Splitting and saving complete!")