from data import *
from data.voc0712 import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import numpy as np
import cupy as cp
import csv
import matplotlib.pyplot as plt
import random


# ================= MODIFIED PARAMETERS =================
num_samples = 7200  # 24 classes × 50 models × 6 rotations = 7200 samples

# Manually create list_IDs with 50 models per class (0_1 to 0_50, 1_1 to 1_50, etc.)
list_IDs = []
for class_id in range(24):
    for model_num in range(1, 51):
        # Add full path and .binvox extension
        model_id = f"data/FNSet/{class_id}_{model_num}.binvox"
        list_IDs.append(model_id)

DIR = 'data/TestSet/'  # Save to a new folder to avoid conflicts
start_counter = 0      # Start saving from 0.png and 0.npy
# =======================================================

random.seed(1024)  # For reproducibility

print("Starting training set creation...")
print(f"Will generate approximately {num_samples} samples ({len(list_IDs)} models × 6 rotations)")

# Load model metadata from CSV
print("\nLoading model metadata from CSV...")
with open('data/minfo_test.csv', mode='r') as infile:
    reader = csv.reader(infile)
    list_size = {rows[0]: int(rows[1]) for rows in reader}

print(f"\nCreating output directory: {DIR}")
# Create output directory if it doesn't exist
os.makedirs(DIR, exist_ok=True)

counter = start_counter
total_skipped = 0
start_time = time.time()

print("\nStarting sample generation...")
for idx in range(num_samples):
    rotation = idx % 6  # 6 rotations per model (0-5)
    m_idx = idx // 6    # Group index for the model
    
    # Fetch a new model every 6th sample
    if rotation == 0:
        print(f"\nProcessing new model ({m_idx + 1}/{len(list_IDs)})")
        cur_model, cur_model_label, cur_model_components = achieve_random_model(list_IDs, list_size)
    
    # Generate image and labels
    print(f"  Generating rotation {rotation}/5 for model {cur_model_label}...", end='')
    img, _ = create_img(cur_model, rotation, True)
    target = achieve_model_gt(cur_model_label, cur_model_components, rotation)
    
    # Skip if no valid labels
    if target.shape[0] == 0:
        print(" [SKIPPED - No valid labels]")
        total_skipped += 1
        continue
    
    # Save files
    filename = os.path.join(DIR, str(counter))
    plt.imsave(filename + '.png', img, cmap='gray', vmin=0, vmax=255)
    np.save(filename + '.npy', target)
    
    print(f" [SAVED as {counter}.png/.npy]")
    counter += 1

    # Print progress every 100 samples
    if counter % 100 == 0:
        elapsed_time = time.time() - start_time
        avg_time_per_sample = elapsed_time / counter
        remaining_samples = num_samples - counter
        est_time_remaining = remaining_samples * avg_time_per_sample
        
        print(f"\nProgress: {counter}/{num_samples} samples generated")
        print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
        print(f"Estimated time remaining: {est_time_remaining/60:.1f} minutes")

print(f"\nDone! Generated {counter - start_counter} samples")
print(f"Skipped {total_skipped} samples due to invalid labels")
print(f"Total time taken: {(time.time() - start_time)/60:.1f} minutes")