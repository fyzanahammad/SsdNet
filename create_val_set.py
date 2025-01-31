from data import *
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
from tqdm import tqdm

# ================= MODIFIED PARAMETERS =================
num_samples = 7200  # 24 classes × 50 models × 6 rotations = 7200 samples

# Manually create list_IDs with 50 models per class
list_IDs = []
for class_id in range(24):          # 24 classes (0 to 23)
    for model_num in range(1, 51):  # 50 models per class (1 to 50)
        model_id = f"{class_id}_{model_num}"
        list_IDs.append(model_id)

DIR = 'data/ValTestSet/'  # New folder for validation data
start_counter = 0         # Start saving from 0.png/.npy
# =======================================================

# Set random seeds for reproducibility
random.seed(1024)
cp.random.seed(1024)

print("\nStarting validation set creation...")
print(f"Will generate approximately {num_samples} samples ({len(list_IDs)} models × 6 rotations)")
print(f"Output directory: {DIR}")
print(f"Batch size: 10 samples")

# Create directory if missing
print("\nChecking/creating output directory...")
os.makedirs(DIR, exist_ok=True)

print(f"Starting from counter: {start_counter}")

# Load model metadata
print("\nLoading model metadata from CSV...")
with open('data/minfo.csv', mode='r') as infile:
    reader = csv.reader(infile)
    list_size = {rows[0]:int(rows[1]) for rows in reader}

# Initialize GPU memory pool
print("\nInitializing GPU memory pool...")
pool = cp.get_default_memory_pool()
with cp.cuda.Device(0):
    batch_size = 10  # Adjust based on GPU memory
    counter = start_counter
    total_skipped = 0
    start_time = time.time()
    
    # Progress bar with tqdm
    with tqdm(total=num_samples, desc="Creating validation samples") as pbar:
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")
            
            for idx in range(batch_start, batch_end):
                rotation = idx % 6  # 6 rotations per model
                m_idx = idx // 6
                
                # Fetch new model every 6th sample
                if rotation == 0:
                    print(f"  Loading new model ({m_idx + 1}/{len(list_IDs)})")
                    cur_model, cur_model_label, cur_model_components = achieve_random_model(
                        list_IDs, list_size
                    )
                
                # Generate data
                img, _ = create_img(cur_model, rotation, True)
                target = achieve_model_gt(cur_model_label, cur_model_components, rotation)
                
                if target.shape[0] == 0:
                    print(f"  [SKIP] Model {cur_model_label}, Rotation {rotation} - No valid labels")
                    total_skipped += 1
                    continue
                
                # Save files
                filename = os.path.join(DIR, str(counter))
                plt.imsave(filename + '.png', img, cmap='gray', vmin=0, vmax=255)
                np.save(filename + '.npy', target)
                counter += 1
                pbar.update(1)
            
            # Free GPU memory
            pool.free_all_blocks()
            print(f"  Freed GPU memory after batch")

            # Print batch statistics
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / ((batch_start//batch_size) + 1)
            remaining_batches = ((num_samples - batch_end) + batch_size - 1) // batch_size
            est_time_remaining = remaining_batches * avg_time_per_batch
            
            print(f"  Batch Stats:")
            print(f"    - Elapsed time: {elapsed_time/60:.1f} minutes")
            print(f"    - Est. remaining: {est_time_remaining/60:.1f} minutes")
            print(f"    - Samples generated: {counter - start_counter}")
            print(f"    - Samples skipped: {total_skipped}")

print("\nValidation set creation completed!")
print(f"Total samples generated: {counter - start_counter}")
print(f"Total samples skipped: {total_skipped}")
print(f"Total time taken: {(time.time() - start_time)/60:.1f} minutes")