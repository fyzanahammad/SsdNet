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

# Original number of samples for 64 grid
#num_samples = 2900000

# Reduced number of samples for 128 grid since each sample is 4x larger (128x128 vs 64x64)
# This helps manage memory and training time while maintaining similar total data volume
num_samples = 725000  # 2900000/4 rounded to nearest thousand

random.seed(1024)
# Original partition for 64 grid
#partition = create_partition(512, 100)

# Modified partition for 128 grid - reduced samples per class to manage memory
partition = create_partition(128, 25)  # Reduced from 512 to 128 samples per class
list_IDs = partition['train']

# Original directory for 64 grid data
#DIR = 'data/TrSet/'

# Directory for training data
DIR = 'data/TrSet/'
if not os.path.exists(DIR):
    os.makedirs(DIR)  # Create directory if it doesn't exist

start_counter = int(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])/2)
random.seed(start_counter)     

print(f"Starting from counter: {start_counter}")      

with open('data/minfo.csv', mode='r') as infile:
    reader = csv.reader(infile)
    list_size = {rows[0]:int(rows[1]) for rows in reader}

counter = start_counter
for idx in range(num_samples):
    rotation = int(idx%6)
    m_idx = int(idx/6)
    if rotation == 0:
        # Pass grid_size=128 to achieve_random_model
        cur_model, cur_model_label, cur_model_components = achieve_random_model(list_IDs, list_size, grid_size=128)
            
    # Original image creation for 64 grid
    #img, _ = create_img(cur_model, rotation, True)        
    
    # Modified image creation for 128 grid - the create_img function now handles 128 grid
    img, _ = create_img(cur_model, rotation, True, grid_size=128)  # Added grid_size parameter
    target = achieve_model_gt(cur_model_label, cur_model_components, rotation)
    
    if target.shape[0] == 0:
        continue
    
    print('Processing sample', idx+1, 'of', num_samples, '...')
    filename = DIR+str(counter)
    plt.imsave(filename+'.png',img,cmap='gray',vmin=0,vmax=255)
    np.save(filename+'.npy',target)
    counter = counter + 1