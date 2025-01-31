"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


import utils.binvox_rw
import csv
import random
import cupy as cp
import os
from pathlib import Path
import matplotlib.pyplot as plt

#VOC_CLASSES = (  # always index 0
#    'aeroplane', 'bicycle', 'bird', 'boat',
#    'bottle', 'bus', 'car', 'cat', 'chair',
#    'cow', 'diningtable', 'dog', 'horse',
#    'motorbike', 'person', 'pottedplant',
#    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_CLASSES = ('O ring', 'Through hole', 'Blind hole', 
               'Triangular passage', 'Rectangular passage', 
               'Circular through slot', 'Triangular through slot', 
               'Rectangular through slot', 'Rectangular blind slot',
               'Triangular pocket', 'Rectangular pocket', 
               'Circular end pocket', 'Triangular blind step', 
               'Circular blind step', 'Rectangular blind step', 
               'Rectangular through step' , '2-sides through step', 
               'Slanted through step', 'Chamfer', 'Round', 
               'Vertical circular end blind slot', 
               'Horizontal circular end blind slot', 
               '6-sides passage', '6-sides pocket')
  

# note: if you used our download scripts, this should be right
#VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")


#class VOCAnnotationTransform(object):
#    """Transforms a VOC annotation into a Tensor of bbox coords and label index
#    Initilized with a dictionary lookup of classnames to indexes
#
#    Arguments:
#        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
#            (default: alphabetic indexing of VOC's 20 classes)
#        keep_difficult (bool, optional): keep difficult instances or not
#            (default: False)
#        height (int): height
#        width (int): width
#    """
#
#    def __init__(self, class_to_ind=None, keep_difficult=False):
#        self.class_to_ind = class_to_ind or dict(
#            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
#        self.keep_difficult = keep_difficult
#
#    def __call__(self, target, width, height):
#        """
#        Arguments:
#            target (annotation) : the target annotation to be made usable
#                will be an ET.Element
#        Returns:
#            a list containing lists of bounding boxes  [bbox coords, class name]
#        """
#        res = []
#        for obj in target.iter('object'):
#            difficult = int(obj.find('difficult').text) == 1
#            if not self.keep_difficult and difficult:
#                continue
#            name = obj.find('name').text.lower().strip()
#            bbox = obj.find('bndbox')
#
#            pts = ['xmin', 'ymin', 'xmax', 'ymax']
#            bndbox = []
#            for i, pt in enumerate(pts):
#                cur_pt = int(bbox.find(pt).text) - 1
#                # scale height or width
#                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
#                bndbox.append(cur_pt)
#            label_idx = self.class_to_ind[name]
#            bndbox.append(label_idx)
#            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
#            # img_id = target.find('filename').text[:-4]
#
#        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

def rotate_sample(sample, rotation, reverse=False):
    """Rotate sample using GPU acceleration
    Args:
        sample: 3D array to rotate
        rotation: rotation angle
        reverse: whether to reverse rotation
    Returns:
        Rotated sample on GPU
    """
    # Move to GPU if not already there
    if not isinstance(sample, cp.ndarray):
        sample = cp.asarray(sample)
    
    # Pre-compute rotation operations
    rotations = {
        (1, True): lambda x: cp.rot90(x, k=1, axes=(2,1)),
        (2, True): lambda x: cp.rot90(x, k=2, axes=(1,2)),
        (3, True): lambda x: cp.rot90(x, k=1, axes=(1,2)),
        (4, True): lambda x: cp.rot90(x, k=3, axes=(1,2)),
        (5, True): lambda x: cp.rot90(x, k=1, axes=(0,2)),
        (1, False): lambda x: cp.rot90(x, k=1, axes=(1,2)),
        (2, False): lambda x: cp.rot90(x, k=2, axes=(1,2)),
        (3, False): lambda x: cp.rot90(x, k=1, axes=(2,1)),
        (4, False): lambda x: cp.rot90(x, k=3, axes=(1,2)),
        (5, False): lambda x: cp.rot90(x, k=1, axes=(2,0))
    }
    
    # Apply rotation if needed
    key = (rotation, reverse)
    if key in rotations:
        sample = rotations[key](sample)
    
    return sample


def rotate_sample24(sample):
    """Rotate sample using GPU acceleration
    Args:
        sample: 3D array to rotate
    Returns:
        Rotated sample on GPU
    """
    # Move to GPU if not already there
    if not isinstance(sample, cp.ndarray):
        sample = cp.asarray(sample)
    
    # Random rotation strategy
    rotation = random.randint(0,23)
    
    # Pre-compute rotation matrices for better performance
    rotations = {
        1: lambda x: cp.rot90(x, 1, (1,2)),
        2: lambda x: cp.rot90(x, 2, (1,2)),
        3: lambda x: cp.rot90(x, 1, (2,1)),
        4: lambda x: cp.rot90(x, 1, (0,1)),
        5: lambda x: cp.rot90(cp.rot90(x, 1, (0,1)), 1, (1,2)),
        6: lambda x: cp.rot90(cp.rot90(x, 1, (0,1)), 2, (1,2)),
        7: lambda x: cp.rot90(cp.rot90(x, 1, (0,1)), 1, (2,1)),
        8: lambda x: cp.rot90(x, 1, (1,0)),
        9: lambda x: cp.rot90(cp.rot90(x, 1, (1,0)), 1, (1,2)),
        10: lambda x: cp.rot90(cp.rot90(x, 1, (1,0)), 2, (1,2)),
        11: lambda x: cp.rot90(cp.rot90(x, 1, (1,0)), 1, (2,1)),
        12: lambda x: cp.rot90(x, 2, (1,0)),
        13: lambda x: cp.rot90(cp.rot90(x, 2, (1,0)), 1, (1,2)),
        14: lambda x: cp.rot90(cp.rot90(x, 2, (1,0)), 2, (1,2)),
        15: lambda x: cp.rot90(cp.rot90(x, 2, (1,0)), 1, (2,1)),
        16: lambda x: cp.rot90(x, 1, (0,2)),
        17: lambda x: cp.rot90(cp.rot90(x, 1, (0,2)), 1, (1,2)),
        18: lambda x: cp.rot90(cp.rot90(x, 1, (0,2)), 2, (1,2)),
        19: lambda x: cp.rot90(cp.rot90(x, 1, (0,2)), 1, (2,1)),
        20: lambda x: cp.rot90(x, 1, (2,0)),
        21: lambda x: cp.rot90(cp.rot90(x, 1, (2,0)), 1, (1,2)),
        22: lambda x: cp.rot90(cp.rot90(x, 1, (2,0)), 2, (1,2)),
        23: lambda x: cp.rot90(cp.rot90(x, 1, (2,0)), 1, (2,1))
    }
    
    # Apply rotation if needed
    if rotation in rotations:
        sample = rotations[rotation](sample)
    
    return sample


def get_label_from_csv(filename):
    
    retarr = np.zeros((0,7))
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            items = row[0].split(',')
            retarr = np.insert(retarr,0,np.asarray(items),0)
            
    
    return retarr[:,6]


def achieve_legal_model(list_IDs, list_size, factor):
    
    while True:
        filename = list_IDs[random.randint(0,len(list_IDs)-1)]
        
        # Convert Windows path to Unix-style for minfo.csv lookup
        unix_path = str(filename).replace('\\', '/')
        cur_factor = list_size[unix_path]
        
        if cur_factor >= factor:
            continue
        
        label = int(os.path.basename(filename).split('_')[0])
        with open(filename, 'rb') as f:
            model = utils.binvox_rw.read_as_3d_array(f).data
        
        return label, model
    



def achieve_random_model(list_IDs, list_size, grid_size=128):  
    """Create a random model with multiple features using GPU acceleration
    Args:
        list_IDs: list of model IDs
        list_size: dictionary of model sizes
        grid_size: size of the grid (64 or 128)
    """
    # Adjusted for 128 grid - allowing fewer features since each is 8x larger
    num_features = random.randint(2,6)  # Reduced max features from 10 to 6
    factor = 152 - 12*num_features  # Doubled for 128 grid
    
    # Initialize arrays directly on GPU
    components = cp.zeros((num_features, grid_size, grid_size, grid_size), dtype=cp.bool_)
    ret_model = cp.ones((grid_size, grid_size, grid_size), dtype=cp.bool_)
    model_label = cp.zeros(0, dtype=cp.int32)
    
    # Create memory pool for efficient GPU memory management
    with cp.cuda.Device(0):
        memory_pool = cp.get_default_memory_pool()
        
        for i in range(num_features):
            label, model = achieve_legal_model(list_IDs, list_size, factor)
            model_label = cp.append(model_label, label)
            
            # Move model to GPU if it's not already there
            if not isinstance(model, cp.ndarray):
                model = cp.asarray(model)
            
            # If model is 64 grid but we need 128, upsample it efficiently
            if model.shape[0] == 64 and grid_size == 128:
                model_128 = cp.repeat(cp.repeat(cp.repeat(model, 2, axis=0), 2, axis=1), 2, axis=2)
                components[i] = rotate_sample24(model_128)
            else:
                components[i] = rotate_sample24(model)
                
            ret_model = ret_model & components[i]  # Use & for boolean arrays
            
            # Free up GPU memory after each iteration
            memory_pool.free_all_blocks()
            
    # Move results back to CPU
    return cp.asnumpy(ret_model), cp.asnumpy(model_label), cp.asnumpy(components)


    
def create_img(obj3d, rotation, grayscale=False, grid_size=64):
    """Create an image from 3D object data using GPU acceleration
    Args:
        obj3d: 3D object data
        rotation: rotation angle
        grayscale: whether to create grayscale image
        grid_size: size of the grid (64 or 128)
    """
    # Move data to GPU if not already there
    if not isinstance(obj3d, cp.ndarray):
        obj3d = cp.asarray(obj3d)
    
    # If input is 64 grid but we need 128, upsample it
    if obj3d.shape[0] == 64 and grid_size == 128:
        # Use GPU for upsampling - more efficient than loops
        obj3d_128 = cp.repeat(cp.repeat(cp.repeat(obj3d, 2, axis=0), 2, axis=1), 2, axis=2)
        cursample = obj3d_128
    else:
        cursample = obj3d.copy()
    
    cursample = rotate_sample(cursample, rotation)
    
    # Use GPU to find first True value along depth axis
    depth_mask = cursample.astype(cp.int8)  # Convert bool to int for argmax
    img0 = cp.zeros((cursample.shape[1], cursample.shape[2]), dtype=cp.float32)
    
    # Find indices of first True value along depth axis
    first_true = cp.argmax(depth_mask, axis=0)
    # Create mask for cases where no True value exists
    no_true_mask = ~cp.any(depth_mask, axis=0)
    
    # Set values based on first True position
    img0[~no_true_mask] = first_true[~no_true_mask].astype(cp.float32) / cursample.shape[0]
    img0[no_true_mask] = 1.0
    
    if img0.mean() == 0:
        flag = False
    else:
        flag = True
                    
    if not grayscale:
        # Use GPU for stacking
        img = cp.stack((img0, img0, img0), axis=2)
    else:
        img = img0
    
    img = img * 255
    
    # Move result back to CPU for matplotlib
    return cp.asnumpy(img), flag

def achieve_model_gt(model_label, model_components, rotation):
    
    img_label = np.zeros((10,6))
    img_nbox = 0
    
    for i in range(len(model_label)):
        cur_component = rotate_sample(model_components[i,:,:,:], rotation)
        
        region = np.where(cur_component[0,:,:]==0)
        
        if len(region[0]) == 0:
            continue
        
        img_label[img_nbox,5] = model_label[i] #+ 1
        img_label[img_nbox,0] = (region[1].min())/64.0
        img_label[img_nbox,1] = (region[0].min())/64.0
        img_label[img_nbox,2] = (region[1].max()+1)/64.0
        img_label[img_nbox,3] = (region[0].max()+1)/64.0
        
        region = np.where(cur_component[:,:,:]==0)
        img_label[img_nbox,4] = (region[0].max())/64.0
        img_nbox += 1
    
    return img_label[0:img_nbox,:]

def create_partition(num_train_per_class = 30, num_val_per_class = 30):
    
    num_classes = 24
    counter = cp.zeros(num_classes)
    partition = {}
    for i in range(num_classes): 
        partition['train',i] = []
        partition['val',i] = []
        
#    for i in range(1,12):
#        partition['test',i] = []
        
    with open(os.devnull, 'w') as devnull:
        for filename in Path('data/FNSet/').glob('*.binvox'):
            namelist = os.path.basename(filename).split('_')
            
            
            label = int(namelist[0])
                
            counter[label] += 1
            
            items = [filename]
            
            if counter[label] % 10 < 9:
                partition['train',label] += items
            elif counter[label] % 10 == 9:
                partition['val',label] += items
    

    
    ret = {}
    ret['train'] = []
    ret['val'] = []
    
#    for testidx in range(1,12):
#        ret['test', testidx] = partition['test', testidx]
    
           
    for i in range(num_classes):      
        random.shuffle(partition['train',i])  
        random.shuffle(partition['val',i])  
        
        ret['train'] += partition['train',i][0:num_train_per_class]
        ret['val'] += partition['val',i][0:num_val_per_class]
    
    random.shuffle(ret['train'])  
    random.shuffle(ret['val'])  
        
    return ret

def create_test(testidx):
    partition = []
    with open(os.devnull, 'w') as devnull:
        for filename in Path('data/MulSet/set' + str(testidx) + '/').glob('*.binvox'):
            partition += [filename]
    return partition


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, list_IDs = None, transform=None, phase = 'train'):
        
        self.transform = transform
        self.list_IDs = list_IDs
        self.phase = phase
        
        if phase == 'test':
            self.num_samples = len(list_IDs)*6
            #self.is_test = True
        elif phase == 'val':
            self.DIR = 'data/ValSet/'
            self.num_samples = int(len([name for name in os.listdir(self.DIR) if os.path.isfile(os.path.join(self.DIR, name))])/2)
            
        elif phase == 'train':
            self.DIR = 'data/TrSet/'
            self.num_samples = int(len([name for name in os.listdir(self.DIR) if os.path.isfile(os.path.join(self.DIR, name))])/2)
            
            #self.num_samples = 143469
            #self.is_test = False
        
#        with open('data/minfo.csv', mode='r') as infile:
#            reader = csv.reader(infile)
#            self.list_size = {rows[0]:int(rows[1]) for rows in reader}



    def __getitem__(self, idx):
        
        
        if self.phase == 'test':
            rotation = int(idx%6)
            m_idx = int(idx/6)
        
            if rotation == 0:
                filename = self.list_IDs[m_idx]
                with open(filename, 'rb') as f:
                    self.cur_model = utils.binvox_rw.read_as_3d_array(f).data
                    self.cur_model_label = get_label_from_csv(str(filename).replace('.binvox','.csv'))
                    
            img, _ = create_img(self.cur_model, rotation)
            
            img, _, _ = self.transform(img, 0, 0)
            
        
        else: #train, val
            
            filename = self.DIR+str(idx)
            img = plt.imread(filename+'.png',format='grayscale')
            target=np.load(filename+'.npy')
                
            img = img[:,:,:3]
            
            
            #da strategy 3
            if self.phase == 'train' and random.randint(0,1) == 0:

                
                filename2 = self.DIR+str(random.randint(0,self.num_samples-1))
                img2 = plt.imread(filename2+'.png',format='grayscale')
                target2=np.load(filename2+'.npy')
                    
                img2 = img2[:,:,:3]
                
                target = np.concatenate((target,target2),axis=0)
                
#                org_img = img[:,:,0]/255
#                tmp = np.ones((66,66))
#                tmp[1:65,1:65] = org_img
#                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#                ax.imshow(tmp, cmap='gray', vmin=0, vmax=1)
#                plt.show()
                
                img = np.maximum(img,img2)
                
#                org_img = img2[:,:,0]/255
#                tmp = np.ones((66,66))
#                tmp[1:65,1:65] = org_img
#                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#                ax.imshow(tmp, cmap='gray', vmin=0, vmax=1)
#                plt.show()
#                
#                org_img = img[:,:,0]/255
#                tmp = np.ones((66,66))
#                tmp[1:65,1:65] = org_img
#                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#                ax.imshow(tmp, cmap='gray', vmin=0, vmax=1)
#                plt.show()
                #print(img.max())
                #print(img.min())
                
                
            
            img, boxes, labels = self.transform(img, target[:, :5], target[:, 5])
            
            self.cur_model_label = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
            
        return torch.from_numpy(img).permute(2, 0, 1).float(), self.cur_model_label

    def __len__(self):
        return self.num_samples
