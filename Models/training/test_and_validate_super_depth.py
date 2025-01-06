#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import sys
sys.path.append('..')
from data_utils.load_data_super_depth import LoadDataSuperDepth
from training.super_depth_trainer import SuperDepthTrainer

def main():

    # Root path
    root = '/mnt/media/SuperDepth/'
    
    # Data paths
    # MUAD
    muad_labels_filepath= root + 'MUAD/height/'
    muad_images_filepath = root + 'MUAD/image/'

    # URBANSYN
    urbansyn_labels_fileapath = root + 'UrbanSyn/height/'
    urbansyn_images_fileapath = root + 'UrbanSyn/image/'

if __name__ == '__main__':
    main()
# %%
