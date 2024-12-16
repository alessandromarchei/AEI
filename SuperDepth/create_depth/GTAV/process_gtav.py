#! /usr/bin/env python3
#%%
import pathlib
import cv2
from PIL import Image
import numpy as np
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData
from SuperDepth.create_depth.common.height_map import HeightMap

def main():

    # Filepaths for data loading and savind
    root_data_path = '/mnt/media/GTAV/'
    root_save_path = '/mnt/media/SuperDepth/GTAV'


    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(root_data_path).glob("*/gta0/cam0/depth/*.bin")])
    images = sorted([f for f in pathlib.Path(root_data_path).glob("*/gta0/cam0/data/*.png")])

    print(len(depth_maps), len(images))

if __name__ == '__main__':
    main()
#%%