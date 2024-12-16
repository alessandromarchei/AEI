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

def removeExtraSamples(depth_maps, images):
 
    depth_map_indices = []

    for i in range(0, len(depth_maps)):
        depth_map_file = str(depth_maps[i])
        depth_map_index = depth_map_file[-10:-4]
        log = depth_map_file[:-21]
        file = log + depth_map_index
        depth_map_indices.append(file)

    filtered_images = []

    for i in range(0, len(images)):
        image_file = str(images[i])
        image_index = image_file[-10:-4]
        log = image_file[:-20]
        file = log + image_index
        if(file in depth_map_indices):
            filtered_images.append(images[i])

    return filtered_images

def main():

    # Filepaths for data loading and savind
    root_data_path = '/mnt/media/GTAV/'
    root_save_path = '/mnt/media/SuperDepth/GTAV'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(root_data_path).glob("*/gta0/cam0/depth/*.bin")])
    images = sorted([f for f in pathlib.Path(root_data_path).glob("*/gta0/cam0/data/*.png")])

    print(len(images), len(depth_maps))

    # Removing extra samples
    images = removeExtraSamples(depth_maps, images)

    # If all data checks have been passed
    num_depth_maps = len(depth_maps)
    num_images = len(images)

    check_data = CheckData(num_images, num_depth_maps)
    check_passed = check_data.getCheck()
    
    if(check_passed):

        print('Beginning processing of data')

if __name__ == '__main__':
    main()
#%%