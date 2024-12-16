#! /usr/bin/env python3
#%%
import pathlib
import cv2
import math
from PIL import Image
import matplotlib.pyplot as plt
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

def ndc_to_depth(ndc, map_xy):
    nc_z = 0.01
    fc_z = 600.0
    lower = ndc + (nc_z / (2 * fc_z)) * map_xy
    return map_xy / lower

def create_nc_xy_map(rows, cols, nc_z, fov_v):
    nc_h = 2 * nc_z * math.tan((fov_v * math.pi) / 360.0)
    nc_w = (cols / rows) * nc_h
    nc_xy_map = np.zeros((rows, cols))
    for j in range(rows):
        for i in range(cols):
            nc_x = abs(((2 * i) / (cols - 1.0)) - 1) * nc_w / 2.0
            nc_y = abs(((2 * j) / (rows - 1.0)) - 1) * nc_h / 2.0
            nc_xy_map[j, i] = math.sqrt(pow(nc_x, 2) + pow(nc_y, 2) + pow(nc_z, 2))
    return nc_xy_map

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
        
        # Focal length of camera
        focal_length = 960

        # Projection centre for Y-axis
        cy = 540
        
        # Size params
        rows = 1080
        cols = 1920
        resize_shape = (1920, 1080)

        # Depth clip ranges
        nc_z = 0.01
        fc_z = 600.0

        # Field of view
        fov_v = 59

        # Max depth
        max_depth = 400

        # Looping through data
        for index in range(0, 1):

            print(f'Processing image {index} of {num_images-1}')

            # Open images and pre-existing masks
            image = Image.open(str(images[index]))

            with open(str(depth_maps[index]), 'rb') as fd:
                f = np.fromfile(fd, dtype=np.float32, count=rows * cols)
                im = f.reshape((rows, cols))

            nc_xy_map = create_nc_xy_map(rows, cols, nc_z, fov_v)
            depth_im_true = ndc_to_depth(im, nc_xy_map)
            depth_im_true[depth_im_true > 400] = 400
            depth_im_resized = cv2.resize(depth_im_true, resize_shape)

            plt.figure()
            plt.imshow(image)
            plt.figure()
            plt.imshow(depth_im_resized)

if __name__ == '__main__':
    main()
#%%