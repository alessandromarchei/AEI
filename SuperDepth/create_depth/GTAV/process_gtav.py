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
    # Depth clip ranges
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

def findDepthBoundaries(depth_map):

    # Getting size of depth map
    size = depth_map.shape
    height = size[0]
    width = size[1]

    # Initializing depth boundary mask
    depth_boundaries = np.zeros_like(depth_map, dtype=np.uint8)

    # Fiding depth boundaries
    for i in range(1, height-1):
        for j in range(1, width-1):

            # Finding derivative
            x_grad = depth_map[i-1,j] - depth_map[i+1, j]
            y_grad = depth_map[i,j-1] - depth_map[i, j+1]
            grad = abs(x_grad) + abs(y_grad)
            
            # Derivative threshold
            if(grad > 10):
                depth_boundaries[i,j] = 255

    return depth_boundaries

def createSparseSupervision(image, height_map, max_height, min_height):

    # Getting size of height map
    size = height_map.shape
    height = size[0]
    width = size[1]

    # Initializing depth boundary mask
    sparse_supervision = np.zeros_like(height_map)

    # Getting pixel access for image
    px = image.load()

    # Fiding image gradients
    for i in range(1, width-1):
        for j in range(1, height-1):

            # Finding image derivative - using green colour channel
            x_grad = px[i-1,j][1] - px[i+1, j][1]
            y_grad = px[i,j-1][1] - px[i, j+1][1]
            grad = abs(x_grad) + abs(y_grad)

            # Derivative threshold to find likely stereo candidates
            if(grad > 50):
                sparse_supervision[j,i] = height_map[j,i]
            else:
                sparse_supervision[j,i] = max_height

            # Max height threshold
            if(height_map[j,i] == max_height):
                sparse_supervision[j,i] = max_height
    
    # Clipping height values for dataset
    sparse_supervision = sparse_supervision.clip(min = min_height, max = max_height)
    return sparse_supervision

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
        max_depth = 1000

        # Camera mounting height above ground
        camera_height = 2.1

        # Height map limits
        max_height = 7
        min_height = -5

        # ID save counter
        counter = 0

        # Looping through data
        for index in range(0, num_images, 3):

            print(f'Processing image {index} of {num_images-1}')

            # Open image
            image = Image.open(str(images[index]))

            # Open and process depth
            with open(str(depth_maps[index]), 'rb') as fd:
                f = np.fromfile(fd, dtype=np.float32, count=rows * cols)
                im = f.reshape((rows, cols))

            nc_xy_map = create_nc_xy_map(rows, cols, nc_z, fov_v)
            depth_im_true = ndc_to_depth(im, nc_xy_map)
            depth_im_true[depth_im_true > max_depth] = max_depth
            depth_map = cv2.resize(depth_im_true, resize_shape)

            # Depth boundaries
            depth_boundaries = findDepthBoundaries(depth_map)

            # Height map
            heightMap = HeightMap(depth_map, max_height, min_height, 
                 camera_height, focal_length, cy)
            minimum_height = heightMap.getMinimumHeight()
            height_map = heightMap.getHeightMap()

            # Sparse supervision
            sparse_supervision = createSparseSupervision(image, height_map, max_height, min_height)

            if(minimum_height > -8):
                

                # Save files
                # RGB Image as PNG
                image_save_path = root_save_path + '/image/' + str(counter) + '.png'
                image.save(image_save_path, "PNG")

                # Depth map as binary file in .npy format
                depth_save_path = root_save_path + '/depth/' + str(counter) + '.npy'
                np.save(depth_save_path, depth_map)

                # Height map as binary file in .npy format
                height_save_path = root_save_path + '/height/' + str(counter) + '.npy'
                np.save(height_save_path, height_map)

                # Sparse supervision map as binary file in .npy format
                supervision_save_path = root_save_path + '/supervision/' + str(counter) + '.npy'
                np.save(supervision_save_path, sparse_supervision)

                # Boundary mask as PNG
                boundary_save_path = root_save_path + '/boundary/' + str(counter) + '.png'
                boundary_mask = Image.fromarray(depth_boundaries)
                boundary_mask.save(boundary_save_path, "PNG")

                # Height map plot for data auditing purposes
                height_plot_save_path = root_save_path + '/height_plot/' + str(counter) + '.png'
                plt.imsave(height_plot_save_path, height_map, cmap='inferno_r')

                # Increment ID save counter
                counter += 1
                
        print('----- Processing complete -----') 

if __name__ == '__main__':
    main()
#%%