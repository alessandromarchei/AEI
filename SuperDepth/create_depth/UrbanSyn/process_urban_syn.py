#! /usr/bin/env python3
#%%
import pathlib
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData
from SuperDepth.create_depth.common.height_map import HeightMap
from SuperDepth.create_depth.common.depth_boundaries import DepthBoundaries

def removeExtraSamples(depth_filepath, depth_maps, images_filepath, images):
    
    depth_path = depth_filepath + 'depth_'
    depth_map_indices = []

    for i in range(0, len(depth_maps)):
        depth_map_file = str(depth_maps[i])
        depth_map_index = depth_map_file[:-4]
        depth_map_index = depth_map_index.replace(depth_path,'')
        depth_map_indices.append(depth_map_index)

    image_path = images_filepath + 'rgb_'
    filtered_images = []

    for i in range(0, len(images)):
        image_file = str(images[i])
        image_index = image_file[:-4]
        image_index = image_index.replace(image_path,'')
        
        if(image_index in depth_map_indices):
            filtered_images.append(images[i])

    return filtered_images


def createDepthMap(depth_data):

    # Getting size of depth data
    size = depth_data.shape
    height = size[0]
    width = size[1]

    # Converting depth data to metric depth values
    depth_map = Image.fromarray(depth_data)
    depth_map = np.asarray(depth_map, dtype=np.float32)
    depth_map = 100000 * depth_map

    # Removing erroneous depth data
    max_depth = np.max(depth_map)
    for i in range(0, height):
        for j in range(0, width):
            if (depth_map[i,j] <= 0):
                depth_map[i,j] = max_depth

    return depth_map

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
            if(grad > 25):
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
    root_data_path = '/mnt/media/UrbanSyn/'
    root_save_path = '/mnt/media/SuperDepth/UrbanSyn'

    # Paths to read ground truth depth and input images from training data
    depth_filepath = root_data_path + 'depth/'
    images_filepath = root_data_path + 'rgb/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*.exr")])
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

    # Remove extra samples
    images = removeExtraSamples(depth_filepath, depth_maps, images_filepath, images)
 
    # If all data checks have been passed
    num_depth_maps = len(depth_maps)
    num_images = len(images)

    check_data = CheckData(num_images, num_depth_maps)
    check_passed = check_data.getCheck()
    
    if(check_passed):

        print('Beginning processing of data')

        # Focal length of camera
        image_height_mm = 2.90000009537
        focal_length_mm = 4.900001049041748
        focal_length = (focal_length_mm/image_height_mm)*1024

        # Projection centre for Y-axis
        cy = 512
        # Camera mounting height above ground
        camera_height = 1.2

        # Height map limits
        max_height = 7
        min_height = -2

        # Looping through data
        for index in range(0, num_images):

            print(f'Processing image {index} of {num_images-1}')
            
            # Open images and pre-existing masks
            image = Image.open(str(images[index]))
            depth_data = cv2.imread(str(depth_maps[index]), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            # Adding check in case data is corrupted
            if depth_data is not None:
                
                # Create metric depth map and height map
                depth_map = createDepthMap(depth_data)

                # Depth boundaries
                boundary_threshold = 10
                depthBoundaries = DepthBoundaries(depth_map, boundary_threshold)
                depth_boundaries = depthBoundaries.getDepthBoundaries()

                # Height map
                heightMap = HeightMap(depth_map, max_height, min_height, 
                 camera_height, focal_length, cy)
                height_map = heightMap.getHeightMap()

                # Sparse supverision
                sparse_supervision = createSparseSupervision(image, height_map, max_height, min_height)
                
                # Save files
                # RGB Image as PNG
                image_save_path = root_save_path + '/image/' + str(index) + '.png'
                image.save(image_save_path, "PNG")

                # Depth map as binary file in .npy format
                depth_save_path = root_save_path + '/depth/' + str(index) + '.npy'
                np.save(depth_save_path, depth_map)

                # Height map as binary file in .npy format
                height_save_path = root_save_path + '/height/' + str(index) + '.npy'
                np.save(height_save_path, height_map)

                # Sparse supervision map as binary file in .npy format
                supervision_save_path = root_save_path + '/supervision/' + str(index) + '.npy'
                np.save(supervision_save_path, sparse_supervision)

                # Boundary mask as PNG
                boundary_save_path = root_save_path + '/boundary/' + str(index) + '.png'
                boundary_mask = Image.fromarray(depth_boundaries)
                boundary_mask.save(boundary_save_path, "PNG")

                # Height map plot for data auditing purposes
                height_plot_save_path = root_save_path + '/height_plot/' + str(index) + '.png'
                plt.imsave(height_plot_save_path, height_map, cmap='inferno_r')
                
        print('----- Processing complete -----') 
      

if __name__ == '__main__':
    main()
#%%