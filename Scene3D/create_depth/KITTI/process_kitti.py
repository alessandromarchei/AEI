#! /usr/bin/env python3
#%%
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData
from Scene3D.create_depth.common.lidar_depth_fill import LidarDepthFill
from Scene3D.create_depth.common.height_map import HeightMap

def removeExtraSamples(image_folders):
    
    filtered_images_left = []
    filtered_images_right = []

    for i in range(0, len(image_folders)):
        images_left = sorted([f for f in pathlib.Path(str(image_folders[i])).glob("*image_02/*data/*.png")])
        images_right = sorted([f for f in pathlib.Path(str(image_folders[i])).glob("*image_03/*data/*.png")])
        for j in range(5, len(images_left) - 5):
            filtered_images_left.append(images_left[j])
            filtered_images_right.append(images_right[j])

    return filtered_images_left, filtered_images_right

def createDepthMap(depth_data):

    assert(np.max(depth_data) > 255)
    depth_map = depth_data.astype('float32') / 256.
    return depth_map


def cropData(image_left, depth_map_fill_only, height_map_fill_only, validity_mask):

    # Getting size of depth map
    size = depth_map_fill_only.shape
    height = size[0]
    width = size[1]

    # Cropping out those parts of data for which depth is unavailable
    image_left = image_left.crop((256, 0, width - 100, height))
    depth_map_fill_only = depth_map_fill_only[:, 256 : width - 100]
    height_map_fill_only = height_map_fill_only[:, 256 : width - 100]
    validity_mask = validity_mask[:, 256 : width - 100]

    return image_left, depth_map_fill_only, height_map_fill_only, validity_mask


def main():

    # Argument parser for data root path and save path
    parser = ArgumentParser()
    parser.add_argument("-r", "--root", dest="root_data_path", help="path to root folder with input ground truth labels and images")
    parser.add_argument("-s", "--save", dest="root_save_path", help="path to folder where processed data will be saved")
    args = parser.parse_args()

    # Filepaths for data loading and savind
    root_data_path = args.root_data_path
    root_save_path = args.root_save_path


    # Paths to read ground truth depth and input images from training data
    depth_filepath = root_data_path + 'train/'
    images_filepath = root_data_path + 'data/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*/proj_depth/*groundtruth/*image_02/*.png")])
    image_folders = sorted([f for f in pathlib.Path(images_filepath).glob("*")])

    # Remove extra samples
    images_left, images_right = removeExtraSamples(image_folders)

    # If all data checks have been passed
    num_depth_maps = len(depth_maps)
    num_images_left = len(images_left)
    num_images_right = len(images_right)

    check_data_left = CheckData(num_images_left, num_depth_maps)
    check_data_right = CheckData(num_images_right, num_depth_maps)

    check_passed_left = check_data_left.getCheck()
    check_passed_right = check_data_right.getCheck()

    if(check_passed_left and check_passed_right):

        print('Beginning processing of data')

        # Focal length of camera
        focal_length = 645.24
        # Projection centre for Y-axis
        cy = 194.13
        # Camera mounting height above ground
        camera_height = 1.65

        # Height map limits
        max_height = 7
        min_height = -2

        # Looping through data with temporal downsampling to get frames every second
        counter = 0

        for index in range(0, num_depth_maps, 2):

            print(f'Processing image {index} of {num_depth_maps-1}')
            
            # Open images and pre-existing masks
            image_left = Image.open(str(images_left[index]))
            depth_data = np.array(Image.open(str(depth_maps[index])), dtype=int)

            # Create depth map
            sparse_depth_map = createDepthMap(depth_data)
            lidarDepthFill = LidarDepthFill(sparse_depth_map)
            depth_map_fill_only = lidarDepthFill.getDepthMapFillOnly()
        
            # Height map
            heightMapFillOnly = HeightMap(depth_map_fill_only, max_height, min_height, 
                camera_height, focal_length, cy)
            height_map_fill_only = heightMapFillOnly.getHeightMap()
            
            # Validity mask
            validity_mask = np.zeros_like(depth_map_fill_only)
            validity_mask[np.where(depth_map_fill_only != 0)] = 1

            # Crop side regions where depth data is missing
            image_left, depth_map_fill_only, height_map_fill_only, validity_mask = \
                cropData(image_left, depth_map_fill_only, height_map_fill_only, validity_mask)

            # Save files
            # RGB Image as PNG
            image_save_path = root_save_path + '/image/' + str(counter) + '.png'
            image_left.save(image_save_path, "PNG")

            # Depth map as binary file in .npy format
            depth_save_path = root_save_path + '/depth/' + str(counter) + '.npy'
            np.save(depth_save_path, depth_map_fill_only)

            # Height map as binary file in .npy format
            height_save_path = root_save_path + '/height/' + str(counter) + '.npy'
            np.save(height_save_path, height_map_fill_only)

            # Validity mask as black and white PNG
            validity_save_path = root_save_path + '/validity/' + str(counter) + '.png'
            validity_mask = Image.fromarray(np.uint8(validity_mask*255))
            validity_mask.save(validity_save_path, "PNG")

            # Height map plot for data auditing purposes
            height_plot_save_path = root_save_path + '/height_plot/' + str(counter) + '.png'
            plt.imsave(height_plot_save_path, height_map_fill_only, cmap='inferno_r')

            counter += 1

        print('----- Processing complete -----') 
                

if __name__ == '__main__':
    main()
#%%