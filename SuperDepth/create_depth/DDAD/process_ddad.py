#%%
import pathlib
import numpy as np
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import json
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData
from SuperDepth.create_depth.common.lidar_depth_fill import LidarDepthFill
from SuperDepth.create_depth.common.height_map import HeightMap


def parseCalib(calib_files):
    
    calib_logs = []
    world_to_camera_transforms = []
    camera_intrinsic_matrices = []
    focal_lengths = []
    cy_vals = []


    for i in range (0, len(calib_files)):
        
        # Get filepath to calibration file
        calib_filepath = str(calib_files[i])
        
        # Storing log associated with calibration file
        calib_log = calib_filepath[26:-58]
        calib_logs.append(calib_log)

        # Reading calibration file JSON
        with open(calib_filepath, 'r') as file:
            data = json.load(file)
            camera_rotation = data['extrinsics'][1]['rotation']
            camera_translation = data['extrinsics'][1]['translation']
            camera_intrinsic = data['intrinsics'][1]

      
            # Get the world to camera transformation matrix
            ###############################################
            # Rotation
            qx = camera_rotation['qx']
            qy = camera_rotation['qy']
            qz = camera_rotation['qz']
            qw = camera_rotation['qw']
            quat = Quaternion(qw, qx, qy, qz)
            world_to_camera_rotation_matrix = quat.rotation_matrix

            # Translation
            tvec=np.float32([0., 0., 0.])
            tvec[0] = camera_translation['x']
            tvec[1] = camera_translation['y']
            tvec[2] = camera_translation['z']
           
            # Full transform
            world_to_camera_projection_matrix = np.eye(4)
            world_to_camera_projection_matrix[:3,:3] = world_to_camera_rotation_matrix
            world_to_camera_projection_matrix[:3, 3] = tvec
            
            # Get camera to image projection matrix
            #######################################
            camera_to_image_projection_matrix = np.eye(3)
            camera_to_image_projection_matrix[0, 0] = camera_intrinsic['fx']
            camera_to_image_projection_matrix[1, 1] = camera_intrinsic['fy']
            camera_to_image_projection_matrix[0, 2] = camera_intrinsic['cx']
            camera_to_image_projection_matrix[1, 2] = camera_intrinsic['cy']
            camera_to_image_projection_matrix[0, 1] = camera_intrinsic['skew']

            # Append to list
            world_to_camera_transforms.append(world_to_camera_projection_matrix)
            camera_intrinsic_matrices.append(camera_to_image_projection_matrix)
            focal_lengths.append(camera_intrinsic['fy'])
            cy_vals.append(camera_intrinsic['cy'])

    return calib_logs, world_to_camera_transforms, camera_intrinsic_matrices, focal_lengths, cy_vals

def projectPoincloudToImage(image, pointcloud, world_to_cam_transform, camera_intrinsic, img_height, img_width):
    
    draw = ImageDraw.Draw(image)

    for i in range(0, len(pointcloud)):

      
        world_point = np.float32([0., 0., 0., 1.])
        world_point[0] = pointcloud[i][0]
        world_point[1] = pointcloud[i][1]
        world_point[2] = pointcloud[i][2]
        
        camera_point = world_to_cam_transform.dot(world_point)
        image_point = camera_intrinsic.dot(camera_point[0:3])
        
        image_projection_x = image_point[0]/image_point[2] 
        image_projection_y = image_point[1]/image_point[2]
    
        x_val = image_projection_x
        y_val = image_projection_y
        draw.ellipse((y_val-5, x_val-5, y_val+5, x_val+5), fill=(255, 125, 0))
               

def main():
    
    # Filepaths for data loading and saving
    root_data_path = '/mnt/media/ddad_train_val/'
    root_save_path = '/mnt/media/SuperDepth/DDAD/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(root_data_path).glob("*/point_cloud/LIDAR/*.npz")])
    images = sorted([f for f in pathlib.Path(root_data_path).glob("*/rgb/CAMERA_01/*.png")])
    calib_files = sorted([f for f in pathlib.Path(root_data_path).glob("*/calibration/*.json")])

    # If all data checks have been passed
    num_depth_maps = len(depth_maps)
    num_images = len(images)
   
    checkData = CheckData(num_images, num_depth_maps)
    check_passed = checkData.getCheck()

    if(check_passed):
        print('Beginning processing of data')

        calib_logs, world_to_camera_transforms, camera_intrinsic_matrices, focal_lengths, cy_vals = \
            parseCalib(calib_files)
        

        # Height map limits
        max_height = 7
        min_height = -5

        # Camera height above ground
        camera_height = 0

        # Looping through data 
        for index in range(0, 1):
            
            # Image
            image = Image.open(str(images[index]))
            img_width, img_height = image.size

            # Pointcloud
            pointcloud = np.load(str(depth_maps[index]))['data']
            print(pointcloud)

            print(str(depth_maps[index]), str(images[index]))

            # Data log and its index
            data_log = str(images[index])[26:32]
            log_index = calib_logs.index(data_log)
            
            # Projection of pointlcoud point to image coordinates       
            projectPoincloudToImage(image, pointcloud, world_to_camera_transforms[log_index], camera_intrinsic_matrices[log_index], img_height, img_width)
            
            plt.figure()
            plt.imshow(image)

if __name__ == '__main__':
    main()
#%%