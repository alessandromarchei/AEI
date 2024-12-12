#%%
import pathlib
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData
from SuperDepth.create_depth.common.lidar_depth_fill import LidarDepthFill
from SuperDepth.create_depth.common.height_map import HeightMap

def quaternion_rotation_matrix(q0, q1, q2, q3):
    
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = [[r00, r01, r02],
                  [r10, r11, r12],
                  [r20, r21, r22]]
                     
    return rot_matrix

def parseCalib(calib_files):
    
    calib_logs = []
    world_to_image_transforms = []
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

            # Get world to camera projection matrix
            q0 = camera_rotation['qx']
            q1 = camera_rotation['qy']
            q2 = camera_rotation['qz']
            q3 = camera_rotation['qw']

            # Convert from quaternion to rotation matrix
            camera_rotation_matrix = quaternion_rotation_matrix(q0, q1, q2, q3)
            
            # Include translation compontent to get final 3x4 matrix
            world_to_camera_projection_matrix = camera_rotation_matrix
            world_to_camera_projection_matrix[0].append(-1*camera_translation['x'])
            world_to_camera_projection_matrix[1].append(-1*camera_translation['y'])
            world_to_camera_projection_matrix[2].append(-1*camera_translation['z'])

            # Get camera to image projection matrix
            camera_to_image_projection_matrix = []
            camera_intrinsic_row = []

            # Contstruct matrix
            camera_intrinsic_row.append(camera_intrinsic['fx'])
            camera_intrinsic_row.append(camera_intrinsic['skew'])
            camera_intrinsic_row.append(camera_intrinsic['cx'])
            camera_to_image_projection_matrix.append(camera_intrinsic_row.copy())
            camera_intrinsic_row.clear()

            camera_intrinsic_row.append(0.0)
            camera_intrinsic_row.append(camera_intrinsic['fy'])
            camera_intrinsic_row.append(camera_intrinsic['cy'])
            camera_to_image_projection_matrix.append(camera_intrinsic_row.copy())
            camera_intrinsic_row.clear()

            camera_intrinsic_row.append(0.0)
            camera_intrinsic_row.append(0.0)
            camera_intrinsic_row.append(1.0)
            camera_to_image_projection_matrix.append(camera_intrinsic_row.copy())

            # Get world to image matrix
            world_to_cam_np = np.array(world_to_camera_projection_matrix)
            cam_to_image_np = np.array(camera_to_image_projection_matrix)
            world_to_image_transform = cam_to_image_np.dot(world_to_cam_np)
            #world_to_image_transform = np.matmul(cam_to_image_np, world_to_cam_np)
            

            # Append to list
            world_to_image_transforms.append(world_to_image_transform)
            focal_lengths.append(camera_intrinsic['fy'])
            cy_vals.append(camera_intrinsic['cy'])

    return calib_logs, world_to_image_transforms, focal_lengths, cy_vals

def projectPoincloudToImage(image, pointcloud, world_to_image_transform, img_height, img_width):
    
    draw = ImageDraw.Draw(image)

    for i in range(0, len(pointcloud)):
        pointcloud_point = []
        pointcloud_point.append(pointcloud[i][0])
        pointcloud_point.append(pointcloud[i][1])
        pointcloud_point.append(pointcloud[i][2])
        pointcloud_point.append(1.0)

        world_point = np.array(pointcloud_point)
        
        scaled_image_point = world_to_image_transform.dot(world_point)
        image_projection_x = scaled_image_point[0]/scaled_image_point[2] 
        image_projection_y = scaled_image_point[1]/scaled_image_point[2]

        if(image_projection_x >= 0 and image_projection_x < img_height):
            if(image_projection_y >= 0 and image_projection_y < img_width):
               
               if(world_point[2] > 0):
                    x_val = round(image_projection_x)
                    y_val = round(image_projection_y)
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

        calib_logs, world_to_image_transforms, focal_lengths, cy_vals = \
            parseCalib(calib_files)
        
        
        # Height map limits
        max_height = 7
        min_height = -5

        # Camera height above ground
        camera_height = 0

        # Looping through data 
        for index in range(20, 21):
            
            # Image
            image = Image.open(str(images[index]))
            img_width, img_height = image.size

            # Pointcloud
            pointcloud = np.load(str(depth_maps[index]))['data']

            print(str(depth_maps[index]), str(images[index]))

            # Data log and its index
            data_log = str(images[index])[26:32]
            log_index = calib_logs.index(data_log)
            
            # Projection of pointlcoud point to image coordinates       
            projectPoincloudToImage(image, pointcloud, world_to_image_transforms[log_index], img_height, img_width)
            
            plt.figure()
            plt.imshow(image)

if __name__ == '__main__':
    main()
#%%