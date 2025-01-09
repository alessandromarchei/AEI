#%%
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from dgp.datasets import SynchronizedSceneDataset
import sys
sys.path.append('../../../../')
from SuperDepth.create_depth.common.lidar_depth_fill import LidarDepthFill
from SuperDepth.create_depth.common.height_map import HeightMap

def cropData(image, depth_map_fill_only, height_map_fill_only, validity_mask):

    # Cropping out those parts of data for which depth is unavailable
    image = image.crop((268, 200, 1668, 900))
    depth_map_fill_only = depth_map_fill_only[200:900, 268:1668]
    height_map_fill_only = height_map_fill_only[200:900, 268:1668]
    validity_mask = validity_mask[200:900, 268:1668]

    return image, depth_map_fill_only, height_map_fill_only, validity_mask

def processSample(sample, max_height, min_height, camera_height):
    
   # Access camera related data
    camera = sample[0][0]
    focal_length = camera['intrinsics'][1][1]
    cy = camera['intrinsics'][1][2]

    # PIL.Image
    image = camera['rgb'] 

    # (H,W) numpy.ndarray, generated from 'lidar' 
    sparse_depth_map = camera['depth']

    # Depth Map
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
    image, depth_map_fill_only, height_map_fill_only, validity_mask = \
      cropData(image, depth_map_fill_only, height_map_fill_only, validity_mask)
    
    return image, depth_map_fill_only, height_map_fill_only, validity_mask 

def saveData(root_save_path, counter, image, depth_map_fill_only, 
            height_map_fill_only, validity_mask):
  
  # Save files
  # RGB Image as PNG
  image_save_path = root_save_path + '/image/' + str(counter) + '.png'
  image.save(image_save_path, "PNG")

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

def main():

  # Path to dataset main json file
  ddad_json_path = '/mnt/media/ddad_train_val/ddad.json'

  # Save path
  root_save_path = '/mnt/media/SuperDepth/DDAD'

  # Load synchronized pairs of camera and lidar frames.
  dataset_train = \
  SynchronizedSceneDataset(ddad_json_path,
      datum_names=('lidar', 'CAMERA_01'),
      generate_depth_from_datum='lidar',
      split='train'
      )
  
  dataset_val = \
  SynchronizedSceneDataset(ddad_json_path,
      datum_names=('lidar', 'CAMERA_01'),
      generate_depth_from_datum='lidar',
      split='val'
      )

  # Iterate through the dataset.
  num_train_samples = len(dataset_train)
  num_val_samples = len(dataset_val)
  total_data = num_train_samples + num_val_samples

  print('Training samples', num_train_samples)
  print('Validation samples', num_val_samples)
  print('Total samples', total_data)

  # Data counter
  counter = 0

  # Camera height above road surface
  camera_height = 1.41

  # Height map limits
  max_height = 7
  min_height = -2
  
  # Training dataset
  for index in range(0, num_train_samples, 2):

    # Get data sample and process it
    sample = dataset_train[index]
    image, depth_map_fill_only, height_map_fill_only, validity_mask = \
        processSample(sample, max_height, min_height, camera_height)
    
    # Save data
    saveData(root_save_path, counter, image, depth_map_fill_only, 
            height_map_fill_only, validity_mask)

    counter += 1

  # Validation dataset
  for index in range(0, num_val_samples, 2):

    # Get data sample and process it
    sample = dataset_val[index]
    image, depth_map_fill_only, height_map_fill_only, validity_mask = \
        processSample(sample, max_height, min_height, camera_height)
    
    # Save data
    saveData(root_save_path, counter, image, depth_map_fill_only, 
        height_map_fill_only, validity_mask)

    counter += 1

if __name__ == '__main__':
  main()

#%%