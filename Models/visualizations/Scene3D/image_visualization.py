#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append('..')
from inference.scene_3d_infer import Scene3DNetworkInfer

def make_visualization(prediction):
    shape = prediction.shape
  
    row = shape[0]
    col = shape[1]
    vis_predict_object = np.zeros((row, col, 3), dtype = "uint8")

    free_space_colour = (0, 255, 91)
    foreground_objects_colour = (255, 93, 61)
    background_objects_colour = (40, 28, 255)

    # Extracting predicted classes and assigning to colourmap
    for x in range(2, row-2):
        for y in range(0, col):
            if(prediction[x,y] > 0.3):
                vis_predict_object[x,y] = foreground_objects_colour
            #if(prediction[x+2,y]/prediction[x-2,y] < 0.98):
            #    vis_predict_object[x,y] = free_space_colour
            #elif(prediction[x,y] >= 0.3 and prediction[x,y]< 0.5):
            #    vis_predict_object[x,y] = background_objects_colour
            #elif(prediction[x,y] >= 0.5):
            #    vis_predict_object[x,y] = foreground_objects_colour

    return vis_predict_object           

def main(): 

    # Saved model checkpoint path
    model_checkpoint_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/Scene3D/2025_01_27/model/iter_990791_epoch_24_step_19799.pth'
    model = Scene3DNetworkInfer(checkpoint_path=model_checkpoint_path)
  

    # Reading input image
    input_image_filepath = '/mnt/media/Scene3D/KITTI/image/380.png'
    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((640, 320))

    # Transparency factor
    alpha = 0.5

    # Run inference and create visualization
    prediction = model.inference(image_pil)
    prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))
    prediction_metric = prediction*7


    obstacle_mask = make_visualization(prediction_metric)  
    image_vis_obj = cv2.addWeighted(obstacle_mask, alpha, frame, 1 - alpha, 0)
    plt.figure()
    plt.imshow(prediction_metric)
    plt.figure()
    plt.imshow(image_vis_obj)


if __name__ == '__main__':
    main()
# %%