#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2
import sys
import numpy as np
from PIL import Image
import cmapy
from argparse import ArgumentParser
sys.path.append('../..')
from inference.scene_3d_infer import Scene3DNetworkInfer


def main(): 

    # Saved model checkpoint path
    model_checkpoint_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/saves/Scene3D/24_03_2025/model/iter_1103999_epoch_2_step_175781.pth'
    model = Scene3DNetworkInfer(checkpoint_path=model_checkpoint_path)
    
    # Create a VideoCapture object and read from input file
    # If the input is taken from the camera, pass 0 instead of the video file name.
    video_filepath = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/saves/Crash_Compilation.mp4'
    cap = cv2.VideoCapture(video_filepath)

    # Output filepath
    output_filepath_obj = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/saves/Crash_Compilation_Depth' + '.avi'


    writer_obj = cv2.VideoWriter(output_filepath_obj,
    cv2.VideoWriter_fourcc(*"MJPG"), 25,(1280,720))

    # Check if video catpure opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # Visualize flag
    vis = True
 
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize((640, 320))
            
            # Run inference
            prediction = model.inference(image_pil)
            prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))
    
            # Create visualization
            prediction_image = 255.0*((prediction - np.min(prediction))/ (np.max(prediction) - np.min(prediction)))
            prediction_image = prediction_image.astype(np.uint8)
            vis_obj = cv2.applyColorMap(prediction_image, cmapy.cmap('viridis'))
            image_vis_obj = cv2.resize(vis_obj, (1280, 720))

            if(vis):
                cv2.imshow('Depth', image_vis_obj)
                cv2.waitKey(10)

            # Writing to video frame
            writer_obj.write(image_vis_obj)

    # When everything done, release the video capture and writer objects
    cap.release()
    writer_obj.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    # %%