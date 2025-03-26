#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append('../..')
from inference.scene_3d_infer import Scene3DNetworkInfer

def main(): 

    # Saved model checkpoint path
    model_checkpoint_path = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/Scene3D/24_03_2025/model/iter_1103999_epoch_2_step_175781.pth'
    model = Scene3DNetworkInfer(checkpoint_path=model_checkpoint_path)
  
    # Reading input image
    input_image_filepath = '/home/zain/Autoware/Privately_Owned_Vehicles/Models/exports/test_image_5.jpg'
    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((640, 320))

    # Run inference and create visualization
    prediction = model.inference(image_pil)
    prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))
    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(prediction)


if __name__ == '__main__':
    main()
# %%