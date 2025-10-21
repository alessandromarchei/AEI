#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser

from Models.inference.scene_seg_infer import SceneSegNetworkInfer
from utils.masks import add_mask_segmentation

def main(): 

    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath", help="path to input image which will be processed by SceneSeg")
    args = parser.parse_args() 

    # Saved model checkpoint path
    print('Loading SceneSeg Model at path:', args.model_checkpoint_path)
    model_checkpoint_path = args.model_checkpoint_path
    model = SceneSegNetworkInfer(checkpoint_path=model_checkpoint_path)
    print('SceneSeg Model Loaded')
  
    # Transparency factor
    alpha = 0.5

    # Reading input image
    print('Reading Image')
    input_image_filepath = args.input_image_filepath
    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((640, 320))

    # Run inference and create visualization
    print('Running Inference and Creating Visualization')
    prediction = model.inference(image_pil)

    #add mask
    image_vis_obj = add_mask_segmentation(frame, prediction, alpha)

    #save a png file
    output_filepath = 'scene_segmentation_output.png'
    cv2.imwrite(output_filepath, image_vis_obj)
    # cv2.imshow('Prediction Objects', image_vis_obj)
    # cv2.waitKey(0)

if __name__ == '__main__':
    main()
# %%