#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../..')
from inference.ego_space_infer import EgoSpaceNetworkInfer


def make_visualization(prediction, image):

    # Creating visualization object
    colour_mask = np.array(image)
    shape = prediction.shape
    row = shape[0]
    col = shape[1]

    # Black-and-white mask
    segMask = np.zeros((row, col), dtype='uint8')

    # Getting foreground object labels
    foreground_lables = np.where(prediction == 1.0)

    # Assigning foreground objects colour
    colour_mask[foreground_lables[0], foreground_lables[1], 0] = 28
    colour_mask[foreground_lables[0], foreground_lables[1], 1] = 148
    colour_mask[foreground_lables[0], foreground_lables[1], 2] = 255

    segMask[foreground_lables[0], foreground_lables[1]] = 255

    return segMask, colour_mask

def main(): 

    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath", help="path to input image which will be processed by EgoSpace")
    args = parser.parse_args() 

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    model = EgoSpaceNetworkInfer(checkpoint_path=model_checkpoint_path)
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
    label_mask, vis_mask = make_visualization(prediction)

    # Apply alpha transparency factor of 0.5
    label_mask_composite = np.uint8(label_mask*0.5)
    label_mask_composite = Image.fromarray(label_mask_composite)
    vis = Image.fromarray(vis_mask)
    visualization = Image.composite(image, vis, label_mask_composite)
    vis_obj = np.array(visualization)

    # Resize and display visualization
    vis_obj = cv2.resize(vis_obj, (frame.shape[1], frame.shape[0]))
    image_vis_obj = cv2.addWeighted(vis_obj, alpha, frame, 1 - alpha, 0)
    cv2.imshow('Prediction Objects', image_vis_obj)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
# %%