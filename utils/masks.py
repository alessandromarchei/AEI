import cv2
import sys
import os
import numpy as np
from PIL import Image

def mask_segmentation(prediction):
    """
    Generate RGB visualization from prediction mask.
    Background = orange, class 1 = purple, class 2 = green.
    """
    shape = prediction.shape
    vis_predict_object = np.zeros((shape[0], shape[1], 3), dtype="uint8")

    # Default background → orange
    vis_predict_object[:, :, 0] = 255
    vis_predict_object[:, :, 1] = 93
    vis_predict_object[:, :, 2] = 61

    # Class 1 (object) → purple
    fg = np.where(prediction == 1)
    vis_predict_object[fg[0], fg[1], :] = (145, 28, 255)

    # Class 2 (road/drivable surface) → green
    road = np.where(prediction == 2)
    vis_predict_object[road[0], road[1], :] = (0, 255, 0)

    return vis_predict_object


def add_mask_segmentation(input_frame, prediction, alpha):
    """
    Overlay segmentation mask on input frame with given alpha transparency.
    """
    mask = mask_segmentation(prediction)
    mask = cv2.resize(mask, (input_frame.shape[1], input_frame.shape[0]))
    output_frame = cv2.addWeighted(mask, alpha, input_frame, 1 - alpha, 0)
    return output_frame

