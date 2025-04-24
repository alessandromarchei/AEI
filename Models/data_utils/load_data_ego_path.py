#! /usr/bin/env python3

import os
import json
import pathlib
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
)))
from PIL import Image
from typing import Literal, get_args
from Models.data_utils.check_data import CheckData

VALID_DATASET_LITERALS = Literal[
    "BDD100K", 
    "COMMA2K19", 
    "CULANE", 
    "CURVELANES", 
    "ROADWORK", 
    "TUSIMPLE"
]
VALID_DATASET_LIST = list(get_args(VALID_DATASET_LITERALS))


class LoadDataEgoPath():
    def __init__(
            self, 
            labels_filepath: str,
            images_filepath: str,
            dataset: VALID_DATASET_LITERALS,
    ):
        
        # ================= Parsing param ================= #

        self.label_filepath = labels_filepath
        self.image_dirpath = images_filepath
        self.dataset_name = dataset

        # ================= Preliminary checks ================= #

        if not (self.dataset_name in VALID_DATASET_LIST):
            raise ValueError("Unknown dataset! Contact our team so we can work on this.")
        
        # Load JSON labels, address the diffs of format across datasets
        with open(self.label_filepath, "r") as f:
            self.labels = json.load(f)

        self.images = sorted([
            f for f in pathlib.Path(self.image_dirpath).glob("*.png")
        ])

        self.N_labels = len(self.labels)
        self.N_images = len(self.images)

        # Sanity check func by Mr. Zain
        checkData = CheckData(
            self.N_images,
            self.N_labels
        )
        
        # ================= Initiate data loading ================= #

        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []

        self.N_trains = 0
        self.N_vals = 0

        if (checkData.getCheck()):
            for set_idx, frame_id in enumerate(self.labels):

                # Check if there might be frame ID mismatch - happened to CULane before, just to make sure
                frame_id_from_img_path = str(self.images[set_idx]).split("/")[-1].replace(".png", "")
                if (frame_id == frame_id_from_img_path):

                    if (set_idx % 10 == 0):
                        # Slap it to Val
                        self.val_images.append(str(self.images[set_idx]))
                        self.val_labels.append(self.labels[frame_id])
                        self.N_vals += 1 
                    else:
                        # Slap it to Train
                        self.train_images.append(str(self.images[set_idx]))
                        self.train_labels.append(self.labels[frame_id])
                        self.N_trains += 1
                else:
                    raise ValueError(f"Mismatch data detected in {self.dataset_name}!")

        print(f"Dataset {self.dataset_name} loaded with {self.N_trains} trains and {self.N_vals} vals.")

    # Get sizes of Train/Val sets
    def getItemCount(self):
        return self.N_trains, self.N_vals
    
    # Point/lane auto audit
    def dataAudit(self, label: list):
        # Convert all points into sublists in case they are tuples - yeah it DOES happens
        if (type(label[0]) == tuple):
            label = [[x, y] for [x, y] in label]

        # Trimming points whose y > 1.0
        fixed_label = label.copy()
        fixed_label = [
            point for point in fixed_label
            if point[1] <= 1.0
        ]

        # Make sure points are bottom-up
        start_y, end_y = fixed_label[0][1], fixed_label[-1][1]
        if (end_y > start_y):       # Top-to-bottom annotation, must reverse
            fixed_label.reverse()

        # Slightly decrease y if y == 1.0
        for i, point in enumerate(fixed_label):
            if (point[1] == 1.0):
                fixed_label[i][1] = 0.9999

            if(point[0] < 0):
                fixed_label[i][0] = 0

            if(point[1] < 0):
                fixed_label[i][1] = 0

        # Comma2k19's jumpy point (but can be happening to others as well)
        for i in range(1, len(fixed_label)):
            if (fixed_label[i][1] > fixed_label[i - 1][1]):
                fixed_label = fixed_label[0 : i]
                break

        return fixed_label

    def fit_cubic_bezier(self, label):

        # Fit a Cubic Bezier curve to raw data points

        # Chord length parameterization
        distances = np.sqrt(np.sum(np.diff(label, axis=0)**2, axis=1))
        cumulative = np.insert(np.cumsum(distances), 0, 0)
        t = cumulative / cumulative[-1]

        # Bézier basis functions
        def bernstein_matrix(t):
            t = np.asarray(t)
            B = np.zeros((len(t), 4))
            B[:, 0] = (1 - t)**3
            B[:, 1] = 3 * (1 - t)**2 * t
            B[:, 2] = 3 * (1 - t) * t**2
            B[:, 3] = t**3
            return B

        B = bernstein_matrix(t)

        # Least squares fitting: B * P = points => P = (B^T B)^-1 B^T * points
        BTB = B.T @ B
        BTP = B.T @ label
        control_points = np.linalg.solve(BTB, BTP)

        # Get control points for cubic bezier curve
        p0 = control_points[0]
        p1 = control_points[1]
        p2 = control_points[2]
        p3 = control_points[3]
   
        return p0, p1, p2, p3
    
    def evaluate_bezier(self, p0, p1, p2, p3, t):

        # Throw an error if parameter t is out of boudns
        if(t < 0 or t > 1):
            raise ValueError('Please ensure t parameter is in the range [0,1]')
        
        # Evaluate cubic bezier curve for value of t in range [0,1]
        x = (1-t)*(1-t)*(1-t)*p0[0] + 3*(1-t)*(1-t)*t*p1[0] \
            + 3*(1-t)*t*t*p2[0] + t*t*t*p3[0]
        
        y = (1-t)*(1-t)*(1-t)*p0[1] + 3*(1-t)*(1-t)*t*p1[1] \
            + 3*(1-t)*t*t*p2[1] + t*t*t*p3[1]
        
        return x,y
    
    def sample_bezier_points(self, p0, p1, p2, p3):

        samples = []

        # Getting samples, and ignoring samples very close to bottom of image
        # due to inaccuracy caused by Cubic Bezier over-fitting
        for i in range(20, 110, 10):

            t_val = i/100
            point = []

            x_point, y_point = self.evaluate_bezier(p0, p1, p2, p3, t_val)
            point.append(x_point)
            point.append(y_point)

            samples.append(point)

        # Linear extrapolation of the samples to the bottom of the image

        # Samples at the very bottom of the image
        first_sample = samples[0]
        second_sample = samples[1]

        # Gradient
        start_gradient = (first_sample[0] - second_sample[0])/ \
            (first_sample[1] - second_sample[1] + 0.000001)

        # How far away is the current start sample from the bottom of the image
        first_sample_y_val_offset = 1 - first_sample[1]

        # Calculate coordinate of extrapolated point
        zero_sample_x = start_gradient*first_sample_y_val_offset + first_sample[0]
        zero_sample_y = 1.0

        # Insert extrapolated point to the bottom of the list
        new_start_point = []
        new_start_point.append(zero_sample_x)
        new_start_point.append(zero_sample_y)
        samples.insert(0, new_start_point)

        return samples
    
    # Get item at index ith, returning img and EgoPath
    def getItem(self, index, is_train: bool):
        if (is_train):
            img = Image.open(str(self.train_images[index])).convert("RGB")
            label = self.train_labels[index]["drivable_path"]
        else:
            img = Image.open(str(self.val_images[index])).convert("RGB")
            label = self.val_labels[index]["drivable_path"]

        # Point/line auto audit
        label = self.dataAudit(label)

        # Flag to store whether data is valid or not
        is_valid = True

        # Keypoints
        keypoints = 0

        # If there are enough points to fit a cubic bezier curve
        if(len(label) >= 4):
            
            # Fit a cubic bezier curve to raw data points
            p0, p1, p2, p3 = self.fit_cubic_bezier(label)

            # Calculate discrete data keypoints based on sampling
            # of cubic bezier curve
            keypoints = self.sample_bezier_points(p0, p1, p2, p3)

        else:
            # Data is not valid since we need at least 4 raw data
            # points to fit a cubic bezier curve
            is_valid = False

        return np.array(img), keypoints, is_valid
    
  