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

# Currently limiting to available datasets only. Will unlock eventually
VALID_DATASET_LITERALS = Literal[
    # "BDD100K",
    # "COMMA2K19",
    # "CULANE",
    "CURVELANES",
    # "ROADWORK",
    # "TUSIMPLE"
]
VALID_DATASET_LIST = list(get_args(VALID_DATASET_LITERALS))


class LoadDataAutoSteer():
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
        self.train_ids = []
        self.val_images = []
        self.val_labels = []
        self.val_ids = []

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
                        self.val_ids.append(frame_id)
                        self.N_vals += 1 
                    else:
                        # Slap it to Train
                        self.train_images.append(str(self.images[set_idx]))
                        self.train_labels.append(self.labels[frame_id])
                        self.train_ids.append(frame_id)
                        self.N_trains += 1
                else:
                    raise ValueError(f"Mismatch data detected in {self.dataset_name}!")

        print(f"Dataset {self.dataset_name} loaded with {self.N_trains} trains and {self.N_vals} vals.")

    # Get sizes of Train/Val sets
    def getItemCount(self):
        return self.N_trains, self.N_vals
       
    # Get item at index ith, returning img and EgoPath
    def getItem(self, index, is_train: bool):
        if (is_train):
            img = Image.open(str(self.train_images[index])).convert("RGB")
            drivable_path_bev = self.train_labels[index]["drivable_path_bev"]
            ego_left_lane_bev = self.train_labels[index]["ego_left_lane_bev"]
            ego_right_lane_bev = self.train_labels[index]["ego_right_lane_bev"]
            drivable_path = self.train_labels[index]["drivable_path"]
            ego_left_lane = self.train_labels[index]["ego_left_lane"]
            ego_right_lane = self.train_labels[index]["ego_right_lane"]
            frame_id = self.train_ids[index]
        else:
            img = Image.open(str(self.val_images[index])).convert("RGB")
            drivable_path_bev = self.train_labels[index]["drivable_path_bev"]
            ego_left_lane_bev = self.train_labels[index]["ego_left_lane_bev"]
            ego_right_lane_bev = self.train_labels[index]["ego_right_lane_bev"]
            drivable_path = self.train_labels[index]["drivable_path"]
            ego_left_lane = self.train_labels[index]["ego_left_lane"]
            ego_right_lane = self.train_labels[index]["ego_right_lane"]
            frame_id = self.val_ids[index]

        W, H = img.size

        # Convert image to OpenCV/Numpy format for augmentations
        img = np.array(img)
        
        return frame_id, img, drivable_path_bev, drivable_path, \
                ego_left_lane_bev, ego_left_lane, ego_right_lane_bev, \
                ego_right_lane