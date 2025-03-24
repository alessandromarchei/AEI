#! /usr/bin/env python3

import json
import os
import shutil
import pathlib
import numpy as np
from PIL import Image, ImageDraw
from typing import Literal, get_args
from check_data import CheckData

VALID_DATASET_LITERALS = Literal[
    "BDD100K", 
    "COMMA2K19", 
    "CULANE", 
    "CURVELANES", 
    "ROADWORK", 
    "TUSIMPLE"
]
VALID_DATASET_LIST = list(get_args(VALID_DATASET_LITERALS))
COMMA2K19_SIZE = {
    "w" : 1048,
    "h" : 524
}


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
        if "data" in self.labels:                   # Some has "data" parent key
            self.labels = self.labels["data"]       # Make sure it gets the "data" part
        # Some even stores ego path data like this:
        # data : [
        #   {
        #       "00001" : ...
        #   }
        # ]
        # So convert it back to dict to be compatible with the rest
        if type(self.labels) is list:
            self.labels = {
                frame_code : content
                for smaller_dict in self.labels
                for frame_code, content in smaller_dict.items()
            }

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
                frame_id_from_img_path = str(self.images[set_idx]).split("/")[-1].replace(".png", "")
                if (frame_id == frame_id_from_img_path):

                    if (self.dataset_name == "COMMA2K19"):
                        self.labels[frame_id]["drivable_path"] = [
                            (
                                point[0] / COMMA2K19_SIZE["w"], 
                                point[1] / COMMA2K19_SIZE["h"]
                            ) for point in self.labels[frame_id]["drivable_path"]
                        ]

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
        print(f"Val/Total = {(self.N_vals / (self.N_trains + self.N_vals))}")

    # Get sizes of Train/Val sets
    def getItemCount(self):
        return self.N_trains, self.N_vals

    # ================= Get item at index ith, returning img and EgoPath ================= #
    
    # For train
    def getItem(self, index, is_train: bool):
        if (is_train):
            img = Image.open(str(self.train_images[index])).convert("RGB")
            label = self.train_labels[index]["drivable_path"]
        else:
            img = Image.open(str(self.val_images[index])).convert("RGB")
            label = self.val_labels[index]["drivable_path"]

        # Filter out those y extremely close to 1.0
        # But under some certain conditions, add back last point of heap
        morethan1_heap = []
        while (label[0][1] >= 0.99):
            morethan1_heap.append(label[0])
            label.pop(0)
        if (
            (len(morethan1_heap) >= 1) and
                (
                    (len(label) <= 1) or 
                    (label[0][1] < 1.0)
                )
        ):
            label.insert(0, morethan1_heap[-1])
        # Convert all of those points into sublists
        label = [[x, y] for [x, y] in label]

        # Reduce the anchor's y coord a bit if it exceeds 1.0
        # (cuz y >= 1.0 will jeopardize the Albumentation)
        if (label[0][1] >= 1.0):
            label[0][1] = 0.9999
        
        return np.array(img), label