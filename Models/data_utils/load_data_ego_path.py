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
COMMA2K19_DIMS = {
    "WIDTH" : 1048,
    "HEIGHT" : 524
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
                                point[0] / COMMA2K19_DIMS["WIDTH"], 
                                point[1] / COMMA2K19_DIMS["HEIGHT"]
                            ) for point in self.labels[frame_id]["drivable_path"]
                        ]

                    if (set_idx % 10 == 0):     # Hard-coded 10% as val set
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
    def getItemTrain(self, index):
        img_train = Image.open(str(self.train_images[index])).convert("RGB")
        label_train = self.train_labels[index]["drivable_path"]
        # Address the issue currently occurs in CurveLanes
        label_train = [[x, y] for [x, y] in label_train if y < 1.0]
        
        return np.array(img_train), label_train
    
    # For val
    def getItemVal(self, index):
        img_val = Image.open(str(self.val_images[index])).convert("RGB")
        label_val = self.val_labels[index]["drivable_path"]
        # Address the issue currently occurs in CurveLanes
        label_val = [[x, y] for [x, y] in label_val if y < 1.0]
        
        return np.array(img_val), label_val
    
    def sampleItemsAudit(
            self, 
            set_type: Literal["train", "val"], 
            vis_output_dir: str,
            n_samples: int = 100
    ):
        print(f"Sampling first {n_samples} images from {set_type} set for audit...")

        if os.path.exists(vis_output_dir):
            print(f"Output path exists. Deleting.")
            shutil.rmtree(vis_output_dir)
        os.makedirs(vis_output_dir)

        for i in range(n_samples):

            # Fetch numpy image and ego path
            if set_type == "train":
                np_img, ego_path = self.getItemTrain(i)
            elif set_type == "val":
                np_img, ego_path = self.getItemVal(i)
            else:
                raise ValueError(f"sampleItemAudit() does not recognize set type {set_type}")
            
            # Convert back to image
            img = Image.fromarray(np_img)

            # Draw specs
            draw = ImageDraw.Draw(img)
            lane_color = (255, 255, 0)
            lane_w = 5
            img_width, img_height = img.size
            frame_name = str(i).zfill(5) + ".png"

            # Renormalize
            ego_path = [
                (float(point[0] * img_width), float(point[1] * img_height)) 
                for point in ego_path
            ]

            # Now draw
            draw.line(ego_path, fill = lane_color, width = lane_w)

            # Then, save
            img.save(os.path.join(vis_output_dir, frame_name))

        print(f"Sampling all done, saved at {vis_output_dir}")