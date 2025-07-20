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
    # "CURVELANES",
    # "ROADWORK",
    "TUSIMPLE"
]
VALID_DATASET_LIST = list(get_args(VALID_DATASET_LITERALS))


class LoadDataBEVEgoPath():
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
        
        # Load JSON labels, get homotrans matrix as well
        with open(self.label_filepath, "r") as f:
            json_data = json.load(f)
            self.homotrans_mat = json_data.pop("standard_homomatrix")
            self.labels = json_data

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
            frame_id = self.train_ids[index]
            bev_egopath = self.train_labels[index]["bev_egopath"]
            reproj_egopath = self.train_labels[index]["reproj_egopath"]
            bev_egoleft = self.train_labels[index]["bev_egoleft"]
            reproj_egoleft = self.train_labels[index]["reproj_egoleft"]
            bev_egoright = self.train_labels[index]["bev_egoright"]
            reproj_egoright = self.train_labels[index]["reproj_egoright"]
        else:
            img = Image.open(str(self.val_images[index])).convert("RGB")
            frame_id = self.val_ids[index]
            bev_egopath = self.val_labels[index]["bev_egopath"]
            reproj_egopath = self.val_labels[index]["reproj_egopath"]
            bev_egoleft = self.val_labels[index]["bev_egoleft"]
            reproj_egoleft = self.val_labels[index]["reproj_egoleft"]
            bev_egoright = self.val_labels[index]["bev_egoright"]
            reproj_egoright = self.val_labels[index]["reproj_egoright"]

        W, H = img.size

        # Convert image to OpenCV/Numpy format for augmentations
        img = np.array(img)

        # Split labels
        xs_bev_egopath = [lab[0] / W for lab in bev_egopath]
        xs_reproj_egopath = [lab[0] / W for lab in reproj_egopath]

        xs_bev_egoleft = [lab[0] / W for lab in bev_egoleft]
        xs_reproj_egoleft = [lab[0] / W for lab in reproj_egoleft]

        xs_bev_egoright = [lab[0] / W for lab in bev_egoright]
        xs_reproj_egoright = [lab[0] / W for lab in reproj_egoright]

        ys_bev = [lab[1] / H for lab in bev_egopath]
        ys_reproj = [lab[1] / H for lab in reproj_egopath]

        flags_egopath = [lab[2] for lab in bev_egopath]
        valids_egopath = [lab[3] for lab in bev_egopath]

        flags_egoleft = [lab[2] for lab in bev_egoleft]
        valids_egoleft = [lab[3] for lab in bev_egoleft]
        
        flags_egoright = [lab[2] for lab in bev_egoright]
        valids_egoright = [lab[3] for lab in bev_egoright]
        
        return [
            frame_id, img,
            self.homotrans_mat,
            xs_bev_egopath, xs_reproj_egopath,
            xs_bev_egoleft, xs_reproj_egoleft,
            xs_bev_egoright, xs_reproj_egoright,
            ys_bev, ys_reproj,
            flags_egopath, valids_egopath,
            flags_egoleft, valids_egoleft,
            flags_egoright, valids_egoright,
        ]
    
# if __name__ == "__main__":

#     dataloader = LoadDataBEVEgoPath(
#         labels_filepath = "/home/tranhuunhathuy/Documents/Autoware/POV_train/pov_datasets/TUSIMPLE/drivable_path_bev.json",
#         images_filepath = "/home/tranhuunhathuy/Documents/Autoware/POV_train/pov_datasets/TUSIMPLE/image_bev",
#         dataset = "TUSIMPLE"
#     )

#     print(dataloader.getItem(0, True))