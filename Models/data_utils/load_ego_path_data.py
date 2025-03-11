#! /usr/bin/env python3

import json
import pathlib
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



class LoadDataEgoPath():
    def __init__(
            self, 
            labels_filepath: str,
            images_filepath: str,
            dataset: VALID_DATASET_LITERALS,
            val_set_fraction: float = 0.1,
    ):
        
        # ================= Parsing param ================= #

        self.label_filepath = labels_filepath
        self.image_dirpath = images_filepath
        self.dataset_name = dataset
        self.val_set_fraction = val_set_fraction

        # ================= Preliminary checks ================= #

        if not (0 <= self.val_set_fraction <= 1):
            raise ValueError(f"val_set_fraction must be within [0, 1]. Currently set to {self.val_set_fraction}")

        if not (self.dataset_name in VALID_DATASET_LIST):
            raise ValueError("Unknown dataset! Contact our team so we can work on this.")
        
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
            for set_idx, label_content in enumerate(self.labels):
                if (set_idx % 10 < val_set_fraction * 10):
                    # Slap it to Val
                    self.val_images.append(str(self.images[set_idx]))
                    self.val_labels.append(self.labels[label_content])
                    self.N_vals += 1 
                else:
                    # Slap it to Train
                    self.train_images.append(str(self.images[set_idx]))
                    self.train_labels.append(self.labels[label_content])
                    self.N_trains += 1

        print(f"Dataset {self.dataset_name} loaded with {self.N_trains} trains and {self.N_vals} vals.")
        print(f"Val/Total = {(self.N_vals / (self.N_trains + self.N_vals)):.4f}")

    def getItemCount(self):
        return self.N_trains, self.N_vals
    

if __name__ == "__main__":
    # Better write some unit tests to check this module
    # Testing cases here
    TuSimpleDataset = LoadDataEgoPath(
        labels_filepath = "/home/tranhuunhathuy/Documents/Autoware/pov_datasets/processed_CULane/drivable_path.json",
        images_filepath = "/home/tranhuunhathuy/Documents/Autoware/pov_datasets/processed_CULane/image",
        dataset = "CULANE",
        val_set_fraction = 0.1
    )