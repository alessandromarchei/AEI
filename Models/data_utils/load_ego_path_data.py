#! /usr/bin/env python3

import argparse
import json
import os
import random
import shutil
import pathlib
from PIL import Image, ImageDraw
import warnings
from datetime import datetime
from typing import Literal, get_args

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
            raise ValueError(f"val_set_fraction must be within [0, 1]. Current set to {self.val_set_fraction}")

        if not (self.dataset_name in VALID_DATASET_LIST):
            raise ValueError("Unknown dataset! Contact our team so we can work on this.")
        
        with open(self.label_filepath, "r") as f:
            self.label_list = json.load(f)
        self.image_list = sorted([
            f for f in pathlib.Path(self.image_dirpath).glob("*.png")
        ])

        self.N_labels = len(self.label_list)
        self.N_images = len(self.image_list)

        if (self.N_labels != self.N_images):
            raise ValueError(f"Number of images ({self.N_images}) does not match number of labels ({self.N_labels})")
        
        if (self.N_labels == 0):
            raise ValueError(f"No labels found in specified path {self.label_filepath}")
        
        if (self.N_images == 0):
            raise ValueError(f"No images found in specified path {self.image_dirpath}")
        
        # ================= Initiate data loading ================= #

        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []

        self.num_train_samples = 0
        self.num_val_samples = 0

        if (self.random_seed):
            random.seed(self.random_seed)

        for img_idx in range (0, self.N_images):
            current_gacha = random.random()
            if (current_gacha <= val_set_fraction):
                self.val_images.append(str(self.image_list[img_idx]))
                self.val_labels.append(self.label_list[str(str(img_idx).zfill(6))])
                self.num_val_samples += 1 
            else:
                self.train_images.append(str(self.image_list[img_idx]))
                self.train_labels.append(self.label_list[str(str(img_idx).zfill(6))])
                self.num_train_samples += 1

        print(f"Dataset {self.dataset_name} loaded with {self.num_train_samples} trains and {self.num_val_samples} vals.")
        print(f"Val/Total = {self.num_val_samples / self.num_train_samples + self.num_val_samples}")

    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples