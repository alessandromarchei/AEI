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
            random_seed = 0
    ):
        
        # ================= Parsing param ================= #

        self.label_filepath = labels_filepath
        self.image_dirpath = images_filepath
        self.dataset_name = dataset
        self.val_set_fraction = val_set_fraction
        self.random_seed = random_seed

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