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