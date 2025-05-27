#! /usr/bin/env python3
import pathlib
import numpy as np
from typing import Literal
from PIL import Image

class LoadDataSceneSeg():
    def __init__(self, labels_filepath, images_filepath):

        self.labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.png")])
        self.images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

        self.num_images = len(self.images)
        self.num_labels = len(self.labels)