#! /usr/bin/env python3
import pathlib
import numpy as np
from typing import Literal
from PIL import Image
from .check_data import CheckData

class LoadDataSceneSeg():
    def __init__(self, labels_filepath, images_filepath):

        # Sort data and get list of input images and ground truth labels
        self.labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.png")])
        self.images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

        # Number of input images and ground truth labels
        self.num_images = len(self.images)
        self.num_labels = len(self.labels)

        # Performing sanity checks to ensure samples are correct in number
        checkData = CheckData(self.num_images, self.num_labels)

        # Lists to store train/val data
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []
        
        # Number of train/val samples
        self.num_train_samples = 0
        self.num_val_samples = 0

        # If all checks have passed, get samples and assign to train/val splits
        if (checkData.getCheck()):
            # Assigning ground truth data to train/val split
            for count in range (0, self.num_images):
        
                if((count+1) % 10 == 0):
                    self.val_images.append(str(self.images[count]))
                    self.val_labels.append(str(self.labels[count]))
                    self.num_val_samples += 1 
                else:
                    self.train_images.append(str(self.images[count]))
                    self.train_labels.append(str(self.labels[count]))
                    self.num_train_samples += 1