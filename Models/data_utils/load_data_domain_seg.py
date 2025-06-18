#! /usr/bin/env python3
import pathlib
import numpy as np
from PIL import Image
from .check_data import CheckData

class LoadDataDomainSeg():
    def __init__(self, labels_filepath, images_filepath):

        # Sort data and get list of input images and ground truth labels
        self.labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.png")])
        self.images = sorted([f for f in pathlib.Path(images_filepath).glob("*.jpg")])

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
        
                if((count+1) % 50 == 0):
                    self.val_images.append(str(self.images[count]))
                    self.val_labels.append(str(self.labels[count]))
                    self.num_val_samples += 1 
                else:
                    self.train_images.append(str(self.images[count]))
                    self.train_labels.append(str(self.labels[count]))
                    self.num_train_samples += 1

    # Getting number of train/val samples
    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples
    
    # Calculating class weights based on pixel frequency
    def getData(self, mask):
        background_pixels = np.where(mask == 0)
        foreground_pixels = np.where(mask != 0)

        # Image Size
        row, col = mask.shape
        num_pixels = row*col

        class_weights = []

        background_class_weight = num_pixels/(len(background_pixels[0]) + 5120)
        class_weights.append(background_class_weight)

        foreground_class_weight = num_pixels/(len(foreground_pixels[0]) + 5120)
        class_weights.append(foreground_class_weight)

        background_image = np.zeros((row, col, 1), dtype='uint8')
        foreground_image = np.zeros((row, col, 1), dtype='uint8')

        background_image[background_pixels[0], background_pixels[1], 0] = 255
        foreground_image[foreground_pixels[0], foreground_pixels[1], 0] = 255

        ground_truth = []
        ground_truth.append(foreground_image)
        ground_truth.append(background_image)

        return ground_truth, class_weights
    
    # Get training data in numpy format
    def getItemTrain(self, index):
        train_image = np.array(Image.open(str(self.train_images[index])).convert('RGB'))
        train_mask = np.array(Image.open(str(self.train_labels[index])))
        ground_truth, class_weights = self.getData(train_mask)
        return  train_image, ground_truth, class_weights
    
    # Get training data in numpy format
    def getItemVal(self, index):
        val_image = np.array(Image.open(str(self.val_images[index])).convert('RGB'))
        val_mask = np.array(Image.open(str(self.val_labels[index])))
        ground_truth, class_weights = self.getData(val_mask)
        return  val_image, ground_truth, class_weights
    