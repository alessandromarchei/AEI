#! /usr/bin/env python3
import pathlib
import numpy as np
from typing import Literal
from PIL import Image
from skimage.transform import resize
from .check_data import CheckData

class LoadDataSceneSeg():
    def __init__(self, labels_filepath, images_filepath, \
        dataset: Literal['URBANSYN', 'MUAD']):

        self.dataset = dataset

        if(self.dataset != 'URBANSYN' and self.dataset != 'MUAD'):
            raise ValueError('Dataset type is not correctly specified')
        
        self.labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.png")])
        self.images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

        self.num_images = len(self.images)
        self.num_labels = len(self.labels)

        checkData = CheckData(self.num_images, self.num_labels)
        
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []
        
        self.num_train_samples = 0
        self.num_val_samples = 0

        if (checkData.getCheck()):
            for count in range (0, self.num_images):
        
                if((count+1) % 10 == 0):
                    self.val_images.append(str(self.images[count]))
                    self.val_labels.append(str(self.labels[count]))
                    self.num_val_samples += 1 
                else:
                    self.train_images.append(str(self.images[count]))
                    self.train_labels.append(str(self.labels[count]))
                    self.num_train_samples += 1

    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples
    
    def getGroundTruth(self, input_label):
        ground_truth = np.load(input_label)
        return ground_truth
    
    def resizeData(self, input_image, input_label):
        image_resized = np.array(resize(input_image, (320, 640), anti_aliasing=True))
        label_resized = resize(input_label, (320, 640), anti_aliasing=True)
        return image_resized, label_resized

    def getItemTrain(self, index):
        self.train_image = Image.open(str(self.train_images[index]))
        self.train_ground_truth = self.getGroundTruth(str(self.train_labels[index]))
        self.train_image, self.train_ground_truth = \
            self.resizeData(self, self.train_image, self.train_ground_truth)
        
        return self.train_image, self.train_ground_truth

    def getItemTrainPath(self, index):
        return str(self.train_images[index]), str(self.train_labels[index])
    
    def getItemVal(self, index):
        self.val_image = Image.open(str(self.val_images[index]))
        self.val_ground_truth = self.getGroundTruth(str(self.val_labels[index]))
        self.val_image, self.val_ground_truth = \
            self.resizeData(self, self.val_image, self.val_ground_truth)
        
        return self.val_image, self.val_ground_truth
    
    def getItemValPath(self, index):
        return str(self.val_images[index]), str(self.val_labels[index])
