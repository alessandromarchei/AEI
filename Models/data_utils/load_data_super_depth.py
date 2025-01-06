#! /usr/bin/env python3
import pathlib
import numpy as np
from typing import Literal
from PIL import Image
from .check_data import CheckData

class LoadDataSuperDepth():
    def __init__(self, labels_filepath, images_filepath, \
        dataset: Literal['URBANSYN', 'MUAD', 'KITTI_TEST'], validity=''):

        self.dataset = dataset

        if(self.dataset != 'URBANSYN' and self.dataset != 'MUAD' and self.dataset != 'KITTI_TEST'):
            raise ValueError('Dataset type is not correctly specified')
        
        self.labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.npy")])
        self.images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])
        
        self.validity = 0
        self.is_validity = False

        if(len(validity > 0)):
            self.validity = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

            checkValidityData = CheckData(self.validity, self.num_labels)
            if(checkValidityData.getCheck()):
                self.is_validity = True

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
        ground_truth = np.expand_dims(ground_truth, axis=-1)
        return ground_truth

    def getItemTrain(self, index):
        self.train_image = Image.open(str(self.train_images[index]))
        self.train_ground_truth = self.getGroundTruth(str(self.train_labels[index]))
        return  np.array(self.train_image), self.train_ground_truth

    def getItemTrainPath(self, index):
        return str(self.train_images[index]), str(self.train_labels[index])
    
    def getItemVal(self, index):
        self.val_image = Image.open(str(self.val_images[index]))
        self.val_ground_truth = self.getGroundTruth(str(self.val_labels[index]))      
        return  np.array(self.val_image), self.val_ground_truth
    
    def getItemValPath(self, index):
        return str(self.val_images[index]), str(self.val_labels[index])
    
    def getItemValidity(self, index):

        if(self.is_validity):
            validity_image = Image.open(str(self.validity[index]))
        else:
            raise ValueError('Validity masks not found - check filepath')
        
        return np.array(validity_image)
    
