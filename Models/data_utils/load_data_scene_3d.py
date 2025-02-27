#! /usr/bin/env python3
import pathlib
import numpy as np
from typing import Literal
from PIL import Image
from .check_data import CheckData

class LoadDataScene3D():
    def __init__(self, labels_filepath, images_filepath, \
        dataset: Literal['URBANSYN', 'MUAD', 'GTAV', 'KITTI', 'DDAD', 'ARGOVERSE', 'MUSES', 'DIVERSE'], validity_filepath=''):

        self.dataset = dataset

        if(self.dataset != 'URBANSYN' and self.dataset != 'MUAD' and self.dataset != 'KITTI'
           and self.dataset!= 'DDAD' and self.dataset!= 'ARGOVERSE' and self.dataset!= 'MUSES'
           and self.dataset!= 'GTAV' and self.dataset!= 'DIVERSE'):
            raise ValueError('Dataset type is not correctly specified')
        
        self.labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.npy")])
        self.num_labels = len(self.labels)

        self.images = sorted([f for f in pathlib.Path(images_filepath).glob("*")])
        self.num_images = len(self.images)

        checkData = CheckData(self.num_images, self.num_labels)

        self.validities = 0
        self.is_validity = False
        self.num_valid_samples = 0

        if(len(validity_filepath) > 0):
            self.validities = sorted([f for f in pathlib.Path(validity_filepath).glob("*.png")])
            self.num_valid_samples = len(self.validities)
            checkValidityData = CheckData(self.num_valid_samples, self.num_labels)

            if(checkValidityData.getCheck()):
                self.is_validity = True
            
   
        self.train_images = []
        self.train_labels = []
        self.train_validities = []
        self.val_images = []
        self.val_labels = []
        self.val_validities = []
        self.all_images = []
        self.all_labels = []
        self.all_validities = []
        self.ddad_train_cams = []
        self.ddad_val_cams = []

        self.num_train_samples = 0
        self.num_val_samples = 0
        self.num_all_samples = 0

        if (checkData.getCheck()):
            for count in range (0, self.num_images):

                self.all_images.append(str(self.images[count]))
                self.all_labels.append(str(self.labels[count]))

                if(self.is_validity):
                    self.all_validities.append(str(self.validities[count]))
                self.num_all_samples += 1

                if((count+1) % 10 == 0):
                    self.val_images.append(str(self.images[count]))
                    self.val_labels.append(str(self.labels[count]))

                    if(self.is_validity):
                        self.val_validities.append(str(self.validities[count]))
                                    
                    if(dataset == 'DDAD'):
                        if(count >= 16000):
                            self.ddad_val_cams.append('back_camera')
                        else:
                            self.ddad_val_cams.append('front_camera')

                    self.num_val_samples += 1 
                else:
                    self.train_images.append(str(self.images[count]))
                    self.train_labels.append(str(self.labels[count]))

                    if(self.is_validity):
                        self.train_validities.append(str(self.validities[count]))

                    if(dataset == 'DDAD'):
                        if(count >= 16000):
                            self.ddad_train_cams.append('back_camera')
                        else:
                            self.ddad_train_cams.append('front_camera')

                    self.num_train_samples += 1

    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples
    
    def getTotalCount(self):
        return self.num_all_samples
    
    def getGroundTruth(self, input_label):
        ground_truth = np.load(input_label)
        ground_truth = np.divide(1, ground_truth, out=np.zeros_like(ground_truth), where=ground_truth!=0)
        ground_truth = np.expand_dims(ground_truth, axis=-1)
        return ground_truth
    
    def getDDADCameras(self):
        return self.ddad_train_cams, self.ddad_val_cams
    
    def getValidity(self, gt, index, is_train):

        validity = 0

        if(self.is_validity):
            if(is_train):
                validity = Image.open(str(self.train_validities[index]))
            else:
                validity = Image.open(str(self.val_validities[index]))

            validity = np.array(validity)
            validity = np.expand_dims(validity, axis=-1)
        else:
            validity = np.full_like(gt, 255).astype('uint8') 

        return validity
    
    def getValidityAll(self, gt, index):
        
        validity = 0

        if(self.is_validity):
            validity = Image.open(str(self.all_validities[index]))
            validity = np.array(validity)
            validity = np.expand_dims(validity, axis=-1)
        else:
            validity = np.full_like(gt, 255).astype('uint8') 

        return validity
    
    def getItemAll(self, index):
        image = Image.open(str(self.all_images[index])).convert('RGB')
        ground_truth = self.getGroundTruth(str(self.all_labels[index]))
        validity = self.getValidityAll(ground_truth, index)

        return  np.array(image), ground_truth, validity

    def getItemTrain(self, index):
        train_image = Image.open(str(self.train_images[index])).convert('RGB')
        train_ground_truth = self.getGroundTruth(str(self.train_labels[index]))
        train_validity = self.getValidity(train_ground_truth, index, is_train=True)

        return  np.array(train_image), train_ground_truth, train_validity

    def getItemTrainPath(self, index):
        return str(self.train_images[index]), str(self.train_labels[index])
    
    def getValidityTrainPath(self, index):
        validity_path = ''

        if(self.is_validity):
            validity_path = str(self.train_validities[index])

        return validity_path
    
    def getItemVal(self, index):
        val_image = Image.open(str(self.val_images[index])).convert('RGB')
        val_ground_truth = self.getGroundTruth(str(self.val_labels[index]))
        val_validity = self.getValidity(val_ground_truth, index, is_train=False) 

        return  np.array(val_image), val_ground_truth, val_validity
    
    def getItemValPath(self, index):
        return str(self.val_images[index]), str(self.val_labels[index])
    
    def getValidityValPath(self, index):
        validity_path = ''

        if(self.is_validity):
            validity_path = str(self.val_validities[index])

        return validity_path

    
