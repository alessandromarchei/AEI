#! /usr/bin/env python3

import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image
import numpy as np
import sys

sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.ego_path_network import EgoPathNetwork
from data_utils.augmentations import Augmentations


class EgoPathTrainer():
    def __init__(
        self,  
        checkpoint_path = "", 
        pretrained_checkpoint_path = "", 
        is_pretrained = False
    ):
        
        # Image and gts
        self.image = None
        self.xs = []
        self.ys = []
        self.flags = []

        # Dims
        self.height = 640
        self.width = 320

        # Tensors
        self.image_tensor = None
        self.xs_tensor = []
        self.flags_tensor = []

        # Model and pred
        self.model = None
        self.prediction = None

        # Losses
        self.loss = 0
        self.data_loss = 0
        self.smoothing_loss = 0
        self.flag_loss = 0
        self.gradient_type = "NUMERICAL"

        # Checking devices (GPU vs CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        print(f"Using {self.device} for inference.")

        if (is_pretrained):

            # Instantiate model for validation or inference - load both pre-trained SceneSeg and SuperDepth weights
            if (len(checkpoint_path) > 0):

                # Loading model with full pre-trained weights
                sceneSegNetwork = SceneSegNetwork()
                self.model = EgoPathNetwork(sceneSegNetwork)

                # If the model is also pre-trained then load the pre-trained downstream weights
                self.model.load_state_dict(torch.load(
                    checkpoint_path, 
                    weights_only = True, 
                    map_location = self.device
                ))
                print("Loading pre-trained model weights of EgoPath from a saved checkpoint file")
            else:
                raise ValueError("Please ensure EgoPath network weights are provided for downstream elements")
            
        else:

            # Instantiate Model for training - load pre-trained SceneSeg weights only
            if (len(pretrained_checkpoint_path) > 0):
                
                # Loading SceneSeg pre-trained for upstream weights
                sceneSegNetwork = SceneSegNetwork()
                sceneSegNetwork.load_state_dict(torch.load(
                    pretrained_checkpoint_path, 
                    weights_only = True, 
                    map_location = self.device
                ))
                    
                # Loading model with pre-trained upstream weights
                self.model = EgoPathNetwork(sceneSegNetwork)
                print("Loading pre-trained backbone model weights only, EgoPath initialised with random weights")
            else:
                raise ValueError("Please ensure EgoPath network weights are provided for upstream elements")