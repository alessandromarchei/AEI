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
        
        