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

BEV_FIGSIZE = (4, 8)


class EgoPathTrainer():
    def __init__(
        self,  
        checkpoint_path = "", 
        pretrained_checkpoint_path = "", 
        is_pretrained = False
    ):
        
        # Image and gts
        self.image = None
        self.H = None
        self.W = None
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

        # Loss scale factors
        self.data_loss_scale_factor = 1.0
        self.smoothing_loss_scale_factor = 1.0
        self.flag_loss_scale_factor = 1.0

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
            
        # Model to device
        self.model = self.model.to(self.device)
        
        # TensorBoard
        self.writer = SummaryWriter()

        # Learning rate and optimizer
        self.learning_rate = 0.0001
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)

        # Loaders
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    
    # Zero gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Learning rate adjustment
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    # Assign input variables
    def set_data(self, image, xs, ys, flags):
        h, w, _ = image.shape
        self.image = image
        self.H = h
        self.W = w
        self.xs = xs
        self.ys = ys
        self.flags = flags

    # Image agumentations
    def apply_augmentations(self, is_train):
        # Augmenting data for train or val/test
        aug = Augmentations(
            is_train = is_train, 
            data_type = "KEYPOINTS"
        )
        aug.setImage(self.image)
        self.image = aug.applyTransformKeypoint(self.image)

    # Load data as Pytorch tensors
    def load_data(self):
        # Converting image to Pytorch tensor
        image_tensor = self.image_loader(self.image)
        image_tensor = image_tensor.unsqueeze(0)
        self.image_tensor = image_tensor.to(self.device)

        # Converting gt lists to Pytorch Tensor
        # Xs
        xs_tensor = torch.from_numpy(self.xs)
        xs_tensor = xs_tensor.unsqueeze(0)
        self.xs_tensor = xs_tensor.to(self.device)

        # Flags
        flags_tensor = torch.from_numpy(self.flags)
        flags_tensor = flags_tensor.unsqueeze(0)
        self.flags_tensor = flags_tensor.to(self.device)
    
    # Run Model
    def run_model(self):
        self.prediction = self.model(self.image_tensor)
        self.loss = self.calc_loss(
            self.prediction, 
            self.xs_tensor,
            self.flags_tensor
        )

    # Calculate loss
    def calc_loss(self, prediction, xs, flags):
        self.data_loss = self.calc_data_loss(prediction, xs, flags)
        self.smoothing_loss = self.calc_smoothing_loss(prediction, xs, flags)
        self.flag_loss = self.calc_flag_loss(prediction, flags)

        total_loss = (
            self.data_loss * self.data_loss_scale_factor + \
            self.smoothing_loss * self.smoothing_loss_scale_factor + \
            self.flag_loss * self.flag_loss_scale_factor
        )

        return total_loss 
    
    # Set scale factors for losses
    def set_loss_scale_factors(
        self,
        data_loss_scale_factor,
        smoothing_loss_scale_factor,
        flag_loss_scale_factor
    ):
        self.data_loss_scale_factor = data_loss_scale_factor
        self.smoothing_loss_scale_factor = smoothing_loss_scale_factor
        self.flag_loss_scale_factor = flag_loss_scale_factor

    # Define whether we are using a NUMERICAL vs ANALYTICAL gradient loss
    def set_gradient_loss_type(self, type):
        if (type == "NUMERICAL"):
            self.gradient_type = "NUMERICAL"
        elif(type == "ANALYTICAL"):
            self.gradient_type = "ANALYTICAL"
        else:
            raise ValueError("Please specify either NUMERICAL or ANALYTICAL gradient loss as a string")
        
    # Data loss - MAE between x-point GTs and preds
    def calc_data_loss(self, pred_xs, gt_xs):
        return torch.abs(pred_xs - gt_xs).mean()

    # Smoothing loss - MAE between gradient angle (tangent angle) of point pairs between GTs and preds
    def calc_smoothing_loss(self, pred_xs, gt_xs):
        pred_gradients = pred_xs[1 : ] - pred_xs[ : -1]
        gt_gradients = gt_xs[1 : ] - gt_xs[ : -1]

        loss = torch.abs(pred_gradients - gt_gradients).mean()

        return loss

    # Flags loss - binary cross entropy loss
    def calc_flag_loss(self, pred_flags, gt_flags):
        pred_flags_tensor = torch.tensor(pred_flags, dtype = torch.float32)
        gt_flags_tensor = torch.tensor(gt_flags, dtype = torch.float32)

        loss = torch.nn.functional.binary_cross_entropy(
            pred_flags_tensor,
            gt_flags_tensor
        )

        return loss
    
    # Loss backward pass
    def loss_backward(self):
        self.loss.backward()

    # Get total loss value
    def get_loss(self):
        return self.loss.item()

    # Get data loss
    def get_data_loss(self):
        scaled_data_loss = self.data_loss * self.data_loss_scale_factor
        return scaled_data_loss.item()
    
    # Get smoothing (gradient) loss
    def get_smoothing_loss(self):
        scaled_smoothing_loss = self.smoothing_loss * self.smoothing_loss_scale_factor
        return scaled_smoothing_loss.item()
    
    # Get flag loss
    def get_flag_loss(self):
        scaled_flag_loss = self.flag_loss * self.flag_loss_scale_factor
        return scaled_flag_loss.item()
    
    # Logging all losses
    def log_loss(self, log_count):
        self.writer.add_scalars(
            "Train", {
                "total_loss" : self.get_loss(),
                "data_loss" : self.get_data_loss(),
                "smoothing_loss" : self.get_smoothing_loss(),
                "flag_loss" : self.get_flag_loss()
            }, 
            (log_count)
        )

    # Run optimizer
    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Set train mode
    def set_train_mode(self):
        self.model = self.model.train()

    # Set evaluation mode
    def set_eval_mode(self):
        self.model = self.model.eval()

    # Save model
    def save_model(self, model_save_path):
        torch.save(
            self.model.state_dict(), 
            model_save_path
        )

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print("Finished training")

    # Save predicted visualization
    def save_visualization(self, log_count):

        # Get pred/gt tensors and detach em
        pred_xs = self.prediction.cpu().detach().numpy()
        gt_xs = self.xs_tensor.cpu().detach().numpy()

        # GROUNDTRUTH

        # Visualize image
        fig_gt = plt.figure(figsize = BEV_FIGSIZE)
        plt.axis("off")
        plt.imshow(self.image)

        # Plot BEV egopath
        plt.plot(
            [x * self.W for x in gt_xs],
            [y * self. H for y in self.ys],
            color = "yellow"
        )

        # Write fig
        self.writer.add_figure(
            "Groundtruth",
            fig_gt,
            global_step = (log_count)
        )

        # PREDICTION

        # Visualize image
        fig_pred = plt.figure(figsize = BEV_FIGSIZE)
        plt.axis("off")
        plt.imshow(self.image)

        # Plot BEV egopath
        plt.plot(
            [x * self.W for x in pred_xs],
            [y * self. H for y in self.ys],
            color = "yellow"
        )

        # Write fig
        self.writer.add_figure(
            "Prediction",
            fig_pred,
            global_step = (log_count)
        )

    # Run validation with metrics
    def validate(self, image, gt_xs, gt_ys, gt_flags):

        # Set data
        self.set_data(image, gt_xs, gt_ys, gt_flags)

        # Augment image
        self.apply_augmentations(is_train = False)

        # Tensor conversion
        self.load_data()

        # Run model
        prediction = self.model(self.image_tensor)

        # Validation loss
        val_loss_tensor = self.calc_data_loss(
            prediction,
            self.xs_tensor
        )

        val_loss = val_loss_tensor.detach().cpu().numpy()

        return val_loss
    
    