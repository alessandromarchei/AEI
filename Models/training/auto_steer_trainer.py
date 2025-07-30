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
from typing import Literal, get_args
import sys

sys.path.append('..')
from model_components.auto_steer_network import AutoSteerNetwork
from data_utils.augmentations import Augmentations


class AutoSteerTrainer():
    def __init__(
        self,  
        checkpoint_path = ""
    ):
        
        # Initializing Data
        self.homotrans_mat = None
        self.bev_image = None
        self.perspective_image = None
        self.bev_egopath = None
        self.bev_egoleft = None
        self.bev_egoright = None
        self.reproj_egopath = None
        self.reproj_egoleft = None
        self.reproj_egoright = None
        self.perspective_H = None
        self.perspective_W = None
        self.BEV_H = None
        self.BEV_W = None

        # Initializing BEV to Image transformation matrix
        self.homotrans_mat_tensor = None

        # Initializing BEV Image tensor
        self.bev_image_tensor = None

        # Initializing Ground Truth Tensors
        self.gt_bev_egopath_tensor = None
        self.gt_bev_egoleft_lane_tensor = None
        self.gt_bev_egoright_lane_tensor = None
        self.gt_reproj_egopath_tensor = None
        self.gt_reproj_egoleft_lane_tensor = None
        self.gt_reproj_egoright_lane_tensor = None

        # Model predictions
        self.pred_bev_ego_path_tensor = None
        self.pred_bev_egoleft_lane_tensor = None
        self.pred_bev_egoright_lane_tensor = None

        # Losses
        self.BEV_loss = None
        self.reprojected_loss = None
        self.total_loss = None

        self.BEV_FIGSIZE = (4, 8)
        self.ORIG_FIGSIZE = (8, 4)

        # Currently limiting to available datasets only. Will unlock eventually
        self.VALID_DATASET_LITERALS = Literal["TUSIMPLE"]
        self.VALID_DATASET_LIST = list(get_args(self.VALID_DATASET_LITERALS))

        # Checking devices (GPU vs CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        print(f"Using {self.device} for inference.")

        # Instantiate model
        self.model = AutoSteerNetwork()
        
        if(checkpoint_path):
            print("Loading trained AutoSteer model from checkpoint")
            self.model.load_state_dict(torch.load \
                (checkpoint_path, weights_only = True))  
        else:
            print("Loading vanilla AutoSteer model for training")
            
        self.model = self.model.to(self.device)
        
        # TensorBoard
        self.writer = SummaryWriter()

        # Learning rate and optimizer
        self.learning_rate = 0.0005
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
    def set_data(self, homotrans_mat, bev_image, perspective_image, \
                bev_egopath, bev_egoleft, bev_egoright, reproj_egopath, \
                reproj_egoleft, reproj_egoright):

        self.homotrans_mat = np.array(homotrans_mat, dtype = "float32")
        self.bev_image = np.array(bev_image)
        self.perspective_image = np.array(perspective_image)
        self.bev_egopath = np.array(bev_egopath, dtype = "float32").transpose()
        self.bev_egoleft = np.array(bev_egoleft, dtype = "float32").transpose()
        self.bev_egoright = np.array(bev_egoright, dtype = "float32").transpose()
        self.reproj_egopath = np.array(reproj_egopath, dtype = "float32").transpose()
        self.reproj_egoleft = np.array(reproj_egoleft, dtype = "float32").transpose()
        self.reproj_egoright = np.array(reproj_egoright, dtype = "float32").transpose()
        self.perspective_H, self.perspective_W, _ = self.perspective_image.shape
        self.BEV_H, self.BEV_W, _ = self.bev_image.shape

    # Image agumentations
    def apply_augmentations(self, is_train):
        # Augmenting data for train or val/test
        aug = Augmentations(
            is_train = is_train, 
            data_type = "KEYPOINTS"
        )
        aug.setImage(self.bev_image)
        self.bev_image = aug.applyTransformKeypoint(self.bev_image)

    # Load data as Pytorch tensors
    def load_data(self):

        # BEV to Image matrix
        homotrans_mat_tensor = torch.from_numpy(self.homotrans_mat)
        homotrans_mat_tensor = homotrans_mat_tensor.type(torch.FloatTensor)
        self.homotrans_mat_tensor = homotrans_mat_tensor.to(self.device)

        # BEV Image
        bev_image_tensor = self.image_loader(self.bev_image)
        bev_image_tensor = bev_image_tensor.unsqueeze(0)
        self.bev_image_tensor = bev_image_tensor.to(self.device)

        # BEV Egopath
        bev_egopath_tensor = torch.from_numpy(self.bev_egopath)
        bev_egopath_tensor = bev_egopath_tensor.type(torch.FloatTensor)
        self.gt_bev_egopath_tensor = bev_egopath_tensor.to(self.device)

        # BEV Egoleft Lane
        bev_egoleft_lane_tensor = torch.from_numpy(self.bev_egoleft)
        bev_egoleft_lane_tensor = bev_egoleft_lane_tensor.type(torch.FloatTensor)
        self.gt_bev_egoleft_lane_tensor = bev_egoleft_lane_tensor.to(self.device)

        # BEV Egoright Lane
        bev_egoright_lane_tensor = torch.from_numpy(self.bev_egoright)
        bev_egoright_lane_tensor = bev_egoright_lane_tensor.type(torch.FloatTensor)
        self.gt_bev_egoright_lane_tensor = bev_egoright_lane_tensor.to(self.device)
        
        # Reprojected Egopath
        reproj_egopath_tensor = torch.from_numpy(self.reproj_egopath)
        reproj_egopath_tensor = reproj_egopath_tensor.type(torch.FloatTensor)
        self.gt_reproj_egopath_tensor = reproj_egopath_tensor.to(self.device)

        # Reprojected Egoleft Lane
        reproj_egoleft_lane_tensor = torch.from_numpy(self.reproj_egoleft)
        reproj_egoleft_lane_tensor = reproj_egoleft_lane_tensor.type(torch.FloatTensor)
        self.gt_reproj_egoleft_lane_tensor = reproj_egoleft_lane_tensor.to(self.device)

        # Reprojected Egoright Lane
        reproj_egoright_lane_tensor = torch.from_numpy(self.reproj_egoright)
        reproj_egoright_lane_tensor = reproj_egoright_lane_tensor.type(torch.FloatTensor)
        self.gt_reproj_egoright_lane_tensor = reproj_egoright_lane_tensor.to(self.device)
    
    # Run Model
    def run_model(self):
        self.pred_bev_ego_path_tensor, self.pred_bev_egoleft_lane_tensor, \
            self.pred_bev_egoright_lane_tensor = self.model(self.bev_image_tensor)

        # BEV Loss
        BEV_data_loss_driving_corridor = self.calc_BEV_data_loss_driving_corridor()
        BEV_gradient_loss_driving_corridor = self.calc_BEV_gradient_loss_driving_corridor()

        self.BEV_loss = BEV_data_loss_driving_corridor + \
            BEV_gradient_loss_driving_corridor
   
        # Reprojected Loss
        reprojected_data_loss_driving_corridor = self.calc_reprojected_data_loss_driving_corridor()
        reprojected_gradient_loss_driving_corridor = self.calc_reprojected_gradient_loss_driving_corridor()

        self.reprojected_loss = reprojected_data_loss_driving_corridor + \
            reprojected_gradient_loss_driving_corridor

        # Total Loss
        self.total_loss = self.BEV_loss + self.reprojected_loss

    # BEV Data Loss for the entire driving corridor
    def calc_BEV_data_loss_driving_corridor(self):

        BEV_egopath_data_loss = \
            self.calc_BEV_data_loss(self.gt_bev_egopath_tensor, self.pred_bev_ego_path_tensor)

        BEV_egoleft_lane_data_loss = \
            self.calc_BEV_data_loss(self.gt_bev_egoleft_lane_tensor, self.pred_bev_egoleft_lane_tensor)
   
        BEV_egoright_lane_data_loss = \
            self.calc_BEV_data_loss(self.gt_bev_egoright_lane_tensor, self.pred_bev_egoright_lane_tensor)


        BEV_data_loss_driving_corridor = BEV_egopath_data_loss +  \
            BEV_egoleft_lane_data_loss + BEV_egoright_lane_data_loss
        
        return BEV_data_loss_driving_corridor
    
    # BEV Gradient Loss for the entire driving corridor
    def calc_BEV_gradient_loss_driving_corridor(self):

        BEV_egopath_gradient_loss = \
            self.calc_BEV_graient_loss(self.gt_bev_egopath_tensor, 
                                       self.pred_bev_ego_path_tensor)

        BEV_egoleft_lane_gradient_loss = \
            self.calc_BEV_graient_loss(self.gt_bev_egoleft_lane_tensor, 
                                       self.pred_bev_egoleft_lane_tensor)
   
        BEV_egoright_lane_gradient_loss = \
            self.calc_BEV_graient_loss(self.gt_bev_egoright_lane_tensor, 
                                       self.pred_bev_egoright_lane_tensor)


        BEV_gradient_loss_driving_corridor = BEV_egopath_gradient_loss +  \
            BEV_egoleft_lane_gradient_loss + BEV_egoright_lane_gradient_loss
        
        return BEV_gradient_loss_driving_corridor
    
    # Reprojected Data Loss for the entire driving corridor
    def calc_reprojected_data_loss_driving_corridor(self):

        reprojected_ego_path_data_loss =  \
            self.calc_reprojected_data_loss(self.gt_reproj_egopath_tensor, 
                                            self.gt_bev_egopath_tensor, 
                                            self.pred_bev_ego_path_tensor)
        
        reprojected_egoleft_lane_data_loss =  \
            self.calc_reprojected_data_loss(self.gt_reproj_egoleft_lane_tensor, 
                                            self.gt_bev_egoleft_lane_tensor, 
                                            self.pred_bev_egoleft_lane_tensor)
        
        reprojected_egoright_lane_data_loss =  \
            self.calc_reprojected_data_loss(self.gt_reproj_egoright_lane_tensor, 
                                            self.gt_bev_egoright_lane_tensor, 
                                            self.pred_bev_egoright_lane_tensor)

        reprojected_data_loss_driving_corridor = reprojected_ego_path_data_loss + \
            reprojected_egoleft_lane_data_loss + reprojected_egoright_lane_data_loss
        
        return reprojected_data_loss_driving_corridor
    
    # Reprojected Gradient Loss for the entire driving corridor
    def calc_reprojected_gradient_loss_driving_corridor(self):

        reprojected_ego_path_gradient_loss =  \
            self.calc_reprojected_gradient_loss(self.gt_reproj_egopath_tensor, 
                                            self.gt_bev_egopath_tensor, 
                                            self.pred_bev_ego_path_tensor)
        
        reprojected_egoleft_lane_gradient_loss =  \
            self.calc_reprojected_gradient_loss(self.gt_reproj_egoleft_lane_tensor, 
                                            self.gt_bev_egoleft_lane_tensor, 
                                            self.pred_bev_egoleft_lane_tensor)
        
        reprojected_egoright_lane_gradient_loss =  \
            self.calc_reprojected_gradient_loss(self.gt_reproj_egoright_lane_tensor, 
                                            self.gt_bev_egoright_lane_tensor, 
                                            self.pred_bev_egoright_lane_tensor)

        reprojected_gradient_loss_driving_corridor = reprojected_ego_path_gradient_loss + \
            reprojected_egoleft_lane_gradient_loss + reprojected_egoright_lane_gradient_loss
        
        return reprojected_gradient_loss_driving_corridor
    
    # BEV Data Loss for a single lane/path element
    # Mean absolute error on predictions
    def calc_BEV_data_loss(self, gt_tensor, pred_tensor):

        gt_tensor_x_vals = gt_tensor[0,:]
        pred_tensor_x_vals = pred_tensor[0]

        data_error_sum = 0
        num_valid_samples = 0

        for i in range(0, len(gt_tensor_x_vals)):

            if(gt_tensor_x_vals[i] >=0 and gt_tensor_x_vals[i] < 1):
                
                error = torch.abs(gt_tensor_x_vals[i] - pred_tensor_x_vals[i])
                data_error_sum = data_error_sum + error
                num_valid_samples = num_valid_samples + 1

        bev_data_loss = data_error_sum/num_valid_samples

        return bev_data_loss
    
    # BEV gradient loss for a single lane/path element
    # Sum of finite difference gradients
    def calc_BEV_graient_loss(self, gt_tensor, pred_tensor):

        gt_tensor_x_vals = gt_tensor[0,:]
        pred_tensor_x_vals = pred_tensor[0]

        bev_gradient_loss = 0

        for i in range(0, len(gt_tensor_x_vals) - 1):

            if(gt_tensor_x_vals[i] >=0 and gt_tensor_x_vals[i] < 1):

                gt_gradient = gt_tensor_x_vals[i+1] - gt_tensor_x_vals[i]
                pred_gradient = pred_tensor_x_vals[i+1] - pred_tensor_x_vals[i]

                error = torch.abs(gt_gradient - pred_gradient)
                bev_gradient_loss = bev_gradient_loss + error

        return bev_gradient_loss
    
    # Reprojected Data Loss for a single lane/path element
    # Mean absolute error on predictions
    def calc_reprojected_data_loss(self, gt_reprojected_tesnor, gt_tensor, pred_tensor):

        prediction_reprojected, _ = \
            self.getPerspectivePointsFromBEV(gt_tensor, pred_tensor)
        
        gt_tensor_x_vals = gt_tensor[0,:]
        gt_reprojected_tensor_x_vals = gt_reprojected_tesnor[0,:]
        gt_reprojected_tensor_y_vals = gt_reprojected_tesnor[1,:]

        data_error_sum = 0
        num_valid_samples = 0

        for i in range(0, len(gt_tensor_x_vals)):

            if(gt_tensor_x_vals[i] >=0 and gt_tensor_x_vals[i] < 1):

                gt_reprojected_x = gt_reprojected_tensor_x_vals[i]
                prediction_reprojected_x = prediction_reprojected[i][0]
                
                gt_reprojected_y = gt_reprojected_tensor_y_vals[i]
                prediction_reprojected_y = prediction_reprojected[i][1]
                
                x_error = torch.abs(gt_reprojected_x - prediction_reprojected_x)
                y_error = torch.abs(gt_reprojected_y - prediction_reprojected_y)
                L1_error = x_error + y_error

                data_error_sum = data_error_sum + L1_error
                num_valid_samples = num_valid_samples + 1

        reprojected_data_loss = data_error_sum/num_valid_samples
        return reprojected_data_loss
    
    # Reprojected points gradient loss for a single lane/path element
    # Sum of finite difference gradients
    def calc_reprojected_gradient_loss(self, gt_reprojected_tesnor, gt_tensor, pred_tensor):

        prediction_reprojected, _ = \
            self.getPerspectivePointsFromBEV(gt_tensor, pred_tensor)
        
        gt_tensor_x_vals = gt_tensor[0,:]
        gt_reprojected_tensor_x_vals = gt_reprojected_tesnor[0,:]

        reprojected_gradient_loss = 0

        for i in range(0, len(gt_tensor_x_vals)-1):

            if(gt_tensor_x_vals[i] >=0 and gt_tensor_x_vals[i] < 1):

                gt_reprojected_gradient = gt_reprojected_tensor_x_vals[i+1] \
                    - gt_reprojected_tensor_x_vals[i]
                
                prediction_reprojected_gradient = prediction_reprojected[i+1][0] \
                    - prediction_reprojected[i][0]
                
                error = torch.abs(gt_reprojected_gradient - prediction_reprojected_gradient)
                reprojected_gradient_loss = reprojected_gradient_loss + error

        return reprojected_gradient_loss

    # Get the list of reprojected points from X,Y BEV coordinates
    def getPerspectivePointsFromBEV(self, gt_tensor, pred_tensor):
        gt_tensor_y_vals = gt_tensor[1,:]
        pred_tensor_x_vals = pred_tensor[0]

        perspective_image_points, perspective_image_points_normalized = \
            self.projectBEVtoImage(pred_tensor_x_vals, gt_tensor_y_vals)

        return perspective_image_points_normalized, perspective_image_points

    # Reproject BEV points to perspective image
    def projectBEVtoImage(self, bev_x_points, bev_y_points):

        perspective_image_points = []
        perspective_image_points_normalized = []

        for i in range(0, len(bev_x_points)):
            
            image_homogenous_point_x = self.BEV_W*bev_x_points[i]*self.homotrans_mat_tensor[0][0] + \
                self.BEV_H*bev_y_points[i]*self.homotrans_mat_tensor[0][1] + self.homotrans_mat_tensor[0][2]
            
            image_homogenous_point_y = self.BEV_W*bev_x_points[i]*self.homotrans_mat_tensor[1][0] + \
                self.BEV_H*bev_y_points[i]*self.homotrans_mat_tensor[1][1] + self.homotrans_mat_tensor[1][2]
            
            image_homogenous_point_scale_factor = self.BEV_W*bev_x_points[i]*self.homotrans_mat_tensor[2][0] + \
                self.BEV_H*bev_y_points[i]*self.homotrans_mat_tensor[2][1] + self.homotrans_mat_tensor[2][2]
            
            image_point = [(image_homogenous_point_x/image_homogenous_point_scale_factor), \
                (image_homogenous_point_y/image_homogenous_point_scale_factor)]
            
            image_point_normalized = [image_point[0]/self.perspective_W, image_point[1]/self.perspective_H]

            perspective_image_points.append(image_point)
            perspective_image_points_normalized.append(image_point_normalized)

        return perspective_image_points, perspective_image_points_normalized

    # Loss backward pass
    def loss_backward(self):
        self.total_loss.backward()

    # Get total loss values
    def get_total_loss(self):
        return self.total_loss.item()
    
    def get_bev_loss(self):
        return self.BEV_loss.item()
    
    def get_reprojected_loss(self):
        return self.reprojected_loss.item()

    # Logging losses - Total, BEV, Reprojected
    def log_loss(self, log_count):
        self.writer.add_scalars(
            "Train_EgoPath", {
                "Total_loss" : self.get_total_loss(),
                "BEV_loss" : self.get_bev_loss(),
                "Reprojected_loss" : self.get_reprojected_loss()
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
    def save_visualization(self, log_count, bev_vis, perspective_vis):

        # Predicted Egopath (BEV)
        pred_bev_ego_path = self.pred_bev_ego_path_tensor.cpu().detach().numpy()
        
        # Predicted Egopath (Reprojected)
        _, pred_reprojected_ego_path_tensor = \
                self.getPerspectivePointsFromBEV(self.gt_bev_egopath_tensor, 
                                            self.pred_bev_ego_path_tensor)
        pred_reprojected_ego_path = pred_reprojected_ego_path_tensor.cpu().detach.numpy()
        pred_reprojected_ego_path_x_vals = [point[0] for point in pred_reprojected_ego_path]
        pred_reprojected_ego_path_y_vals = [point[1] for point in pred_reprojected_ego_path]

        # Predicted Egoleft Lane (BEV)
        prev_bev_egoleft_lane = self.pred_bev_egoleft_lane_tensor.cpu().detach().numpy()

        # Predicted Egoleft Lane (Reprojected)
        _, pred_reprojected_egoleft_lane_tensor = \
                self.getPerspectivePointsFromBEV(self.gt_bev_egoleft_lane_tensor, 
                                            self.pred_bev_egoleft_lane_tensor)
        pred_reprojected_egoleft_lane = pred_reprojected_egoleft_lane_tensor.cpu().detach.numpy()
        pred_reprojected_egoleft_lane_x_vals = [point[0] for point in pred_reprojected_egoleft_lane]
        pred_reprojected_egoleft_lane_y_vals = [point[1] for point in pred_reprojected_egoleft_lane]

        # Predicted Egoright Lane (BEV)
        pred_bev_egoright_lane = self.pred_bev_egoright_lane_tensor.cpu().detach().numpy()

        # Predicted Egoright Lane (Reprojected)
        _, pred_reprojected_egoright_lane_tensor = \
                self.getPerspectivePointsFromBEV(self.gt_bev_egoright_lane_tensor, 
                                            self.pred_bev_egoright_lane_tensor)
        pred_reprojected_egoright_lane = pred_reprojected_egoright_lane_tensor.cpu().detach.numpy()
        pred_reprojected_egoright_lane_x_vals = [point[0] for point in pred_reprojected_egoright_lane]
        pred_reprojected_egoright_lane_y_vals = [point[1] for point in pred_reprojected_egoright_lane]

        # BEV fixed y-values of anchors
        bev_y_vals = self.gt_bev_egopath_tensor[1,:].cpu().detach().numpy()*self.BEV_H

        # Visualize Predictions - BEV
        fig_pred_bev = plt.figure(figsize=(8, 4))
        plt.imshow(self.bev_image)
        plt.plot(pred_bev_ego_path*self.BEV_W, bev_y_vals, 'yellow')
        plt.plot(prev_bev_egoleft_lane*self.BEV_W, bev_y_vals, 'green')
        plt.plot(pred_bev_egoright_lane*self.BEV_W, bev_y_vals, 'cyan')
        self.writer.add_figure("Prediction (BEV)", fig_pred_bev, global_step = (log_count))
        
        # Visualize Ground Truth - BEV
        fig_gt_bev = plt.figure(figsize=(8, 4))
        plt.imshow(bev_vis)
        self.writer.add_figure("Ground Truth (BEV)", fig_gt_bev, global_step = (log_count))

        # Visualize Predictions - Perspective
        fig_pred_perspective = plt.figure(figsize=(4, 8))
        plt.imshow(self.perspective_image)
        plt.plot(pred_reprojected_ego_path_x_vals, pred_reprojected_ego_path_y_vals, 'yellow')
        plt.plot(pred_reprojected_egoleft_lane_x_vals, pred_reprojected_egoleft_lane_y_vals, 'green')
        plt.plot(pred_reprojected_egoright_lane_x_vals, pred_reprojected_egoright_lane_y_vals, 'cyan')
        self.writer.add_figure("Prediction (Perspective)", fig_pred_perspective, global_step = (log_count))

        # Visualize Ground Truth - Perspective
        fig_gt_perspective = plt.figure(figsize=(4, 8))
        plt.imshow(perspective_vis)
        self.writer.add_figure("Ground Truth (Perspective)", fig_gt_perspective, global_step = (log_count))
      
        
    # Run validation with metrics
    def validate(
        self,
        orig_vis, image, 
        xs_bev_egopath,
        xs_reproj_egopath,
        xs_bev_egoleft,
        xs_reproj_egoleft,
        xs_bev_egoright,
        xs_reproj_egoright,
        ys_bev,
        ys_reproj,
        valids_egopath,
        valids_egoleft,
        valids_egoright,
        mat,
        save_path = None
    ):

        # Set data
        self.set_data(
            orig_vis, image, 
            xs_bev_egopath,
            xs_reproj_egopath,
            xs_bev_egoleft,
            xs_reproj_egoleft,
            xs_bev_egoright,
            xs_reproj_egoright,
            ys_bev,
            ys_reproj,
            valids_egopath,
            valids_egoleft,
            valids_egoright,
            mat
        )

        # Augment image
        self.apply_augmentations(is_train = False)

        # Tensor conversion
        self.load_data()

        # Run model
        self.pred_xs_egopath, self.pred_xs_egoleft, self.pred_xs_egoright = self.model(self.image_tensor)

        # Validation losses
        
        # Egopath
        val_bev_loss_egopath = self.calc_bev_loss(
            self.pred_xs_egopath,
            self.xs_bev_egopath,
            self.valids_tensor_egopath
        )
        val_reproj_loss_egopath = self.calc_reproj_loss(
            self.reproject_line(self.pred_xs_egopath, self.mat),
            self.xs_reproj_egopath,
            self.valids_tensor_egopath
        )
        val_total_loss_egopath = self.calc_total_loss(
            val_bev_loss_egopath,
            val_reproj_loss_egopath
        )

        # Egoleft
        val_bev_loss_egoleft = self.calc_bev_loss(
            self.pred_xs_egoleft,
            self.xs_bev_egoleft,
            self.valids_tensor_egoleft
        )
        val_reproj_loss_egoleft = self.calc_reproj_loss(
            self.reproject_line(self.pred_xs_egoleft, self.mat),
            self.xs_reproj_egoleft,
            self.valids_tensor_egoleft
        )
        val_total_loss_egoleft = self.calc_total_loss(
            val_bev_loss_egoleft,
            val_reproj_loss_egoleft
        )

        # Egoright
        val_bev_loss_egoright = self.calc_bev_loss(
            self.pred_xs_egoright,
            self.xs_bev_egoright,
            self.valids_tensor_egoright
        )
        val_reproj_loss_egoright = self.calc_reproj_loss(
            self.reproject_line(self.pred_xs_egoright, self.mat),
            self.xs_reproj_egoright,
            self.valids_tensor_egoright
        )
        val_total_loss_egoright = self.calc_total_loss(
            val_bev_loss_egoright,
            val_reproj_loss_egoright
        )

        # Save this visualization, both BEV and original
        if (save_path):
            self.visualizeBEV(
                list_pred_xs = [
                    self.pred_xs_egopath.cpu().detach().numpy(),
                    self.pred_xs_egoleft.cpu().detach().numpy(),
                    self.pred_xs_egoright.cpu().detach().numpy()
                ],
                list_labels = [
                    "Predicted_EgoPath",
                    "Predicted_EgoLeft",
                    "Predicted_EgoRight"
                ],
                list_colors = [
                    "yellow",
                    "green",
                    "cyan"
                ],
                save_path = f"{save_path}_BEV.jpg"
            )
            self.visualizeOriginal(
                list_pred_xs = [
                    self.pred_xs_egopath.cpu().detach().numpy(),
                    self.pred_xs_egoleft.cpu().detach().numpy(),
                    self.pred_xs_egoright.cpu().detach().numpy()
                ],
                list_labels = [
                    "Predicted_EgoPath_reprojected",
                    "Predicted_EgoLeft_reprojected",
                    "Predicted_EgoRight_reprojected"
                ],
                list_colors = [
                    "yellow",
                    "green",
                    "cyan"
                ],
                transform_matrix = mat,
                save_path = f"{save_path}_Orig.jpg"
            )

        return [
            val_total_loss_egopath,
            val_bev_loss_egopath,
            val_reproj_loss_egopath,
            val_total_loss_egoleft,
            val_bev_loss_egoleft,
            val_reproj_loss_egoleft,
            val_total_loss_egoright,
            val_bev_loss_egoright,
            val_reproj_loss_egoright
        ]
    
    # Log val loss to TensorBoard
    def log_validation(self, msdict):
        
        # Val score for each dataset
        for dataset in self.VALID_DATASET_LIST:
            
            # Egopath
            self.writer.add_scalars(
                f"Val Scores - EgoPath - {dataset}", 
                {
                    "val_total" : msdict[dataset]["val_egopath"]["total_score"],
                    "val_bev" : msdict[dataset]["val_egopath"]["bev_score"],
                    "val_reproj" : msdict[dataset]["val_egopath"]["reproj_score"]
                },
                (msdict["log_counter"])
            )

            # Egoleft
            self.writer.add_scalars(
                f"Val Scores - EgoPath - {dataset}", 
                {
                    "val_total" : msdict[dataset]["val_egoleft"]["total_score"],
                    "val_bev" : msdict[dataset]["val_egoleft"]["bev_score"],
                    "val_reproj" : msdict[dataset]["val_egoleft"]["reproj_score"]
                },
                (msdict["log_counter"])
            )

            # Egoright
            self.writer.add_scalars(
                f"Val Scores - EgoLeft - {dataset}", 
                {
                    "val_total" : msdict[dataset]["val_egoright"]["total_score"],
                    "val_bev" : msdict[dataset]["val_egoright"]["bev_score"],
                    "val_reproj" : msdict[dataset]["val_egoright"]["reproj_score"]
                },
                (msdict["log_counter"])
            )

        # Overall val score
        self.writer.add_scalars(
            "Val Score - Overall",
            {
                "overall_val_total" : msdict["overall_val_total_score"],
                "overall_val_bev" : msdict["overall_val_bev_score"],
                "overall_val_reproj" : msdict["overall_val_reproj_score"]
            },
            (msdict["log_counter"])
        )

    # Validate network on TEST dataset and visualize result
    # def test(
    #     self,
    #     image_test,
    #     save_path
    # ):
        
    #     # Acquire test image
    #     frame = cv2.imread(image_test, cv2.IMREAD_COLOR)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     test_img = Image.fromarray(frame).resize((320, 640))

    #     # Load as tensor
    #     test_img_tensor = self.image_loader(test_img).unsqueeze(0).to(self.device)

    #     # Model inference
    #     test_pred_xs = self.model(test_img_tensor).cpu().detach().numpy()

    #     # Vis it
    #     self.visualizeBEV(
    #         test_pred_xs,
    #         save_path
    #     )

    # Inverse transform: BEV -> original perspective
    def invTrans(
        self,
        line,
        transform_matrix
    ):
        inv_mat = np.linalg.inv(transform_matrix)

        np_line = np.array(
            line, 
            dtype = np.float32
        ).reshape(-1, 1, 2)
        orig_line = cv2.perspectiveTransform(np_line, inv_mat)
        orig_line = [tuple(point[0]) for point in orig_line]
        return orig_line


    # Visualize BEV perspective
    def visualizeBEV(
        self,
        list_pred_xs : list,
        list_colors : list,
        list_labels : list,
        save_path = None
    ):
        # Visualize image
        H, W, _ = self.image.shape
        fig_BEV = plt.figure(figsize = self.BEV_FIGSIZE)
        plt.axis("off")
        plt.imshow(self.image)

        # Plot BEV lines
        assert len(list_pred_xs) == len(list_colors) == len(list_labels), \
            f"Mismatched lengths: {len(list_pred_xs)}, {len(list_colors)}, {len(list_labels)}"
        for i in range(len(list_pred_xs)):
            pred_xs = list_pred_xs[i]
            color = list_colors[i]
            label = list_labels[i]
            BEV_egopath = list(zip(pred_xs[0], self.ys))
            renormed_BEV_egopath = [
                (
                    BEV_egopath[i][0] * W, 
                    BEV_egopath[i][1] * H
                ) 
                for i in range(len(BEV_egopath))
                if (self.valids[i] > 0.0)
            ]
            plt.plot(
                [p[0] for p in renormed_BEV_egopath],
                [p[1] for p in renormed_BEV_egopath],
                color = color,
                label = label
            )

        # Write fig
        if (save_path):
            fig_BEV.savefig(save_path)
        plt.close(fig_BEV)

        return fig_BEV

    # Visualize original perspective
    def visualizeOriginal(
        self,
        list_pred_xs : list,
        list_colors : list,
        list_labels : list,
        transform_matrix,
        save_path = None
    ):
        # Visualize image
        H, W, _ = self.image.shape
        fig_orig = plt.figure(figsize = self.ORIG_FIGSIZE)
        plt.axis("off")
        plt.imshow(self.orig_vis)
        orig_W, orig_H = self.orig_vis.size

        assert len(list_pred_xs) == len(list_colors) == len(list_labels), \
            f"Mismatched lengths: {len(list_pred_xs)}, {len(list_colors)}, {len(list_labels)}"
        
        for i in range(len(list_pred_xs)):
            pred_xs = list_pred_xs[i]
            color = list_colors[i]
            label = list_labels[i]

            # Reproject BEV to original perspective
            reproj_line = self.invTrans(
                list(zip(
                    [
                        pred_xs[0][i] * W
                        for i in range(len(pred_xs[0]))
                        if (self.valids[i] > 0.0)
                    ],
                    [
                        self.ys[i] * H
                        for i in range(len(self.ys))
                        if (self.valids[i] > 0.0)
                    ]
                )),
                transform_matrix
            )

            # Plot reproj line (valids only)
            trimmed_reproj_line = [
                p for p in reproj_line
                if 0 <= p[0] <= orig_W
                and 0 <= p[1] <= orig_H
            ]
            plt.plot(
                [p[0] for p in trimmed_reproj_line],
                [p[1] for p in trimmed_reproj_line],
                color = color,
                label = label
            )

        # Write fig
        if (save_path):
            fig_orig.savefig(save_path)
        plt.close(fig_orig)

        return fig_orig