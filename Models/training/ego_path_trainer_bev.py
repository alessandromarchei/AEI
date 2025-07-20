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
from model_components.scene_seg_network import SceneSegNetwork
from model_components.ego_path_network import EgoPathNetwork
from data_utils.augmentations import Augmentations


class BEVEgoPathTrainer():
    def __init__(
        self,  
        checkpoint_path = "", 
        pretrained_checkpoint_path = "", 
        is_pretrained = False
    ):
        
        # Images and gts
        self.orig_vis = None
        self.image = None

        self.H = None
        self.W = None

        self.xs_bev_egopath = []
        self.xs_reproj_egopath = []
        self.xs_bev_egoleft = []
        self.xs_reproj_egoleft = []
        self.xs_bev_egoright = []
        self.xs_reproj_egoright = []

        self.ys_bev = []
        self.ys_reproj = []

        self.valids_egopath = []
        self.valids_egoleft = []
        self.valids_egoright = []

        self.mat = []

        # Tensors
        self.image_tensor = None

        self.xs_tensor_bev_egopath = []
        self.xs_tensor_reproj_egopath = []
        self.xs_tensor_bev_egoleft = []
        self.xs_tensor_reproj_egoleft = []
        self.xs_tensor_bev_egoright = []
        self.xs_tensor_reproj_egoright = []

        self.valids_tensor_egopath = []
        self.valids_tensor_egoleft = []
        self.valids_tensor_egoright = []

        # Model and preds
        self.model = None
        self.pred_xs_egopath = None
        self.pred_xs_egoleft = None
        self.pred_xs_egoright = None

        # Losses
        self.bev_data_loss = 0
        self.bev_gradient_loss = 0
        self.bev_loss = 0

        self.reproj_data_loss = 0
        self.reproj_gradient_loss = 0
        self.reproj_loss = 0

        self.total_loss = 0
        
        self.gradient_type = "NUMERICAL"

        # Loss scale factors
        self.alpha = 1.0        # Scale factor of bev_gradient_loss
        self.beta = 1.0         # Scale factor of reproj_gradient_loss
        self.gamma = 1.0        # Scale factor of reproj_loss

        self.BEV_FIGSIZE = (4, 8)
        self.ORIG_FIGSIZE = (8, 4)

        # Currently limiting to available datasets only. Will unlock eventually
        self.VALID_DATASET_LITERALS = Literal[
            # "BDD100K",
            # "COMMA2K19",
            # "CULANE",
            # "CURVELANES",
            # "ROADWORK",
            "TUSIMPLE"
        ]
        self.VALID_DATASET_LIST = list(get_args(self.VALID_DATASET_LITERALS))

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
        self.learning_rate = 0.00005
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
    def set_data(
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
        mat
    ):
        
        # Parse all data
        self.orig_vis = orig_vis
        h, w, _ = image.shape
        self.image = image
        self.H = h
        self.W = w
        self.xs_bev_egopath = np.array(
            xs_bev_egopath, 
            dtype = "float32"
        )
        self.xs_reproj_egopath = np.array(
            xs_reproj_egopath, 
            dtype = "float32"
        )
        self.xs_bev_egoleft = np.array(
            xs_bev_egoleft, 
            dtype = "float32"
        )
        self.xs_reproj_egoleft = np.array(
            xs_reproj_egoleft, 
            dtype = "float32"
        )
        self.xs_bev_egoright = np.array(
            xs_bev_egoright, 
            dtype = "float32"
        )
        self.xs_reproj_egoright = np.array(
            xs_reproj_egoright, 
            dtype = "float32"
        )
        self.ys_bev = np.array(
            ys_bev, 
            dtype = "float32"
        )
        self.ys_reproj = np.array(
            ys_reproj, 
            dtype = "float32"
        )
        self.valids_egopath = np.array(
            valids_egopath, 
            dtype = "float32"
        )
        self.valids_egoleft = np.array(
            valids_egoleft, 
            dtype = "float32"
        )
        self.valids_egoright = np.array(
            valids_egoright, 
            dtype = "float32"
        )
        self.mat = np.array(
            mat, 
            dtype = "float32"
        )

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

        # Xs of bev egopath
        xs_tensor_bev_egopath = torch.from_numpy(self.xs_bev_egopath)
        xs_tensor_bev_egopath = xs_tensor_bev_egopath.unsqueeze(0)
        self.xs_tensor_bev_egopath = xs_tensor_bev_egopath.to(self.device)
        
        # Xs of reproj egopath
        xs_tensor_reproj_egopath = torch.from_numpy(self.xs_reproj_egopath)
        xs_tensor_reproj_egopath = xs_tensor_reproj_egopath.unsqueeze(0)
        self.xs_tensor_reproj_egopath = xs_tensor_reproj_egopath.to(self.device)

        # Xs of bev egoleft
        xs_tensor_bev_egoleft = torch.from_numpy(self.xs_bev_egoleft)
        xs_tensor_bev_egoleft = xs_tensor_bev_egoleft.unsqueeze(0)
        self.xs_tensor_bev_egoleft = xs_tensor_bev_egoleft.to(self.device)

        # Xs of reproj egoleft
        xs_tensor_reproj_egoleft = torch.from_numpy(self.xs_reproj_egoleft)
        xs_tensor_reproj_egoleft = xs_tensor_reproj_egoleft.unsqueeze(0)
        self.xs_tensor_reproj_egoleft = xs_tensor_reproj_egoleft.to(self.device)

        # Xs of bev egoright
        xs_tensor_bev_egoright = torch.from_numpy(self.xs_bev_egoright)
        xs_tensor_bev_egoright = xs_tensor_bev_egoright.unsqueeze(0)
        self.xs_tensor_bev_egoright = xs_tensor_bev_egoright.to(self.device)

        # Xs of reproj egoright
        xs_tensor_reproj_egoright = torch.from_numpy(self.xs_reproj_egoright)
        xs_tensor_reproj_egoright = xs_tensor_reproj_egoright.unsqueeze(0)
        self.xs_tensor_reproj_egoright = xs_tensor_reproj_egoright.to(self.device)

        # Valids of egopath
        valids_tensor_egopath = torch.from_numpy(self.valids_egopath)
        valids_tensor_egopath = valids_tensor_egopath.unsqueeze(0)
        self.valids_tensor_egopath = valids_tensor_egopath.to(self.device)

        # Valids of egoleft
        valids_tensor_egoleft = torch.from_numpy(self.valids_egoleft)
        valids_tensor_egoleft = valids_tensor_egoleft.unsqueeze(0)
        self.valids_tensor_egoleft = valids_tensor_egoleft.to(self.device)

        # Valids of egoright
        valids_tensor_egoright = torch.from_numpy(self.valids_egoright)
        valids_tensor_egoright = valids_tensor_egoright.unsqueeze(0)
        self.valids_tensor_egoright = valids_tensor_egoright.to(self.device)
    
    # Run Model
    def run_model(self):
        self.pred_xs = self.model(self.image_tensor)
        self.loss = self.calc_loss(
            self.pred_xs, 
            self.xs_tensor,
            self.valids_tensor
        )

    # Calculate loss
    def calc_loss(self, pred_xs, xs, valids):
        self.data_loss = self.calc_data_loss(pred_xs, xs, valids)
        self.smoothing_loss = self.calc_smoothing_loss(pred_xs, xs, valids)

        total_loss = (
            self.data_loss * self.data_loss_scale_factor + \
            self.smoothing_loss * self.smoothing_loss_scale_factor
        )

        return total_loss 
    
    # Set scale factors for losses
    def set_loss_scale_factors(
        self,
        data_loss_scale_factor,
        smoothing_loss_scale_factor,
    ):
        self.data_loss_scale_factor = data_loss_scale_factor
        self.smoothing_loss_scale_factor = smoothing_loss_scale_factor

    # Define whether we are using a NUMERICAL vs ANALYTICAL gradient loss
    def set_gradient_loss_type(self, type):
        if (type == "NUMERICAL"):
            self.gradient_type = "NUMERICAL"
        elif(type == "ANALYTICAL"):
            self.gradient_type = "ANALYTICAL"
        else:
            raise ValueError("Please specify either NUMERICAL or ANALYTICAL gradient loss as a string")
        
    # Data loss - MAE between x-point GTs and preds
    def calc_data_loss(self, pred_xs, gt_xs, valids):
        num_valids = torch.sum(valids)
        loss = torch.sum(torch.abs(pred_xs * valids - gt_xs * valids)) / num_valids
        return loss

    # Smoothing loss - MAE between gradient angle (tangent angle) of point pairs between GTs and preds
    def calc_smoothing_loss(self, pred_xs, gt_xs, valids):
        pred_xs_valids = pred_xs * valids
        gt_xs_valids = gt_xs * valids
        pred_gradients = pred_xs_valids[0][1 : ] - pred_xs_valids[0][ : -1]
        gt_gradients = gt_xs_valids[0][1 : ] - gt_xs_valids[0][ : -1]

        num_valids = torch.sum(valids)
        loss = torch.sum(torch.abs(pred_gradients - gt_gradients)) / num_valids

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
    
    
    # Logging all losses
    def log_loss(self, log_count):
        self.writer.add_scalars(
            "Train", {
                "total_loss" : self.get_loss(),
                "data_loss" : self.get_data_loss(),
                "smoothing_loss" : self.get_smoothing_loss()
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
    def save_visualization(self, log_count, orig_vis):

        # Get pred/gt tensors and detach em
        pred_xs = self.pred_xs.cpu().detach().numpy()
        gt_xs = self.xs_tensor.cpu().detach().numpy()

        # BEV GROUNDTRUTH

        # Plot fig
        fig_gt_bev = self.visualizeBEV(gt_xs)

        # Write fig
        self.writer.add_figure(
            "BEV - Groundtruth",
            fig_gt_bev,
            global_step = (log_count)
        )

        # BEV PREDICTION

        # Plot fig
        fig_pred_bev = self.visualizeBEV(pred_xs)

        # Write fig
        self.writer.add_figure(
            "BEV - Prediction",
            fig_pred_bev,
            global_step = (log_count)
        )

        # ORIGINAL GROUNDTRUTH (basically just slap the visualization img)

        # Prep image tensor
        fig_orig = self.visualizeOriginal(pred_xs, self.mat)

        # Write image
        self.writer.add_figure(
            "Original - Groundtruth vs Prediction",
            fig_orig,
            global_step = (log_count)
        )

    # Run validation with metrics
    def validate(
        self, 
        orig_vis,
        image, 
        gt_xs, gt_ys, valids, mat, 
        save_path = None
    ):

        # Set data
        self.set_data(orig_vis, image, gt_xs, gt_ys, valids, mat)

        # Augment image
        self.apply_augmentations(is_train = False)

        # Tensor conversion
        self.load_data()

        # Run model
        self.pred_xs = self.model(self.image_tensor)

        # Validation loss
        val_data_loss_tensor = self.calc_data_loss(
            self.pred_xs,
            self.xs_tensor,
            self.valids_tensor
        )

        val_smoothing_loss_tensor = self.calc_smoothing_loss(
            self.pred_xs,
            self.xs_tensor,
            self.valids_tensor
        )

        val_data_loss = val_data_loss_tensor.detach().cpu().numpy()
        val_smoothing_loss = val_smoothing_loss_tensor.detach().cpu().numpy()
        sum_val_loss = val_data_loss + val_smoothing_loss

        # Save this visualization, both BEV and original
        if (save_path):
            pred_xs = self.pred_xs.detach().cpu().numpy()
            self.visualizeBEV(
                pred_xs,
                f"{save_path}_BEV.jpg"
            )
            self.visualizeOriginal(
                pred_xs,
                mat,
                f"{save_path}_Orig.jpg"
            )

        return sum_val_loss, val_data_loss, val_smoothing_loss
    
    # Log val loss to TensorBoard
    def log_validation(self, msdict):
        # Val score for each dataset
        val_score_payload = {}
        val_data_score_payload = {}
        val_smooth_score_payload = {}
        for dataset in self.VALID_DATASET_LIST:
            val_score_payload[dataset] = msdict[dataset]["val_score"]
            val_data_score_payload[dataset] = msdict[dataset]["val_data_score"]
            val_smooth_score_payload[dataset] = msdict[dataset]["val_smooth_score"]
            
            self.writer.add_scalars(
                f"Val Score - {dataset}", 
                {
                    "val_score" : val_score_payload[dataset],
                    "val_data" : val_data_score_payload[dataset],
                    "val_smooth" : val_smooth_score_payload[dataset]
                },
                (msdict["log_counter"])
            )

        # Overall val score
        self.writer.add_scalars(
            "Val Score - Overall",
            {
                "overall_val_score" : msdict["overall_val_score"],
                "overall_val_data_score" : msdict["overall_val_data_score"],
                "overall_val_smooth_score" : msdict["overall_val_smooth_score"]
            },
            (msdict["log_counter"])
        )

    # Validate network on TEST dataset and visualize result
    def test(
        self,
        image_test,
        save_path
    ):
        
        # Acquire test image
        frame = cv2.imread(image_test, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        test_img = Image.fromarray(frame).resize((320, 640))

        # Load as tensor
        test_img_tensor = self.image_loader(test_img).unsqueeze(0).to(self.device)

        # Model inference
        test_pred_xs = self.model(test_img_tensor).cpu().detach().numpy()

        # Vis it
        self.visualizeBEV(
            test_pred_xs,
            save_path
        )

    # Inverse transform: BEV -> original perspective
    def invTrans(
        self,
        line,
        transform_matrix
    ):
        inv_mat = np.linalg.inv(transform_matrix)
        orig_view = cv2.warpPerspective(
            self.image, inv_mat,
            (self.H, self.W)        # 320 x 640 ==> 640 x 320
        )
        np_line = np.array(
            line, 
            dtype = np.float32
        ).reshape(-1, 1, 2)
        orig_line = cv2.perspectiveTransform(np_line, inv_mat)
        orig_line = [tuple(point[0]) for point in orig_line]

        return orig_view, orig_line

    # Visualize BEV perspective
    def visualizeBEV(
        self,
        pred_xs,
        save_path = None
    ):
        # Visualize image
        H, W, _ = self.image.shape
        fig_BEV = plt.figure(figsize = self.BEV_FIGSIZE)
        plt.axis("off")
        plt.imshow(self.image)

        # Plot BEV egopath
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
            color = "yellow"
        )

        # Write fig
        if (save_path):
            fig_BEV.savefig(save_path)
        plt.close(fig_BEV)

        return fig_BEV

    # Visualize original perspective
    def visualizeOriginal(
        self,
        pred_xs,
        transform_matrix,
        save_path = None
    ):
        # Visualize image
        H, W, _ = self.image.shape
        fig_orig = plt.figure(figsize = self.ORIG_FIGSIZE)
        plt.axis("off")

        # Inverse transform
        _, orig_egopath = self.invTrans(
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

        # Plot original perspective and its egopath
        plt.imshow(self.orig_vis)
        orig_W, orig_H = self.orig_vis.size
        trimmed_orig_egopath = [
            p for p in orig_egopath
            if 0 <= p[0] <= orig_W
            and 0 <= p[1] <= orig_H
        ]
        plt.plot(
            [p[0] for p in trimmed_orig_egopath],
            [p[1] for p in trimmed_orig_egopath],
            color = "cyan"
        )

        # Write fig
        if (save_path):
            fig_orig.savefig(save_path)
        plt.close(fig_orig)

        return fig_orig