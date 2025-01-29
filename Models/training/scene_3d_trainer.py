
import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from typing import Literal
import numpy as np
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.scene_3d_network import Scene3DNetwork
from data_utils.augmentations import Augmentations

class Scene3DTrainer():
    def __init__(self,  checkpoint_path = '', pretrained_checkpoint_path = '', is_pretrained = False):

        self.image = 0
        self.image_val = 0
        self.gt = 0
        self.gt_val = 0
        self.validity = 0
        self.validity_val = 0
        self.validity_tensor = 0
        self.validity_val_tensor = 0
        self.augmented = 0
        self.augmented_val = 0
        self.image_tensor = 0
        self.image_val_tensor = 0
        self.gt_tensor = 0
        self.gt_val_tensor = 0
        self.loss = 0
        self.prediction = 0
        self.calc_loss = 0
        self.model = 0

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        if(is_pretrained):

            # Instantiate Model for validation or inference - load both pre-traiend SceneSeg and SuperDepth weights
            if(len(checkpoint_path) > 0):

                # Loading model with full pre-trained weights
                sceneSegNetwork = SceneSegNetwork()
                self.model = Scene3DNetwork(sceneSegNetwork)

                # If the model is also pre-trained then load the pre-trained downstream weights
                self.model.load_state_dict(torch.load \
                    (checkpoint_path, weights_only=True, map_location=self.device))
                print('Loading pre-trained model weights of Scene3D and upstream SceneSeg weights as well')
            else:
                raise ValueError('Please ensure Scene3D network weights are provided for downstream elements')
            
        else:

            # Instantiate Model for training - load pre-traiend SceneSeg weights only
            if(len(pretrained_checkpoint_path) > 0):
                
                # Loading SceneSeg pre-trained for upstream weights
                sceneSegNetwork = SceneSegNetwork()
                sceneSegNetwork.load_state_dict(torch.load \
                    (pretrained_checkpoint_path, weights_only=True, map_location=self.device))
                    
                # Loading model with pre-trained upstream weights
                self.model = Scene3DNetwork(sceneSegNetwork)
                print('Loading pre-trained model weights of upstream SceneSeg only, Scene3D initialised with random weights')
            else:
                raise ValueError('Please ensure SceneSeg network weights are provided for upstream elements')
        
       
        # Model to device
        self.model = self.model.to(self.device)
        
        # TensorBoard
        self.writer = SummaryWriter()

        # Learning rate and optimizer
        self.learning_rate = 0.00001
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)

        # Loaders
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        # Loaders
        self.validity_loader = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

        # Gradient filters
        # Gradient - x
        self.gx_filter = torch.Tensor([[1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]])
        self.gx_filter = self.gx_filter.view((1,1,3,3))
        self.gx_filter = self.gx_filter.type(torch.cuda.FloatTensor)
        self.gx_filter.to(self.device)
        
        # Gradient - y
        self.gy_filter = torch.Tensor([[1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]])
        self.gy_filter = self.gy_filter.view((1,1,3,3))
        self.gy_filter = self.gy_filter.type(torch.cuda.FloatTensor)
        self.gy_filter.to(self.device)

    # Logging Training Loss
    def log_loss(self, log_count):
        self.writer.add_scalar("Loss/train", self.get_loss(), (log_count))

    # Logging Validation mAE overall
    def log_val_mAE(self, mAE_overall, mAE_kitti, 
                        mAE_ddad, mAE_urbansyn, log_count):
        
        print('Logging Validation')      
        
        self.writer.add_scalars("Val/mAE_dataset",{
            'mAE_kitti': mAE_kitti,
            'mAE_ddad': mAE_ddad,
            'mAE_urbansyn': mAE_urbansyn
        }, (log_count))

        self.writer.add_scalar("Val/mAE", mAE_overall, (log_count))        

    # Assign input variables
    def set_data(self, image, gt, validity):
        self.image = image
        self.gt = gt
        self.validity = validity

    def set_val_data(self, image_val, gt_val, validity_val):
        self.image_val = image_val
        self.gt_val = gt_val
        self.validity_val = validity_val
    
    # Image agumentations
    def apply_augmentations(self, is_train):

        if(is_train):
            # Augmenting Data for training
            augTrain = Augmentations(is_train=True, data_type='DEPTH')
            augTrain.setDataDepth(self.image, self.gt, self.validity)
            self.image, self.augmented, self.validity = \
                augTrain.applyTransformDepth(image=self.image, 
                                             ground_truth=self.gt, validity=self.validity)
        else:
            # Augmenting Data for testing/validation
            augVal = Augmentations(is_train=False, data_type='DEPTH')
            augVal.setDataDepth(self.image_val, self.gt_val, self.validity_val)
            self.image_val, self.augmented_val, self.validity_val = \
                augVal.applyTransformDepth(image=self.image_val, 
                                           ground_truth=self.gt_val, validity=self.validity_val)
    
    # Load Data
    def load_data(self, is_train):
        self.load_image_tensor(is_train)
        self.load_gt_tensor(is_train)
        self.load_validity_tensor(is_train)


    def mAE_validity_loss(self):

        mAE = torch.abs(self.prediction - self.gt_tensor)*(self.validity_tensor)
        mAE_loss = torch.mean(mAE)

        return mAE_loss
    
    def edge_validity_loss(self):

        G_x_pred = nn.functional.conv2d(self.prediction, self.gx_filter, padding=1)
        G_y_pred = nn.functional.conv2d(self.prediction, self.gy_filter, padding=1)
        G_pred = torch.pow(G_x_pred,2)+ torch.pow(G_y_pred,2)

        G_x_gt = nn.functional.conv2d(self.gt_tensor, self.gx_filter, padding=1)
        G_y_gt = nn.functional.conv2d(self.gt_tensor, self.gy_filter, padding=1)
        G_gt = torch.pow(G_x_gt,2)+ torch.pow(G_y_gt,2)

        edge_diff_MSE = torch.abs(G_pred - G_gt)*(self.validity_tensor)
        edge_diff_mAE = torch.abs(G_x_pred - G_x_gt)*(self.validity_tensor) + \
                            torch.abs(G_y_pred - G_y_gt)*(self.validity_tensor)
        edge_loss = torch.mean(edge_diff_MSE) + torch.mean(edge_diff_mAE)

        return edge_loss
        
    # Run Model
    def run_model(self, epoch, dataset: Literal['URBANSYN', 'GTAV', 'MUAD', 'KITTI', 'DDAD']):     
        
        self.prediction = self.model(self.image_tensor)
        mAE_loss = self.mAE_validity_loss()
        
        total_loss = 0

        if(dataset == 'URBANSYN' or dataset == 'GTAV' or dataset == 'MUAD'):
            edge_loss = self.edge_validity_loss()
            total_loss = mAE_loss + edge_loss
        else:
            total_loss = mAE_loss
            
        self.calc_loss = total_loss

    # Loss Backward Pass
    def loss_backward(self): 
        self.calc_loss.backward()

    # Get loss value
    def get_loss(self):
        return self.calc_loss.item()

    # Run Optimizer
    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Set train mode
    def set_train_mode(self):
        self.model = self.model.train()

    # Set evaluation mode
    def set_eval_mode(self):
        self.model = self.model.eval()
    
    # Save predicted visualization
    def save_visualization(self, log_count):

        # Converting prediction output to visualization
        prediction_vis = self.prediction.squeeze(0).cpu().detach()
        prediction_vis = prediction_vis.permute(1, 2, 0)
        prediction_vis = prediction_vis.numpy()
  
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(self.image)
        axs[0].set_title('Image',fontweight ="bold") 
        axs[1].imshow(self.augmented)
        axs[1].set_title('Ground Truth',fontweight ="bold") 
        axs[2].imshow(prediction_vis)
        axs[2].set_title('Prediction',fontweight ="bold") 
        self.writer.add_figure('predictions vs. actuals', \
        fig, global_step=(log_count))
    
    # Load Image as Tensor
    def load_image_tensor(self, is_train):

        if(is_train):
            image_tensor = self.image_loader(self.image)
            image_tensor = image_tensor.unsqueeze(0)
            self.image_tensor = image_tensor.to(self.device)
        else:
            image_val_tensor = self.image_loader(self.image_val)
            image_val_tensor = image_val_tensor.unsqueeze(0)
            self.image_val_tensor = image_val_tensor.to(self.device)

    # Load Ground Truth as Tensor
    def load_gt_tensor(self, is_train):

        if(is_train):
            gt_tensor = torch.from_numpy(self.augmented)
            gt_tensor = gt_tensor.permute(2, 0, 1)
            gt_tensor = gt_tensor.unsqueeze(0)
            gt_tensor = gt_tensor.type(torch.FloatTensor)
            self.gt_tensor = gt_tensor.to(self.device)
        else:
            gt_val_tensor = torch.from_numpy(self.augmented_val)
            gt_val_tensor = gt_val_tensor.permute(2, 0, 1)
            gt_val_tensor = gt_val_tensor.unsqueeze(0)
            gt_val_tensor = gt_val_tensor.type(torch.FloatTensor)
            self.gt_val_tensor = gt_val_tensor.to(self.device)

    # Load Image as Tensor
    def load_validity_tensor(self, is_train):

        if(is_train):
            validity_tensor = self.validity_loader(self.validity)
            validity_tensor = validity_tensor.unsqueeze(0)
            self.validity_tensor = validity_tensor.to(self.device)
        else:
            validity_val_tensor = self.validity_loader(self.validity_val)
            validity_val_tensor = validity_val_tensor.unsqueeze(0)
            self.validity_val_tensor = validity_val_tensor.to(self.device)
    
    # Zero Gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Save Model
    def save_model(self, model_save_path):
        print('Saving model')
        torch.save(self.model.state_dict(), model_save_path)

    # Run Validation and calculate metrics
    def validate(self, image_val, gt_val, validity_val):

        # Set Data
        self.set_val_data(image_val, gt_val, validity_val)

        # Augmenting Image
        self.apply_augmentations(is_train=False)
        
        # Converting to tensor and loading
        self.load_data(is_train=False)

        # Running model
        output_val = self.model(self.image_val_tensor)

        # Calculate loss
        abs_diff = torch.abs(output_val - self.gt_val_tensor)*(self.validity_val_tensor)
        accuracy = torch.mean(abs_diff)
        accuracy_val = accuracy.detach().cpu().numpy()

        return accuracy_val

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')