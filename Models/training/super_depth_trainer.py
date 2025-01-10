
import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.super_depth_network import SuperDepthNetwork
from data_utils.augmentations import Augmentations

class SuperDepthTrainer():
    def __init__(self,  checkpoint_path = '', pretrained_checkpoint_path = '', is_pretrained = False):

        self.image = 0
        self.image_val = 0
        self.gt = 0
        self.gt_val = 0
        self.validity = 0
        self.validity_val = 0
        self.validity_tensor = 0
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
                self.model = SuperDepthNetwork(sceneSegNetwork)

                # If the model is also pre-trained then load the pre-trained downstream weights
                self.model.load_state_dict(torch.load \
                    (checkpoint_path, weights_only=True, map_location=self.device))
                print('Loading pre-trained model weights of SuperDepth and upstream SceneSeg weights as well')
            else:
                raise ValueError('Please ensure SuperDepth network weights are provided for downstream elements')
            
        else:

            # Instantiate Model for training - load pre-traiend SceneSeg weights only
            if(len(pretrained_checkpoint_path) > 0):
                
                # Loading SceneSeg pre-trained for upstream weights
                sceneSegNetwork = SceneSegNetwork()
                sceneSegNetwork.load_state_dict(torch.load \
                    (pretrained_checkpoint_path, weights_only=True, map_location=self.device))
                    
                # Loading model with pre-trained upstream weights
                self.model = SuperDepthNetwork(sceneSegNetwork)
                print('Loading pre-trained model weights of upstream SceneSeg only, SuperDepth initialised with random weights')
            else:
                raise ValueError('Please ensure SceneSeg network weights are provided for upstream elements')
        
       
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

    # Logging Training Loss
    def log_loss(self, log_count):
        print('Logging Training Loss', log_count, self.get_loss())
        self.writer.add_scalar("Loss/train", self.get_loss(), (log_count))

    # Logging Validation mAE
    def log_val_mAE(self, mAE, log_count):
        print('Logging Validation')      
        self.writer.add_scalar("Val/mAE", mAE, (log_count))
        

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
            self.image, self.augmented, self.validity = \
                augTrain.applyTransformDepth(image=self.image, 
                                             ground_truth=self.gt, validity=self.validity)
        else:
            # Augmenting Data for testing/validation
            augVal = Augmentations(is_train=False, data_type='DEPTH')
            
            self.image_val, self.augmented_val, self.validity_val = \
                augVal.applyTransformDepth(image=self.image_val, 
                                           ground_truth=self.gt_val, validity=self.validity_val)
    
    # Load Data
    def load_data(self, is_train):
        self.load_image_tensor(is_train)
        self.load_gt_tensor(is_train)


    # Run Model
    def run_model(self):     
        self.loss = nn.L1Loss()
        self.prediction = self.model(self.image_tensor)
        self.calc_loss = self.loss(self.prediction, self.gt_tensor)

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

    # Normalize ground truth value for visualization
    def shift_height(self, height):
        height = height + np.min(height)
        return height
    
    # Save predicted visualization
    def save_visualization(self, log_count):

        print('Saving Visualization')

        # Converting prediction output to visualization
        prediction_vis = self.prediction.squeeze(0).cpu().detach()
        prediction_vis = prediction_vis.permute(1, 2, 0)
        prediction_vis = prediction_vis.numpy()
        prediction_vis = self.shift_height(prediction_vis)

        # Normalizing ground truth height to same range as predicition
        augmented_vis = self.shift_height(self.augmented)/7

        fig, axs = plt.subplots(1,3)
        axs[0].imshow(self.image)
        axs[0].set_title('Image',fontweight ="bold") 
        axs[1].imshow(augmented_vis)
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
            gt_tensor = torch.div(gt_tensor, 7)
            gt_tensor = gt_tensor.unsqueeze(0)
            self.gt_tensor = gt_tensor.to(self.device)
        else:
            gt_val_tensor = torch.from_numpy(self.augmented_val)
            gt_val_tensor = gt_val_tensor.permute(2, 0, 1)
            gt_val_tensor = torch.div(gt_val_tensor, 7)
            gt_val_tensor = gt_val_tensor.unsqueeze(0)
            self.gt_val_tensor = gt_val_tensor.to(self.device)
    
    # Zero Gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Save Model
    def save_model(self, model_save_path):
        print('Saving model')
        torch.save(self.model.state_dict(), model_save_path)
    
    # Run Validation and calculate metrics
    def validate(self, image_val, gt_val):

        # Set Data
        self.set_val_data(image_val, gt_val)

        # Augmenting Image
        self.apply_augmentations(is_train=False)
        
        # Converting to tensor and loading
        self.load_data(is_train=False)

        # Running model
        output_val = self.model(self.image_val_tensor)

        # Conversions
        output_val = output_val.squeeze(0).cpu().detach()
        output_val = output_val.permute(1, 2, 0)
        output_val = output_val.numpy()

        ground_truth_val = self.gt_val_tensor.squeeze(0).cpu().detach()
        ground_truth_val = ground_truth_val.permute(1, 2, 0)
        ground_truth_val = ground_truth_val.numpy()

        # Calculating mean absolute normalized error
        accuracy = np.average(np.abs(ground_truth_val - output_val))
        return accuracy

    def test(self, image_val, gt_val, validity):
        
        # Set Data
        self.set_val_data(image_val, gt_val)

        # Augmenting Image
        self.apply_augmentations(is_train=False)
        
        # Converting to tensor and loading
        self.load_data(is_train=False)

        # Running model
        output_val = self.model(self.image_val_tensor)

        # Conversions
        output_val = output_val.squeeze(0).cpu().detach()
        output_val = output_val.permute(1, 2, 0)
        output_val = output_val.numpy()

        ground_truth_val = self.gt_val_tensor.squeeze(0).cpu().detach()
        ground_truth_val = ground_truth_val.permute(1, 2, 0)
        ground_truth_val = ground_truth_val.numpy()

        # Calculating absolute normalized error
        abs_diff = np.abs(ground_truth_val - output_val)

        # Validity mask
        validity[validity!=0] = 1
        validity = validity.astype('float32')
        validity = np.expand_dims(validity, axis=-1)
        height = abs_diff.shape[0]
        width = abs_diff.shape[1]
        validity = cv2.resize(validity, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
        
        # Apply validity mask to get only valid data
        valid_data = abs_diff*validity
        accuracy = np.average(valid_data)
        return accuracy

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')