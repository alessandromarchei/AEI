
import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
)))
from model_components.scene_seg_network import SceneSegNetwork
from model_components.ego_path_network import EgoPathNetwork
from data_utils.augmentations import Augmentations

class EgoPathTrainer():
    def __init__(self,  checkpoint_path = '', pretrained_checkpoint_path = '', is_pretrained = False):
        
        # Image and Ground Truth
        self.image = 0
        self.gt = 0

        # Size of Image
        self.height = 320
        self.width = 640

        # Tensors
        self.image_tensor = 0
        self.gt_tensor = 0
        
        # Model and Prediction
        self.model = 0
        self.prediction = 0

        # Losses
        self.loss = 0
        self.endpoint_loss = 0
        self.mid_point_loss = 0
        self.gradient_loss = 0
        self.control_points_loss = 0
        self.gradient_type = 'NUMERICAL'

        # Loss scale factors
        self.endpoint_loss_scale_factor = 1
        self.control_points_scale_factor = 1
        self.grad_scale_factor = 1
        self.mid_point_scale_factor = 2

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        if(is_pretrained):

            # Instantiate Model for validation or inference - load both pre-traiend SceneSeg and SuperDepth weights
            if(len(checkpoint_path) > 0):

                # Loading model with full pre-trained weights
                sceneSegNetwork = SceneSegNetwork()
                self.model = EgoPathNetwork(sceneSegNetwork)

                # If the model is also pre-trained then load the pre-trained downstream weights
                self.model.load_state_dict(torch.load \
                    (checkpoint_path, weights_only=True, map_location=self.device))
                print('Loading pre-trained model weights of EgoPath from a saved checkpoint file')
            else:
                raise ValueError('Please ensure EgoPath network weights are provided for downstream elements')
            
        else:

            # Instantiate Model for training - load pre-traiend SceneSeg weights only
            if(len(pretrained_checkpoint_path) > 0):
                
                # Loading SceneSeg pre-trained for upstream weights
                sceneSegNetwork = SceneSegNetwork()
                sceneSegNetwork.load_state_dict(torch.load \
                    (pretrained_checkpoint_path, weights_only=True, map_location=self.device))
                    
                # Loading model with pre-trained upstream weights
                self.model = EgoPathNetwork(sceneSegNetwork)
                print('Loading pre-trained backbone model weights only, EgoPath initialised with random weights')
            else:
                raise ValueError('Please ensure EgoPath network weights are provided for upstream elements')
        
       
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

    # Zero Gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Learning Rate adjustment
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    # Assign input variables
    def set_data(self, image, gt):
        self.image = image
        self.gt = gt

    # Image agumentations
    def apply_augmentations(self, is_train):
        if(is_train):
            # Augmenting Data for training
            augTrain = Augmentations(is_train=True, data_type='KEYPOINTS')
            augTrain.setDataKeypoints(self.image)
            self.image = augTrain.applyTransformKeypoint(self.image)
        else:
            # Augmenting Data for testing/validation
            augVal = Augmentations(is_train=False, data_type='KEYPOINTS')
            augVal.setDataKeypoints(self.image)
            self.image = augVal.applyTransformKeypoint(self.image)

    # Load Data as Pytorch Tensors
    def load_data(self):
        
        # Converting image to Pytorch Tensor
        image_tensor = self.image_loader(self.image)
        image_tensor = image_tensor.unsqueeze(0)
        self.image_tensor = image_tensor.to(self.device)

        # Converting keypoint list to Pytorch Tensor
        # List is in x0,y0,x1,y1,....xn, yn format
        gt_tensor = torch.from_numpy(self.gt)
        gt_tensor = gt_tensor.unsqueeze(0)
        self.gt_tensor = gt_tensor.to(self.device)
    
    # Run Model
    def run_model(self):
        self.prediction = self.model(self.image_tensor)
        self.loss = self.calc_loss(self.prediction, self.gt_tensor)

    # Calculate loss
    def calc_loss(self, prediction, ground_truth):

        # Endpoint loss - align the start and end control points of the Prediciton
        # vs Ground Truth Bezier Curves
        self.endpoint_loss = self.calc_endpoints_loss(prediction, ground_truth)

        # Mid-point loss - similar to the BezierLaneNet paper, this loss ensures that
        # points along the curve have small x and y deviation - also acts as a regulariation term
        self.mid_point_loss = self.calc_mid_points_loss(prediction, ground_truth)

        # Control Points loss - make sure Bezier control points match
        self.control_points_loss = self.calc_control_points_loss(prediction, ground_truth)

        # Gradient Loss - either NUMERICAL tangent angle calcualation or
        # ANALYTICAL derviative of bezier curve, this loss ensures the curve is 
        # smooth and acts as a regularization term
        if(self.gradient_type == 'NUMERICAL'):
            self.gradient_loss = self.calc_numerical_gradient_loss(prediction, ground_truth)
        elif(self.gradient_type == 'ANALYTICAL'):
            self.gradient_loss = self.calc_analytical_gradient_loss(prediction, ground_truth)

        # Total loss is sum of individual losses multiplied by scailng factors
        total_loss = self.gradient_loss*self.grad_scale_factor + \
            self.mid_point_loss*self.mid_point_scale_factor + \
            self.control_points_loss*self.control_points_scale_factor + \
            self.gradient_loss*self.grad_scale_factor
     
        return total_loss 
    

    # Set scale factors for losses
    def set_loss_scale_factors(self, endpoint_loss_scale_factor, 
            mid_point_scale_factor, grad_scale_factor, control_point_scale_factor):
        
        # Loss scale factors
        self.endpoint_loss_scale_factor = endpoint_loss_scale_factor
        self.grad_scale_factor = grad_scale_factor
        self.mid_point_scale_factor = mid_point_scale_factor
        self.control_points_scale_factor = control_point_scale_factor


    # Define whether we are using a NUMERICAL vs ANALYTICAL gradient loss
    def set_gradient_loss_type(self, type):

        if(type == 'NUMERICAL'):
            self.gradient_type = 'NUMERICAL'
        elif(type == 'ANALYTICAL'):
            self.gradient_type = 'ANALYTICAL'
        else:
            raise ValueError('Please specify either NUMERICAL or ANALYTICAL gradient loss as a string')
        
    # Calculate the error between Bezier Control Points
    def calc_control_points_loss(self, prediction, ground_truth):
        control_points_loss = torch.mean(torch.abs(prediction-ground_truth))
        return control_points_loss
    
    # Calculate the endpoints loss term 
    def calc_endpoints_loss(self, prediction, ground_truth):

        # Prediction Start Point (x,y)
        pred_x_start = prediction[0][0]
        pred_y_start = prediction[0][1]

        # Ground Truth Start Point (x,y)
        gt_x_start = ground_truth[0][0]
        gt_y_start = ground_truth[0][1]

        # Start Point mAE Loss
        start_point_mAE = torch.abs(pred_x_start - gt_x_start) + \
            torch.abs(pred_y_start - gt_y_start)
        
        # Prediction End Point (x,y)
        pred_x_end = prediction[0][-2]
        pred_y_end = prediction[0][-1]

        # Ground Truth End Point (x,y)
        gt_x_end = ground_truth[0][-2]
        gt_y_end = ground_truth[0][-1]

        # Start Point mAE Loss
        end_point_mAE = torch.abs(pred_x_end - gt_x_end) + \
            torch.abs(pred_y_end - gt_y_end)

        # Total End Point mAE Loss
        total_end_point_mAE = start_point_mAE + end_point_mAE
        return total_end_point_mAE
    
    # Evaluate the x,y coordinates of a bezier curve at a given t-parameter value
    def evaluate_bezier(self, bezier, t):
 
         # Throw an error if parameter t is out of boudns
         if(t < 0 or t > 1):
             raise ValueError('Please ensure t parameter is in the range [0,1]')
         
         # Evaluate cubic bezier curve for value of t in range [0,1]
         x = (1-t)*(1-t)*(1-t)*bezier[0][0] + 3*(1-t)*(1-t)*t*bezier[0][2] \
             + 3*(1-t)*t*t*bezier[0][4] + t*t*t*bezier[0][6]
         
         y = (1-t)*(1-t)*(1-t)*bezier[0][1] + 3*(1-t)*(1-t)*t*bezier[0][3] \
             + 3*(1-t)*t*t*bezier[0][5] + t*t*t*bezier[0][7]
         
         return x,y

    # Calculate the mid-points loss term - similar to the BezierLaneNet paper
    def calc_mid_points_loss(self, prediction, ground_truth):

        delta_sum = 0
        sample_count = 0

        # Sample the bezier curve for Prediction and Ground Truth
        # ignoring the start and end points
        for i in range(5, 100, 5):

            # Sample counter
            sample_count +=1

            # t-parameter
            t = i/100

            # Get x-values for prediction and ground truth
            x_pred, y_pred = self.evaluate_bezier(prediction, t)
            x_gt, y_gt = self.evaluate_bezier(ground_truth, t)

            # Find the difference in x-values and sum
            delta_val = torch.abs(x_pred - x_gt) + torch.abs(y_pred - y_gt)

            delta_sum = delta_sum + delta_val

        # Find average absolute x-deviation between mid-pionts
        mAE_mid_points_delta = delta_sum/sample_count
        return mAE_mid_points_delta

    # Calculate the numerical gradient loss term
    def calc_numerical_gradient_loss(self, prediction, ground_truth):
        
        # Running totals
        grad_sum = 0
        sample_count = 0

        # Sample the bezier curve for Prediction and Ground Truth
        # ignoring the start and end points
        sampling_rate = 5

        for i in range(0, 100, sampling_rate):

            # Sample counter
            sample_count +=1

            # t-parameter current
            t = i/100

            # t-parameter for next sampel
            t_next = (i+sampling_rate)/100

            # Get pair-wise samples for Prediction and Ground Truth
            x_pred, y_pred = self.evaluate_bezier(prediction, t)
            x_pred_next, y_pred_next = self.evaluate_bezier(prediction, t_next)
            x_gt, y_gt = self.evaluate_bezier(ground_truth, t)
            x_gt_next, y_gt_next = self.evaluate_bezier(ground_truth, t_next)

            # Calcualte difference in x,y values between consecutive 
            # pairs of points for Prediction and Ground Truth
            dxp = x_pred_next - x_pred + 1e-6
            dyp = y_pred_next - y_pred + 1e-6
            dxg = x_gt_next - x_gt + 1e-6
            dyg = y_gt_next - y_gt + 1e-6

            # Find tangent angle betwen consecutive pairs of points
            # for the Ground Truth and Prediction
            grad_g = torch.atan2(dyg, dxg)
            grad_p = torch.atan2(dyp, dxp)

            # Calculate the absoulte angle error and sum for all
            # samples
            grad_diff = torch.abs(grad_g - grad_p)
            grad_sum = grad_sum + grad_diff

        # Calcualte the mAE gradient error
        mAE_gradient = grad_sum/sample_count
        return mAE_gradient
    
    # Calculate the analytical gradient loss term
    def calc_analytical_gradient_loss(self, prediction, ground_truth):

        # Running totals
        grad_sum = 0
        sample_count = 0

        # Sample the bezier curve for Prediction and Ground Truth
        # ignoring the start and end points
        sampling_rate = 5

        for i in range(0, 100, sampling_rate):

            # Sample counter
            sample_count +=1

            # t-parameter current
            t = i/100

            # Partial Derivative for Predictions
            # dx/dt
            pred_x0 = prediction[0][0]
            pred_x1 = prediction[0][2]
            pred_x2 = prediction[0][4]
            pred_x3 = prediction[0][6]

            dx_dt_pred = 3*(1-t)*(1-t)(pred_x1 - pred_x0) + 6*(1-t)*(t)(pred_x2 - pred_x1) \
                        + 3*(t)*(t)(pred_x3 - pred_x2)
            
            # dy/dt
            pred_y0 = prediction[0][1]
            pred_y1 = prediction[0][3]
            pred_y2 = prediction[0][5]
            pred_y3 = prediction[0][7]

            dy_dt_pred = 3*(1-t)*(1-t)(pred_y1 - pred_y0) + 6*(1-t)*(t)(pred_y2 - pred_y1) \
                        + 3*(t)*(t)(pred_y3 - pred_y2)
            
            # Partial Derivative for Ground Truth
            # dx/dt
            gt_x0 = ground_truth[0][0]
            gt_x1 = ground_truth[0][2]
            gt_x2 = ground_truth[0][4]
            gt_x3 = ground_truth[0][6]

            dx_dt_gt = 3*(1-t)*(1-t)(gt_x1 - gt_x0) + 6*(1-t)*(t)(gt_x2 - gt_x1) \
                        + 3*(t)*(t)(gt_x3 - gt_x2)
            
            # dy/dt
            gt_y0 = ground_truth[0][1]
            gt_y1 = ground_truth[0][3]
            gt_y2 = ground_truth[0][5]
            gt_y3 = ground_truth[0][7]

            dy_dt_gt = 3*(1-t)*(1-t)(gt_y1 - gt_y0) + 6*(1-t)*(t)(gt_y2 - gt_y1) \
                        + 3*(t)*(t)(gt_y3 - gt_y2)
            
            derivative_error = torch.abs(dx_dt_pred - dx_dt_gt) \
                + torch.abs(dy_dt_pred - dy_dt_gt)
            
            grad_sum += derivative_error

        # Calcualte the mAE gradient error
        mAE_gradient = grad_sum/sample_count
        return mAE_gradient
        
    # Loss Backward Pass
    def loss_backward(self):
        self.loss.backward()

    # Get loss value
    def get_loss(self):
        return self.loss.item()

    # Get endpoint loss
    def get_endpoint_loss(self):
        scaled_endpoint_loss = self.endpoint_loss*self.endpoint_loss_scale_factor
        return scaled_endpoint_loss.item()

    # Get gradient loss
    def get_gradient_loss(self):
        scaled_grad_loss = self.gradient_loss*self.grad_scale_factor
        return scaled_grad_loss.item()
    
    # Get mid point loss
    def get_mid_point_loss(self):
        scaled_midpoint_loss = self.mid_point_loss*self.mid_point_scale_factor
        return scaled_midpoint_loss.item()
    
    # Get control point loss
    def get_control_point_loss(self):
        scaled_control_point_loss = self.control_points_loss*self.control_points_scale_factor
        return scaled_control_point_loss.item()

    # Logging Loss
    def log_loss(self, log_count):
        self.writer.add_scalars("Train",{
            'total_loss': self.get_loss(),
            'control_point_loss': self.get_control_point_loss(),
            'endpoint_loss': self.get_endpoint_loss(),
            'midpoint_loss': self.get_mid_point_loss(),
            'gradient_loss': self.get_gradient_loss()
        }, (log_count))

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

    # Save Model
    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')


    # Save predicted visualization
    def save_visualization(self, log_count):
        
        # Get the prediction and ground truth tensors and detach them
        pred_vis = self.prediction.cpu().detach().numpy()
        gt_vis = self.gt_tensor.cpu().detach().numpy()

        # Get a list of x points and y points for Ground Truth and Prediction
        pred_x_points = []
        pred_y_points = []
        gt_x_points = []
        gt_y_points = []

        for i in range(0, 110, 10):

            t = i/100
            
            pred_x, pred_y = self.evaluate_bezier(pred_vis, t)
            gt_x, gt_y = self.evaluate_bezier(gt_vis, t)

            # Applying scaling based on image size
            # to ensure data is correctly visualized
            pred_x_points.append(pred_x*self.width)
            pred_y_points.append(pred_y*self.height)
            gt_x_points.append(gt_x*self.width)
            gt_y_points.append(gt_y*self.height)
        

        # Visualize the Ground Truth
        fig_gt = plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.imshow(self.image)

        # Plot the final curve
        plt.plot(gt_x_points, gt_y_points, color="cyan")

        # Plot the control points
        # Start control point - RED
        plt.scatter((gt_vis[0][0]*self.width), (gt_vis[0][1]*self.height), \
                    marker='o', color='red', s=10)
        # Middle control points - Green
        plt.scatter((gt_vis[0][2]*self.width, gt_vis[0][4]*self.width), \
                    (gt_vis[0][3]*self.height, gt_vis[0][5]*self.height),
                    marker='o', color='lawngreen', s=10)
        # End control points - YELLOW
        plt.scatter((gt_vis[0][6]*self.width), (gt_vis[0][7]*self.height), \
                    marker='o', color='yellow', s=10)
        
        self.writer.add_figure('Ground Truth', \
            fig_gt, global_step=(log_count))

        # Visualize the Prediction

        fig_pred = plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.imshow(self.image)

        # Plot the final curve
        plt.plot(pred_x_points, pred_y_points, color="cyan")

        # Plot the control points
        # Start control point - RED
        plt.scatter((pred_vis[0][0]*self.width), (pred_vis[0][1]*self.height), \
                    marker='o', color='red', s=10)
        # Middle control points - GREEN
        plt.scatter((pred_vis[0][2]*self.width, pred_vis[0][4]*self.width), \
                    (pred_vis[0][3]*self.height, pred_vis[0][5]*self.height),
                    marker='o', color='lawngreen', s=10)
        # End control points - YELLOW
        plt.scatter((pred_vis[0][6]*self.width), (pred_vis[0][7]*self.height), \
                    marker='o', color='yellow', s=10)
        self.writer.add_figure('Prediction', \
            fig_pred, global_step=(log_count))


    # Run Validation and calculate metrics
    def validate(self, image, gt):

        # Set Data
        self.set_data(image, gt)

        # Augmenting Image
        self.apply_augmentations(is_train=False)

        # Converting to tensor and loading
        self.load_data()

        # Running model
        prediction = self.model(self.image_tensor)

        # Calculating validation loss as mAE between Ground Truth
        # and Predicted Bezier curve control points
        validation_loss_tensor = self.calc_control_points_loss(prediction, self.gt_tensor)
        validation_loss = validation_loss_tensor.detach().cpu().numpy()
       
        return validation_loss

    # Logging validation losses to Tensor Board
    def log_validation(self, log_count, bdd100k_val_score, comma2k19_val_score,
            culane_val_score, curvelanes_val_score, roadwork_val_score,
            tusimple_val_score, overall_validation_score):
        
        # Dataset specific validation scores
        self.writer.add_scalars("Val Score - Dataset",{
            'BDD100K': bdd100k_val_score,
            'COMMA2k19': comma2k19_val_score,
            'CULANE': culane_val_score,
            'CURVELANES': curvelanes_val_score,
            'ROADWORK': roadwork_val_score,
            'TUSIMPLE': tusimple_val_score,
        }, (log_count))
        
        # Overall validation score
        self.writer.add_scalar("Val Score - Overall", 
            overall_validation_score, (log_count))

    '''
    # Run network on test image and visualize result
    def test(self, image_test, save_path):
        
        # Read test image
        frame = cv2.imread(image_test, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get height and width of test image
        test_image_height, test_image_width, _ = frame.shape

        # Resize test image
        image_pil = Image.fromarray(frame)
        image_pil = image_pil.resize((640, 320))

        # Load test image as a Pytorch Tensor
        test_image_tensor = self.image_loader(image_pil)
        test_image_tensor = test_image_tensor.unsqueeze(0)
        test_image_tensor = test_image_tensor.to(self.device)

        # Run inference on the test image tensor
        test_output = self.model(test_image_tensor)

        # Detach output for visualization
        test_vis = test_output.cpu().detach().numpy()

        # Evaluate the curve shape based on prediction from test data
        test_pred_x_points = []
        test_pred_y_points = []

        for i in range(0, 110, 10):

            # t-parameter values
            t = i/100
            
            # Get x,y values based on t-parameter value
            test_pred_x, test_pred_y = self.evaluate_bezier(test_vis, t)
   
            # Applying scaling based on image size
            # to ensure data is correctly visualized
            test_pred_x_points.append(test_pred_x*test_image_width)
            test_pred_y_points.append(test_pred_y*test_image_height)
        
        # Visualize the Ground Truth
        fig_test = plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.imshow(frame)

        # Plot the final curve
        plt.plot(test_pred_x_points, test_pred_y_points, color="cyan")

        # Plot the control points
        # Start control point - RED
        plt.scatter((test_vis[0][0]*test_image_width), (test_vis[0][1]*test_image_height), \
                    marker='o', color='red', s=10)
        # Middle control points - Green
        plt.scatter((test_vis[0][2]*test_image_width, test_vis[0][4]*test_image_width), \
                    (test_vis[0][3]*test_image_height, test_vis[0][5]*test_image_height),
                    marker='o', color='lawngreen', s=10)
        # End control points - YELLOW
        plt.scatter((test_vis[0][6]*test_image_width), (test_vis[0][7]*test_image_height), \
                    marker='o', color='yellow', s=10)
        
        # Save the visualization to disk based on the save path
        fig_test.savefig(save_path)   
        plt.close(fig_test)    
        '''