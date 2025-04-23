
import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.ego_path_network import EgoPathNetwork
from data_utils.augmentations import Augmentations

class EgoPathTrainer():
    def __init__(self,  checkpoint_path = '', pretrained_checkpoint_path = '', is_pretrained = False):
        
        # Image and Ground Truth
        self.image = 0
        self.gt = 0

        # Tensors
        self.image_tensor = 0
        self.gt_tensor = 0
        
        # Model and Prediction
        self.model = 0
        self.prediction = 0

        # Losses
        self.loss = 0
        self.endpoint_loss = 0
        self.gradient_loss = 0
        self.loss_scale_factor = 1

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
                print('Loading pre-trained model weights of EgoPath and upstream SceneSeg weights as well')
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
                print('Loading pre-trained model weights of upstream SceneSeg only, EgoPath initialised with random weights')
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
            augTrain.setDataKeypoints(self.image, self.gt)
            self.image, self.gt = \
                augTrain.applyTransformKeypoint(self.image, self.gt)

        else:
            # Augmenting Data for testing/validation
            augVal = Augmentations(is_train=False, data_type='KEYPOINTS')
            augVal.setDataKeypoints(self.image, self.gt)
            self.image, self.gt = \
                augVal.applyTransformKeypoint(self.image, self.gt)

    # Run Model
    def run_model(self):
        self.prediction = self.model(self.image_tensor)
        self.loss = self.calc_loss(self.prediction, self.gt_tensor)
        print(self.loss.item())

    # Loss Backward Pass
    def loss_backward(self):
        self.loss.backward()

    # Get loss value
    def get_loss(self):
        return self.loss.item()

    # Get endpoint loss
    def get_endpoint_loss(self):
        return self.endpoint_loss.item()

    # Get gradient loss
    def get_gradient_loss(self):
        return self.gradient_loss.item()

    # Logging Loss
    def log_loss(self, log_count):
        self.writer.add_scalars("Train",{
            'total_loss': self.get_loss(),
            'endpoint_loss': self.get_endpoint_loss(),
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

    # Zero Gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Save Model
    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')

    def fit_cubic_bezier_numpy(self):

        # Chord length parameterization
        distances = np.sqrt(np.sum(np.diff(self.gt, axis=0)**2, axis=1))
        cumulative = np.insert(np.cumsum(distances), 0, 0)
        t = cumulative / cumulative[-1]

        # Bézier basis functions
        def bernstein_matrix(t):
            t = np.asarray(t)
            B = np.zeros((len(t), 4))
            B[:, 0] = (1 - t)**3
            B[:, 1] = 3 * (1 - t)**2 * t
            B[:, 2] = 3 * (1 - t) * t**2
            B[:, 3] = t**3
            return B

        B = bernstein_matrix(t)

        # Least squares fitting: B * P = points => P = (B^T B)^-1 B^T * points
        BTB = B.T @ B
        BTP = B.T @ self.gt
        control_points = np.linalg.solve(BTB, BTP)

        return control_points  # shape (4, 2)
    

    # Load Data
    def load_data(self):
        
        # Converting image to Pytorch Tensor
        image_tensor = self.image_loader(self.image)
        image_tensor = image_tensor.unsqueeze(0)
        self.image_tensor = image_tensor.to(self.device)

        # Fitting bezier curve to augmented ground truth data
        gt_tensor = self.fit_bezier(self.gt)
        self.gt_tensor = gt_tensor.to(self.device)

    def fit_bezier(self, drivable_path):
        """
        Fit a cubic Bezier curve to a list of (x,y) tuples using PyTorch.

        Parameters:
            drivable_path: list/array of (x,y) coordinates, or a torch.Tensor of shape (n,2).

        Returns:
            Tensor of shape (1,8) of control points [P0, P1, P2, P3].
        """
        # ensure float tensor
        pts = torch.as_tensor(drivable_path, dtype=torch.float32)

        # chord‐length parameterization
        diffs = pts[1:] - pts[:-1]                              # (n-1,2)
        seg_lengths = torch.norm(diffs, dim=1)                  # (n-1,)
        t = torch.cat([torch.zeros(1, dtype=seg_lengths.dtype), seg_lengths.cumsum(0)])  # (n,)
        t = t / t[-1]                                           # normalize to [0,1]

        P0, P3 = pts[0], pts[-1]

        T = t.unsqueeze(1)                                      # (n,1)
        A = 3 * (1 - T)**2 * T                                  # (n,1)
        B = 3 * (1 - T) * T**2                                  # (n,1)
        M = torch.cat([A, B], dim=1)                            # (n,2)

        # subtract fixed‐endpoint contributions
        rhs = pts \
            - ((1 - t)**3).unsqueeze(1) * P0 \
            - (t**3).unsqueeze(1) * P3                           # (n,2)

        # least‐squares for P1,P2 via pseudoinverse
        sol = torch.pinverse(M) @ rhs                           # (2,2)
        P1, P2 = sol[0], sol[1]

        # return torch.vstack([P0, P1, P2, P3])
        return torch.vstack([P0, P1, P2, P3]).view(-1).unsqueeze(0)


    def evaluate_bezier(self, ctrl_pts, t):
        """
        Evaluate a cubic Bezier at parameter value t.

        Parameters:
            ctrl_pts: Tensor (4,2).
            t : parameter in range 0 to 1

        Returns:
            x and y locations of bezier curve at parameter t
        """

        x = (1-t)*(1-t)*(1-t)*ctrl_pts[0][0] + 3*(1-t)*(1-t)*t*ctrl_pts[0][2] \
            + 3*(1-t)*t*t*ctrl_pts[0][4] + t*t*t*ctrl_pts[0][6]
        
        y = (1-t)*(1-t)*(1-t)*ctrl_pts[0][1] + 3*(1-t)*(1-t)*t*ctrl_pts[0][3] \
            + 3*(1-t)*t*t*ctrl_pts[0][5] + t*t*t*ctrl_pts[0][7]
        
        return x,y



    def bezier_analytic_derivative(self, ctrl_pts, t_vals):
        """
        Compute tangent of cubic Bezier at t_vals.

        Parameters:
            ctrl_pts: Tensor (4,2).
            t_vals:  array‐like or tensor of shape (m,) in [0,1].

        Returns:
            Tensor (m,2) of derivative vectors.
        """
        ctrl = torch.as_tensor(ctrl_pts, dtype=torch.float32)
        t = torch.as_tensor(t_vals, dtype=ctrl.dtype).view(-1, 1)       # (m,1)
        one_minus_t = (1 - t)                                           # (m,1)
        # Derivative: B'(t) = 3*(1-t)^2 (P1-P0) + 6*(1-t)*t (P2-P1) + 3*t^2 (P3-P2)
        term1 = (3 * one_minus_t**2) * (ctrl[1] - ctrl[0])
        term2 = (6 * one_minus_t * t) * (ctrl[2] - ctrl[1])
        term3 = (3 * t**2) * (ctrl[3] - ctrl[2])

        return term1 + term2 + term3


    def calc_endpoint_loss(self, pred_ctrl_pts, gt_ctrl_pts):
        """
        Endpoint loss = sum of absolute differences at P0 and P3.
        """
        diff_P0 = torch.abs(pred_ctrl_pts[0][0] - gt_ctrl_pts[0][0]) + \
            torch.abs(pred_ctrl_pts[0][1] - gt_ctrl_pts[0][1])
        
        diff_P3 = torch.abs(pred_ctrl_pts[0][6] - gt_ctrl_pts[0][6]) + \
            torch.abs(pred_ctrl_pts[0][7] - gt_ctrl_pts[0][7])
        #print(pred_ctrl_pts[0][0].item(), gt_ctrl_pts[0][0].item())
        #print(diff_P0.item(), diff_P3.item())
        return diff_P0 + diff_P3



    def calc_analytic_gradient_loss(self, pred_ctrl_pts, gt_ctrl_pts, num_samples=25):
        """
        Gradient loss = mean absolute difference of derivatives over num_samples.
        """
        p = torch.as_tensor(pred_ctrl_pts, dtype=torch.float32)
        g = torch.as_tensor(gt_ctrl_pts,   dtype=torch.float32)
        t = torch.linspace(0, 1, num_samples, dtype=p.dtype)

        dp = self.bezier_analytic_derivative(p, t)
        dg = self.bezier_analytic_derivative(g, t)
        return torch.mean(torch.abs(dp - dg))


    def calc_numerical_gradient_loss(self, pred_ctrl_pts, gt_ctrl_pts, step=4):
        """
        Numeric gradient loss via finite differences:
        mean absolute error between arctan slopes of GT vs. pred Bézier,
        sampled every `step/100` in t ∈ (0,1).
        """

        # sample points t_i = 4/100, 8/100, …, 96/100
        idx      = torch.arange(step, 100+step, step, dtype=pred_ctrl_pts.dtype)
        t        = idx / 100.0
        t_prev   = (idx - step) / 100.0
        
        num_samples = len(t)
        grad_sum = 0

        for i in range(0, num_samples):
        
            # evaluate both curves at t and t_prev
            xp, yp   = self.evaluate_bezier(pred_ctrl_pts, t[i])
            xp0, yp0 = self.evaluate_bezier(pred_ctrl_pts, t_prev[i])
            xg, yg   = self.evaluate_bezier(gt_ctrl_pts, t[i])
            xg0, yg0 = self.evaluate_bezier(gt_ctrl_pts, t_prev[i])

            # finite‐difference slopes (add tiny eps to avoid div0)
            dxg = xg  - xg0 + 1e-4
            dyg = yg  - yg0 + 1e-4
            dxp = xp  - xp0 + 1e-4
            dyp = yp  - yp0 + 1e-4

            grad_g = torch.atan(dxg / dyg)
            grad_p = torch.atan(dxp / dyp)
            grad_diff = torch.abs(grad_g - grad_p)
            grad_sum = grad_sum + grad_diff

        return grad_sum/num_samples


    def calc_loss(self, pred_ctrl_pts, gt_ctrl_pts):
        """
        Combined loss = alpha * endpoint + beta * gradient.
        """
        self.endpoint_loss = self.calc_endpoint_loss(pred_ctrl_pts, gt_ctrl_pts)
        self.gradient_loss = self.calc_numerical_gradient_loss(pred_ctrl_pts, gt_ctrl_pts)
        return (self.gradient_loss*self.loss_scale_factor) + self.endpoint_loss


    # Save predicted visualization
    def save_visualization(self, log_count):
        # 1) fit and sample
        num_samples=100
        ctrl_pts   = self.fit_bezier(self.gt)                                   # (8) Tensor
        ctrl_pts = ctrl_pts.view(4,2)
        t_vals     = torch.linspace(0, 1, num_samples, dtype=ctrl_pts.dtype)           # (num_samples,)
        curve_pts  = self.evaluate_bezier(ctrl_pts, t_vals)                            # (num_samples,2)

        # 2) to numpy arrays for plotting
        pts   = self.gt.cpu().numpy()
        curve = curve_pts.cpu().numpy()
        ctrl  = ctrl_pts.cpu().numpy()

        # 3) plot
        fig = plt.figure(figsize=(8, 4))
        plt.imshow(img)
        plt.plot(pts[:,0],   pts[:,1],   marker='o', linestyle='--')  # waypoints
        plt.plot(curve[:,0], curve[:,1])                              # Bézier curve
        plt.scatter(ctrl[:,0], ctrl[:,1], marker='x', s=100)          # control points
        plt.axis('off')
        self.writer.add_figure('predictions vs. actuals', \
        fig, global_step=(log_count))


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
        gt_ctrl_pts = self.fit_bezier(self.gt)
        self.loss = self.calc_loss(prediction, gt_ctrl_pts)

        gradient_loss = self.gradient_loss.detach().cpu().numpy()
        endpoint_loss = self.endpoint_loss.detach().cpu().numpy()

        return gradient_loss, endpoint_loss

    # Run network on test image and visualize result
    # TODO
    def test(self, image_test, save_path):

        frame = cv2.imread(image_test, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame)
        image_pil = image_pil.resize((640, 320))

        test_image_tensor = self.image_loader(image_pil)
        test_image_tensor = test_image_tensor.unsqueeze(0)
        test_image_tensor = test_image_tensor.to(self.device)
        test_output = self.model(test_image_tensor)

        test_output = test_output.squeeze(0).cpu().detach()
        test_output = test_output.permute(1, 2, 0)
        test_output = test_output.numpy()
        test_output = cv2.resize(test_output, (frame.shape[1], frame.shape[0]))

        plt.imsave(save_path, test_output, cmap='viridis')
