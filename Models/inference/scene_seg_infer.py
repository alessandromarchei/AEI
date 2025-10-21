#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
from torchvision import transforms
import sys
from Models.model_components.scene_seg_network import SceneSegNetwork
import onnxruntime as ort
import numpy as np

class SceneSegNetworkInfer():
    def __init__(self, checkpoint_path = ''):

        # Image loader
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
            
        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')
            
        # Instantiate model, load to device and set to evaluation mode
        self.model = SceneSegNetwork()

        if(len(checkpoint_path) > 0):
            self.model.load_state_dict(torch.load \
                (checkpoint_path, weights_only=True, map_location=self.device))
        else:
            raise ValueError('No path to checkpiont file provided in class initialization')
        
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def inference(self, image):

        width, height = image.size
        if(width != 640 or height != 320):
            raise ValueError('Incorrect input size - input image must have height of 320px and width of 640px')

        #normalize INPUT image and convert to tensor
        image_tensor = self.image_loader(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
    
        # Run model
        prediction = self.model(image_tensor)

        # Get output, find max class probability and convert to numpy array
        prediction = prediction.squeeze(0).cpu().detach()
        prediction = prediction.permute(1, 2, 0)
        _, output = torch.max(prediction, dim=2)
        output = output.numpy()

        return output
        


class SceneSegOnnxInfer:
    def __init__(self, model_path):
        # Providers: try GPU (CUDA), then fallback to CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Assume single input/output
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def normalize_image(self, img):
        #normalize image with (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) (image - mean) / std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        return img


    def preprocess_image(self, pil_image):
        # Convert PIL → numpy → CHW float32/uint8 depending on model
        img = np.array(pil_image).astype(np.float32) / 255.0

        #normalize image with (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) (image - mean) / std
        img = self.normalize_image(img)


        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)   # NCHW
        return img
    

    def inference(self, pil_image):
        # Convert PIL → numpy → CHW float32/uint8 depending on model. input is of shape N C H W

        img = self.preprocess_image(pil_image).astype(np.float32)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: img})
        #outpout shape is N C H W, with C = num_classes, here 3

        #each pixel is now a vector of class probabilities
        # [prob_class0, prob_class1, prob_class2]

        # B C H W → C H W (C = num_classes)
        prediction = outputs[0]              # numpy array, shape (1, C, H, W)
        prediction = np.squeeze(prediction, 0)   # (C, H, W)

        # Argmax over channels → segmentation mask (H, W), take the highest prob class for each pixel
        output = np.argmax(prediction, axis=0).astype(np.uint8)

        return output