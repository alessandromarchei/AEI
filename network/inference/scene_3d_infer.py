import torch
from torchvision import transforms
import sys
from network.model_components.scene_seg_network import SceneSegNetwork
from network.model_components.scene_3d_network import Scene3DNetwork

import onnxruntime as ort
import numpy as np

# TensorRT + PyCUDA
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA context

from PIL import Image

class Scene3DNetworkInfer():
    def __init__(self, checkpoint_path = ''):

        # Image loader
        self.image_loader = transforms.ToTensor()

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')
            
        # Instantiate model, load to device and set to evaluation mode
        sceneSegNetwork = SceneSegNetwork()
        self.model = Scene3DNetwork(sceneSegNetwork)

        if len(checkpoint_path) > 0:
            state = torch.load(checkpoint_path, map_location=self.device)
            # strict=False per ignorare i buffer mancanti
            self.model.load_state_dict(state, strict=False) #no contol on missing keys, since mean and std were added later
        else:
            raise ValueError("No path to checkpoint file provided in class initialization")
        
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def inference(self, image):

        width, height = image.size
        if(width != 640 or height != 320):
            raise ValueError('Incorrect input size - input image must have height of 320px and width of 640px')

        image_tensor = self.image_loader(image).unsqueeze(0).to(self.device)  # [1,3,320,640]

        # Run model
        prediction = self.model(image_tensor)

        # Get output, find max class probability and convert to numpy array
        prediction = prediction.squeeze(0).cpu().detach()
        prediction = prediction.permute(1, 2, 0)
        output = prediction.numpy()

        return output
        


# class Scene3DOnnxInfer:
#     def __init__(self, model_path):
#         # Providers: try GPU (CUDA), then fallback to CPU
#         providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#         self.session = ort.InferenceSession(model_path, providers=providers)

#         # Assume single input/output
#         self.input_name = self.session.get_inputs()[0].name
#         self.output_name = self.session.get_outputs()[0].name
    
#     def normalize_image(self, img):
#         #normalize image with (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) (image - mean) / std
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#         img = (img - mean) / std
#         return img


#     def preprocess_image(self, pil_image):
#         # Convert PIL → numpy → CHW float32/uint8 depending on model
#         img = np.array(pil_image).astype(np.float32) / 255.0

#         #normalize image with (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) (image - mean) / std
#         img = self.normalize_image(img)


#         img = np.transpose(img, (2, 0, 1))  # HWC → CHW
#         img = np.expand_dims(img, axis=0)   # NCHW
#         return img
    

#     def inference(self, pil_image):
#         # Convert PIL → numpy → CHW float32/uint8 depending on model. input is of shape N C H W

#         img = self.preprocess_image(pil_image).astype(np.float32)

#         # Run inference
#         outputs = self.session.run([self.output_name], {self.input_name: img})
#         #outpout shape is N C H W, with C = num_classes, here 3

#         #each pixel is now a vector of class probabilities
#         # [prob_class0, prob_class1, prob_class2]

#         # B C H W → C H W (C = num_classes)
#         prediction = outputs[0]              # numpy array, shape (1, C, H, W)
#         prediction = np.squeeze(prediction, 0)   # (C, H, W)

#         # Argmax over channels → segmentation mask (H, W), take the highest prob class for each pixel
#         output = np.argmax(prediction, axis=0).astype(np.uint8)

#         return output



class Scene3DOnnxInfer:
    def __init__(self, model_path):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Debug shapes/dtypes once
        print("[ONNX] input:",  self.session.get_inputs()[0].shape,  self.session.get_inputs()[0].type)
        print("[ONNX] output:", self.session.get_outputs()[0].shape, self.session.get_outputs()[0].type)

    def inference(self, pil_image: Image.Image):
        # PIL (RGB) -> float32 [0,1] HWC -> NCHW
        img = np.array(pil_image).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))         # HWC -> CHW
        img = np.expand_dims(img, axis=0)          # -> NCHW

        outputs = self.session.run([self.output_name], {self.input_name: img})
        out = outputs[0]

        # Squeeze all singleton dims
        out = np.squeeze(out)  # could become (H,W) or (C,H,W)
        # If channel-first, convert to (H,W,C)
        if out.ndim == 3:
            # assume C,H,W -> H,W,C
            out = np.transpose(out, (1, 2, 0))
        elif out.ndim == 2:
            # make it (H,W,1) to be consistent with PyTorch branch
            out = out[..., None]

        # keep float32; visualization will scale
        return out.astype(np.float32)


class Scene3DTrtInfer:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # assume one input, one output
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.stream = cuda.Stream()
        self.bindings = []
        self.host_inputs, self.device_inputs = [], []
        self.host_outputs, self.device_outputs = [], []

        for binding in self.engine:
            shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

    def inference(self, pil_image: Image.Image):
        # Input: PIL → [0..1] float32, NCHW
        img = np.array(pil_image).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)   # NCHW

        # Transfer to device
        cuda.memcpy_htod_async(self.device_inputs[0], img.ravel(), self.stream)

        # Bind
        self.context.set_tensor_address(self.input_name, int(self.device_inputs[0]))
        self.context.set_tensor_address(self.output_name, int(self.device_outputs[0]))

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy back
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.device_outputs[0], self.stream)
        self.stream.synchronize()

        # Output = 1 CHANNEL [1,H,W]
        mask = np.reshape(self.host_outputs[0], self.engine.get_tensor_shape(self.output_name))
        return np.squeeze(mask, 0).astype(np.uint8)
