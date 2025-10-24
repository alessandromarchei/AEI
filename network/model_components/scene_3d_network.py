from .pre_trained_backbone import PreTrainedBackbone
from .depth_context import DepthContext
from .scene_3d_neck import Scene3DNeck
from .scene_3d_head import Scene3DHead

import torch.nn as nn
import torch

class Scene3DNetwork(nn.Module):
    def __init__(self, pretrained, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        super(Scene3DNetwork, self).__init__()

        # Upstream blocks
        self.PreTrainedBackbone = PreTrainedBackbone(pretrained)

        # Depth Context
        self.DepthContext = DepthContext()

        # Neck
        self.DepthNeck = Scene3DNeck()

        # Depth Head
        self.SuperDepthHead = Scene3DHead()
    
        # Buffers for normalization. put inside the model to be saved/loaded with it, so everything is self-contained is inside the onnx
        mean = torch.tensor(mean, dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor(std , dtype=torch.float32).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)



    def forward(self, image):
        # assume input [N,3,H,W] con valori [0..1]
        image = (image - self.mean) / self.std

        features = self.PreTrainedBackbone(image)
        deep_features = features[4]
        context = self.DepthContext(deep_features)
        neck = self.DepthNeck(context, features)
        prediction = self.SuperDepthHead(neck, features)
        return prediction