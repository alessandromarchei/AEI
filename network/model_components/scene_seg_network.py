from .backbone import Backbone
from .scene_context import SceneContext
from .scene_neck import SceneNeck
from .scene_seg_head import SceneSegHead
import torch.nn as nn
import torch


class SceneSegNetwork(nn.Module):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        super(SceneSegNetwork,self).__init__()
        self.Backbone = Backbone()
        self.SceneContext = SceneContext()
        self.SceneNeck = SceneNeck()
        self.SceneSegHead = SceneSegHead()

        # Buffers for normalization. put inside the model to be saved/loaded with it, so everything is self-contained is inside the onnx
        mean = torch.tensor(mean).view(1,3,1,1)  # shape [1,C,1,1]
        std = torch.tensor(std).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, image):
        # assume input [N,3,H,W] con valori [0..1]
        x = (image - self.mean) / self.std

        # forward nella rete
        features = self.Backbone(x)
        deep_features = features[4]
        context = self.SceneContext(deep_features)
        neck = self.SceneNeck(context, features)
        logits = self.SceneSegHead(neck, features)  # [N,C,H,W]

        # postprocess: argmax per pixel
        pred = torch.argmax(logits, dim=1).to(torch.uint8)  # [N,H,W]

        return pred
