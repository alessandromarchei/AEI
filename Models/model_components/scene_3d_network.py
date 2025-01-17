from .depth_neck import DepthNeck
from .scene_3d_head import Scene3DHead
from .scene_3d_upstream import Scene3DUpstream
import torch.nn as nn

class Scene3DNetwork(nn.Module):
    def __init__(self, pretrained):
        super(Scene3DNetwork, self).__init__()

        # Upstream blocks
        self.SuperDepthUpstream = Scene3DUpstream(pretrained)

        # Neck
        self.DepthNeck = DepthNeck()

        # Depth Head
        self.SuperDepthHead = Scene3DHead()
    

    def forward(self, image):
        features, context = self.SuperDepthUpstream(image)
        neck = self.DepthNeck(context, features)
        prediction = self.SuperDepthHead(neck, features)
        return prediction