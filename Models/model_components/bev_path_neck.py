#! /usr/bin/env python3
import torch
import torch.nn as nn

class BEVPathNeck(nn.Module):
    def __init__(self):
        super(BEVPathNeck, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.avg_pool = nn.AvgPool2d(2, stride=2)

        # Context - Extraction Layers
        self.fusion_layer_0 = nn.Conv2d(32, 1280, 1)
        self.fusion_layer_1 = nn.Conv2d(24, 1280, 1)
        self.fusion_layer_2 = nn.Conv2d(40, 1280, 1)
        self.fusion_layer_3 = nn.Conv2d(80, 1280, 1)
     

    def forward(self, features):
        
        feature_0 = features[0]
        feature_0 = self.avg_pool(feature_0)
        feature_0 = self.avg_pool(feature_0)
        feature_0 = self.avg_pool(feature_0)
        feature_0 = self.avg_pool(feature_0)
        feature_0 = self.fusion_layer_0(feature_0)

        feature_1 = features[1]
        feature_1 = self.avg_pool(feature_1)
        feature_1 = self.avg_pool(feature_1)
        feature_1 = self.avg_pool(feature_1)
        feature_1 = self.fusion_layer_1(feature_1)

        feature_2 = features[2]
        feature_2 = self.avg_pool(feature_2)
        feature_2 = self.avg_pool(feature_2)
        feature_2 = self.fusion_layer_2(feature_2)

        feature_3 = features[3]
        feature_3 = self.avg_pool(feature_3)   
        feature_3 = self.fusion_layer_3(feature_3) 

        feature_4 = features[4]

        deep_features = feature_0 + feature_1 + feature_2 + \
                        feature_3 + feature_4
        
        return deep_features