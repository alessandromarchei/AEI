#! /usr/bin/env python3
import torch
import torch.nn as nn

class BEVPathContext(nn.Module):
    def __init__(self):
        super(BEVPathContext, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)
        self.avg_pool = nn.AvgPool2d(2, stride=2)

        # Feature convolution
        self.feature_conv = nn.Conv2d(1280, 1280, 3, 1, 1)

        # Context - MLP Layers
        self.context_layer_0 = nn.Linear(1280, 800)
        self.context_layer_1 = nn.Linear(800, 800)
        self.context_layer_2 = nn.Linear(800, 200)

        # Context - Extraction Layers
        self.context_layer_3 = nn.Conv2d(1, 128, 3, 1, 1)
        self.context_layer_4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.context_layer_5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.context_layer_6 = nn.Conv2d(512, 1280, 3, 1, 1)

        # Context - Decode layers
        self.context_layer_7 = nn.Linear(1280, 800)
        self.context_layer_8 = nn.Linear(800, 800)
     

    def forward(self, features):

        bev_features = self.avg_pool(features)
        bev_features = self.feature_conv(bev_features)
        
        # Pooling and averaging channel layers to get a single vector
        feature_vector = torch.mean(bev_features, dim = [2,3])

        # MLP
        c0 = self.context_layer_0(feature_vector)
        c0 = self.dropout(c0)
        c0 = self.GeLU(c0)
        c1 = self.context_layer_1(c0)
        c1 = self.dropout(c1)
        c1 = self.GeLU(c1)
        c2 = self.context_layer_2(c1)
        c2 = self.dropout(c2)
        c2 = self.sigmoid(c2)
        
        # Reshape
        c3 = c2.reshape([20, 10])
        c3 = c3.unsqueeze(0)
        c3 = c3.unsqueeze(0)
        
        # Context
        c4 = self.context_layer_3(c3)
        c4 = self.GeLU(c4)
        c5 = self.context_layer_4(c4)
        c5 = self.GeLU(c5)
        c6 = self.context_layer_5(c5)
        c6 = self.GeLU(c6)
        c7 = self.context_layer_6(c6)
        context = self.GeLU(c7)

        # Attention
        context = context*bev_features + bev_features

        # Context feature vector
        context_feature_vector = torch.max(context, dim = [2,3])

        # Decoding driving path related features
        path_features = self.context_layer_7(context_feature_vector)
        path_features = self.dropout(path_features)
        path_features = self.GeLU(path_features)
     
        path_features = self.context_layer_8(path_features)
        path_features = self.dropout(path_features)
        path_features = self.GeLU(path_features)

        return path_features   