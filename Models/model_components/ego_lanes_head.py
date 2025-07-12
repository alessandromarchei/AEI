#! /usr/bin/env python3
import torch
import torch.nn as nn

class EgoLanesHead(nn.Module):
    def __init__(self):
        super(EgoLanesHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.dropout = nn.Dropout(p=0.25)

        # Context - MLP Layers
        self.ego_path_layer_0 = nn.Linear(800, 200)
        self.ego_path_layer_1 = nn.Linear(200, 33)
        self.ego_path_layer_2 = nn.Linear(200, 33)
 

    def forward(self, features):
        # Pooling and averaging channel layers to get a single vector
        feature_vector = torch.mean(features, dim = [2,3])

        # MLP
        p0 = self.ego_path_layer_0(feature_vector)
        p0 = self.GeLU(p0)
        ego_left_lane = self.ego_path_layer_1(p0)
        ego_right_lane = self.ego_path_layer_2(p0)

        # Final result
        return ego_left_lane, ego_right_lane